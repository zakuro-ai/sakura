"""risk_sim: supply-chain Value-at-Risk (task ``risk_simulation``, model ``scvar-copula``).

Nothing is trained. ``train`` runs the four estimators of ``sakura.atelier.risk`` (historical, delta-normal,
peaks-over-threshold EVT and the Student-t copula Monte Carlo) on the dataset's three tables and records
the headline numbers as metrics. ``export`` (format ``json``) writes ``results.json``: every estimate, the
run manifest (input digests, hyper-parameters, seed, library versions, CPU) and ``results_sha256``.

``runtime_eval`` is the audit replay: it recomputes everything from the delivered ``results.json``'s own
recorded hyper-parameters and the materialised data, and scores ``replay_match`` = 1.0 when the digest is
identical, 0.0 when not (``reason`` then says which numbers moved). Bit-exact holds for the same inputs,
library versions and CPU family; ``results.json`` records all three so a mismatch can be explained.

``predict`` is the what-if: ``{"spend": {factor: amount}, "alpha": 0.99, "iterations": 20000}`` returns the
Monte Carlo VaR with that exposure, in well under a second at that size.

CPU only; numpy, scipy, pandas and pyarrow (``docs/atelier/envs/risk_sim.txt``), imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

from sakura.atelier import risk
from sakura.atelier.types import (Artifact, Checkpoint, MaterialisedData, MetricsSink, ResolvedSpec, RunDir,
                                  RuntimeEval, TrainResult)
from sakura.atelier.wrap_up import max_steps_of

RESULTS = "results.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> dict[str, str]:
    import numpy
    import pandas
    import scipy

    return {"numpy": numpy.__version__, "scipy": scipy.__version__, "pandas": pandas.__version__,
            "python": platform.python_version()}


def _cpu() -> dict[str, str]:
    """What a bit-exact replay depends on beyond the versions: the CPU family (libm and BLAS kernels
    differ between vendors)."""
    vendor = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith(("vendor_id", "CPU implementer")):
                vendor = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"machine": platform.machine(), "system": platform.system(),
            "vendor": vendor or platform.processor() or "unknown"}


def _metrics(results: dict[str, Any]) -> dict[str, float]:
    """Headline numbers: net VaR per method at each alpha, plus the Monte Carlo gross VaR and ES."""
    out: dict[str, float] = {}
    for a in results["alphas"]:
        k = risk._key(a)
        pct = f"{a * 100:g}".replace(".", "_")
        for method in ("historical", "delta_normal", "evt", "monte_carlo"):
            out[f"{method}_net_var_{pct}"] = float(results[method][k]["net"])
        mc = results["monte_carlo"][k]
        out[f"monte_carlo_gross_var_{pct}"] = float(mc["gross"])
        out[f"monte_carlo_net_es_{pct}"] = float(mc["net_es"])
        out[f"monte_carlo_net_se_{pct}"] = float(mc["net_se"])
    return out


def _evaluation(results: dict[str, Any]) -> dict[str, Any]:
    """The report's ``evaluation`` block: one ``splits`` row per method (net VaR and ES at each alpha), and
    the warnings a reader should not miss."""
    splits: dict[str, dict[str, Any]] = {}
    for method in ("historical", "delta_normal", "evt", "monte_carlo"):
        row: dict[str, Any] = {"n": results[method].get("n") or results["monte_carlo"]["runs"][-1]["iterations"]}
        for a in results["alphas"]:
            r = results[method][risk._key(a)]
            pct = f"{a * 100:g}"
            row[f"net_var_{pct}"] = r["net"]
            row[f"net_es_{pct}"] = r.get("net_es")
        splits[method] = row
    warnings = []
    top = risk._key(max(results["alphas"]))
    dn, mc = results["delta_normal"][top]["net"], results["monte_carlo"][top]["net"]
    if mc > 0 and dn < 0.8 * mc:
        warnings.append(f"delta-normal net VaR at {max(results['alphas'])} is {100 * (1 - dn / mc):.0f}% below "
                        "the Monte Carlo estimate: the normal fit misses the joint tail")
    if results["evt"]["gross_fit"]["n_excess"] < 30:
        warnings.append(f"EVT fitted on {results['evt']['gross_fit']['n_excess']} excesses only: its tail "
                        "extrapolation is uncertain")
    first, last = results["monte_carlo"]["runs"][0], results["monte_carlo"]["runs"][-1]
    if first is not last:
        drift = abs(first[top]["net"] - last[top]["net"])
        if drift > 3 * last[top]["net_se"]:
            warnings.append(f"Monte Carlo net VaR moved {drift:.3g} between {first['iterations']} and "
                            f"{last['iterations']} draws: the smaller run has not converged")
    convergence = [{"iterations": run["iterations"], "net_var": run[top]["net"], "net_se": run[top]["net_se"],
                    "alpha": max(results["alphas"])} for run in results["monte_carlo"]["runs"]]
    return {"splits": splits, "warnings": warnings, "convergence": convergence}


#: Draws a Try-it request uses: well under a second on a CPU, a fraction of a percent off the full run.
TRY_ITERATIONS = 20_000


def try_example(results: dict[str, Any], exposure: Any) -> dict[str, Any]:
    """A ready-to-send ``predict`` request built from this run's own data, so a person trying the model
    does not have to guess the input shape or the factor names: the largest exposure raised by 20 %, at the
    run's top confidence level. ``evaluation.try_example`` in the report; the hub pre-fills Try it with it."""
    spend = exposure.groupby(exposure["factor"].astype(str))["spend"].sum()
    top = str(spend.idxmax())
    return {"spend": {top: round(float(spend[top]) * 1.2, 2)}, "alpha": max(results["alphas"]),
            "iterations": TRY_ITERATIONS}


class RiskSimBackend:
    name = "risk_sim"
    tasks = frozenset({"risk_simulation"})
    #: No torch here; the engine's seeding of numpy's global RNG is irrelevant (one explicit Generator).
    deterministic_algorithms = False

    def train(self, spec: ResolvedSpec, data: MaterialisedData, out: RunDir, metrics: MetricsSink,
              resume: Checkpoint | None) -> TrainResult:
        t0 = time.time()
        # limits.max_steps: nothing is trained, the run's length is its Monte Carlo sample sizes. Each is
        # capped at max_steps, but never below the 1000 draws `risk_options` accepts; equal sizes collapse
        # into one. The capped ladder is what the results manifest records, so `runtime_eval` and the
        # replay read the iterations that really ran.
        hp, cap, limits_applied = spec.hyperparameters, max_steps_of(spec), None
        if cap is not None:
            planned = [int(n) for n in hp.get("iterations", risk.RiskOptions.iterations)]
            capped = list(dict.fromkeys(min(n, max(cap, risk.MIN_ITERATIONS)) for n in planned))
            if capped != planned:
                hp, limits_applied = {**hp, "iterations": capped}, {"max_steps": cap}
        opt = risk.risk_options(hp, spec.spec.seed)
        results = risk.run_all(data.train, opt)
        # Nothing here is training, so nothing goes to `metrics` (metrics.jsonl): the runner reads every
        # point there as an epoch and the hub draws it as a training curve, which a risk run has not got.
        # The Monte Carlo convergence ladder is reported in `evaluation.convergence` instead.
        manifest: dict[str, Any] = {
            "data_sha256": data.manifest_sha256,
            # Fingerprints and sizes only: a runner's local paths mean nothing once the file is delivered.
            "tables": {name: {k: v for k, v in entry.items() if k != "path"}
                       for name, entry in data.extra.get("risk_tables", {}).items()},
            "hyperparameters": hp,
            "seed": spec.spec.seed,
        }
        digest = risk.results_digest(results)
        versions, cpu = _versions(), _cpu()
        manifest["versions"], manifest["cpu"] = versions, cpu
        # The inputs travel with the results: `predict` and the replay need nothing but this one file.
        embedded = risk.embed_tables(data.train)
        manifest["inputs_sha256"] = risk.inputs_digest(embedded)
        model = {"results": results, "manifest": manifest, "results_sha256": digest, "inputs": embedded}
        return TrainResult(model=model, metrics=_metrics(results), train_seconds=time.time() - t0,
                           provenance={"results_sha256": digest, **versions, "cpu_vendor": cpu["vendor"]},
                           evaluation={**_evaluation(results),
                                       "try_example": try_example(results, data.train["exposure"])},
                           limits_applied=limits_applied)

    def export(self, result: TrainResult, fmt: str, out: RunDir) -> Artifact:
        if fmt != "json":
            raise ValueError(f"risk_simulation exports json only, not {fmt!r}")
        out.artifacts.mkdir(parents=True, exist_ok=True)
        path = out.artifacts / RESULTS
        path.write_text(json.dumps(result.model, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
        return Artifact(path=path, format="json", sha256=_sha256(path), bytes=path.stat().st_size)

    def runtime_eval(self, artifact: Artifact, data: MaterialisedData) -> RuntimeEval:
        """Replay, from the delivered file alone: its embedded inputs must hash to `inputs_sha256` and equal
        the data this run materialised, its numbers must hash to `results_sha256`, and recomputing from those
        inputs, hyper-parameters and seed must reproduce that digest."""
        if artifact.format != "json":  # a file delivered beside the results (the licence): nothing to replay
            return RuntimeEval(format=artifact.format, runtime="none", metric="replay_match", value=None, n=0,
                               matches_training=None, reason=f"{artifact.format} is not a risk result")
        delivered = json.loads(artifact.path.read_text(encoding="utf-8"))
        manifest = delivered["manifest"]
        n = sum(int(x) for x in manifest["hyperparameters"].get("iterations", risk.RiskOptions.iterations))
        if manifest["data_sha256"] != data.manifest_sha256:
            return RuntimeEval(format="json", runtime="numpy", metric="replay_match", value=0.0, n=n,
                               matches_training=False,
                               reason=f"data differs: delivered {manifest['data_sha256'][:12]}, "
                                      f"replayed on {data.manifest_sha256[:12]}")
        # The numbers delivered must be the numbers digested: an edited result with its old digest left in
        # place is caught here, before anything is recomputed.
        if risk.results_digest(delivered["results"]) != delivered["results_sha256"]:
            return RuntimeEval(format="json", runtime="numpy", metric="replay_match", value=0.0, n=n,
                               matches_training=False,
                               reason="results.json was altered: its numbers do not hash to its results_sha256")
        embedded = delivered["inputs"]
        if risk.inputs_digest(embedded) != manifest["inputs_sha256"]:
            return RuntimeEval(format="json", runtime="numpy", metric="replay_match", value=0.0, n=n,
                               matches_training=False,
                               reason="results.json was altered: its inputs do not hash to its inputs_sha256")
        if risk.inputs_digest(risk.embed_tables(data.train)) != manifest["inputs_sha256"]:
            return RuntimeEval(format="json", runtime="numpy", metric="replay_match", value=0.0, n=n,
                               matches_training=False,
                               reason="the inputs results.json carries are not the tables of this dataset")
        opt = risk.risk_options(manifest["hyperparameters"], manifest["seed"])
        again = risk.run_all(risk.unembed_tables(embedded), opt)
        digest = risk.results_digest(again)
        if digest == delivered["results_sha256"]:
            return RuntimeEval(format="json", runtime="numpy", metric="replay_match", value=1.0, n=n,
                               matches_training=True)
        top = risk._key(max(opt.alphas))
        was, now = delivered["results"]["monte_carlo"][top]["net"], again["monte_carlo"][top]["net"]
        same_env = manifest["versions"] == _versions() and manifest["cpu"] == _cpu()
        return RuntimeEval(format="json", runtime="numpy", metric="replay_match", value=0.0, n=n,
                           matches_training=False,
                           reason=f"digest differs (Monte Carlo net VaR {was!r} -> {now!r}); "
                                  + ("same versions and CPU" if same_env else
                                     f"run on {manifest['versions']} {manifest['cpu']}, replay on "
                                     f"{_versions()} {_cpu()}"))

    def predict(self, artifact: Artifact, inputs: Any) -> Any:
        """What-if: the Monte Carlo VaR with some factors' spend replaced. ``inputs``:
        ``{"spend": {factor: amount}, "alpha": 0.99, "iterations": 20000}`` (all optional)."""
        import numpy as np

        delivered = json.loads(artifact.path.read_text(encoding="utf-8"))
        manifest = delivered["manifest"]
        tables = risk.unembed_tables(delivered["inputs"])
        alpha = float(inputs.get("alpha", max(delivered["results"]["alphas"])))
        hp = {**manifest["hyperparameters"], "alphas": [alpha],
              "iterations": [int(inputs.get("iterations", 20_000))]}
        opt = risk.risk_options(hp, manifest["seed"])
        rng = np.random.Generator(np.random.PCG64(opt.seed))
        t0 = time.perf_counter()
        sim = risk.simulate(tables, opt, opt.iterations[0], rng, spend_override=inputs.get("spend") or None)
        gross, _ = risk.var_es(sim["gross"], alpha)
        net, net_es = risk.var_es(sim["net"], alpha)
        return {"alpha": alpha, "iterations": opt.iterations[0], "gross_var": gross, "net_var": net,
                "net_es": net_es, "defended": gross - net, "seconds": round(time.perf_counter() - t0, 4)}

    #: The notebook shows these by source, in pipeline order (notebook/build.py).
    NOTEBOOK_CELLS: tuple[Any, ...] = (
        risk.risk_options, risk.var_es, risk.historical, risk.delta_normal, risk.gpd_tail, risk.gpd_var_es,
        risk.evt, risk.correlation, risk.defence_cascade, risk.simulate, risk.monte_carlo, risk.run_all,
    )


BACKEND = RiskSimBackend()
# train/export/runtime_eval anchor the notebook's sections; the estimators come before `train` (Model).
RiskSimBackend.NOTEBOOK_CELLS = (*RiskSimBackend.NOTEBOOK_CELLS, BACKEND.train, BACKEND.export, BACKEND.runtime_eval)
