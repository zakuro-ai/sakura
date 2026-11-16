"""Supply-chain Value-at-Risk (task ``risk_simulation``): the numbers, free of the engine.

Four estimates of one loss distribution, side by side, at each confidence level ``alpha``:

* ``historical``  -- empirical quantiles of the recorded scenario losses (``losses`` table);
* ``delta_normal`` -- a Gaussian fitted to the recorded NET losses (mean + z * std): fast, thin-tailed;
* ``evt``         -- peaks over a threshold: a Generalised Pareto fitted to the excesses above the
  ``evt_threshold`` quantile, extrapolated into the tail;
* ``monte_carlo`` -- a Student-t copula over the factor returns (``factors`` table) plus Poisson jumps,
  revalued on the spend exposure (``exposure`` table), then passed through the defence layers.

Every estimate reports ``gross`` (unmitigated), ``defended`` (absorbed by the defence layers) and ``net``
(``max(0, gross - defended)``) as VaR (the ``alpha`` quantile) and ES (the mean beyond it).

Units are whatever the tables use (the synthetic dataset uses billions of a currency on a one-year
horizon); nothing here converts them. The Monte Carlo draws come from ONE ``numpy.random.Generator``
seeded with ``seed``, so a rerun with the same inputs, versions and CPU gives the same bytes:
``results_digest`` is what a replay compares.

Only numpy, scipy and pandas are imported, and only inside the functions that need them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The three tables a risk dataset holds, by file stem. Each may be ``.parquet`` or ``.csv``.
TABLES = ("losses", "factors", "exposure")
#: Columns each table must have. ``factors`` holds one ``r_<factor>`` log-return column per factor.
REQUIRED_COLUMNS = {
    "losses": ("gross", "defended"),
    "exposure": ("factor", "spend", "sensitivity"),
}


@dataclass(frozen=True)
class DefenceLayer:
    """A defence layer absorbs ``share`` of the loss still standing when it is reached, up to ``cap``."""

    name: str
    share: float
    cap: float


@dataclass(frozen=True)
class RiskOptions:
    alphas: tuple[float, ...] = (0.95, 0.975, 0.99)
    #: Monte Carlo sample sizes, run one after the other; the LAST is the headline.
    iterations: tuple[int, ...] = (10_000, 100_000)
    #: Degrees of freedom of the Student-t copula (lower = more joint tail events).
    copula_nu: float = 4.0
    #: Poisson rate of a jump per factor per horizon, and the mean jump size (exponential, as a return).
    jump_rate: float = 0.20
    jump_scale: float = 0.25
    #: How many periods of the factor returns make one horizon (12 monthly returns = one year).
    periods_per_horizon: int = 12
    #: Quantile of the recorded losses above which the GPD is fitted.
    evt_threshold: float = 0.90
    #: Batches used for the Monte Carlo standard error (std of the per-batch quantile / sqrt(batches)).
    se_batches: int = 20
    defence: tuple[DefenceLayer, ...] = field(default_factory=tuple)
    #: Rescale the simulated gross loss so its 0.99 quantile equals this value. None (the default) leaves
    #: the simulation alone: a calibrated number is an input, not a result.
    calibrate_gross_q99: float | None = None
    seed: int = 0


#: The smallest Monte Carlo sample size a run accepts (``limits.max_steps`` never caps below it).
MIN_ITERATIONS = 1_000


def risk_options(hp: dict[str, Any], seed: int) -> RiskOptions:
    """``RiskOptions`` from the resolved hyper-parameters. Unknown keys are refused, so a typo in
    ``overrides`` fails at the start of the run, not silently."""
    known = {"alphas", "iterations", "copula_nu", "jump_rate", "jump_scale", "periods_per_horizon",
             "evt_threshold", "se_batches", "defence", "calibrate_gross_q99"}
    unknown = sorted(set(hp) - known)
    if unknown:
        raise ValueError(f"risk_simulation: unknown hyperparameters {unknown} (known: {sorted(known)})")
    layers = tuple(DefenceLayer(name=str(d["name"]), share=float(d["share"]), cap=float(d["cap"]))
                   for d in hp.get("defence") or ())
    for layer in layers:
        if not 0.0 <= layer.share <= 1.0 or layer.cap < 0:
            raise ValueError(f"defence layer {layer.name!r}: share must be in [0, 1] and cap >= 0")
    alphas = tuple(float(a) for a in hp.get("alphas", RiskOptions.alphas))
    if not alphas or any(not 0.5 < a < 1.0 for a in alphas):
        raise ValueError(f"alphas must be in (0.5, 1), got {list(alphas)}")
    iterations = tuple(int(n) for n in hp.get("iterations", RiskOptions.iterations))
    if not iterations or any(n < MIN_ITERATIONS or n > 5_000_000 for n in iterations):
        raise ValueError(f"iterations must each be between 1000 and 5000000, got {list(iterations)}")
    calib = hp.get("calibrate_gross_q99")
    return RiskOptions(
        alphas=alphas, iterations=iterations,
        copula_nu=float(hp.get("copula_nu", RiskOptions.copula_nu)),
        jump_rate=float(hp.get("jump_rate", RiskOptions.jump_rate)),
        jump_scale=float(hp.get("jump_scale", RiskOptions.jump_scale)),
        periods_per_horizon=int(hp.get("periods_per_horizon", RiskOptions.periods_per_horizon)),
        evt_threshold=float(hp.get("evt_threshold", RiskOptions.evt_threshold)),
        se_batches=int(hp.get("se_batches", RiskOptions.se_batches)),
        defence=layers, calibrate_gross_q99=None if calib is None else float(calib), seed=int(seed))


# ------------------------------------------------------------------ data


def find_tables(root: Path) -> dict[str, Path]:
    """``{table: path}`` for the three tables under ``root`` (searched recursively, first match wins, a
    ``.parquet`` before a ``.csv`` of the same stem). Raises naming every table that is missing."""
    found: dict[str, Path] = {}
    for name in TABLES:
        for ext in (".parquet", ".csv"):
            hits = sorted(p for p in root.rglob(f"{name}{ext}") if p.is_file())
            if hits:
                found[name] = hits[0]
                break
    missing = [t for t in TABLES if t not in found]
    if missing:
        raise ValueError(f"risk dataset is missing tables {missing}: expected files named "
                         f"{', '.join(t + '.parquet' for t in TABLES)} (or .csv)")
    return found


def load_tables(paths: dict[str, Path]) -> dict[str, Any]:
    """The three tables as DataFrames, validated."""
    import pandas as pd

    return check_tables({name: (pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p))
                         for name, p in paths.items()})


def embed_tables(tables: dict[str, Any]) -> dict[str, Any]:
    """The tables as plain JSON (``{name: {columns, data}}``), for an artifact that must carry its own
    inputs. Floats survive a JSON round trip exactly (shortest repr), so a run rebuilt from them is the same
    run."""
    out: dict[str, Any] = json.loads(json.dumps(
        {name: df.to_dict(orient="split", index=False) for name, df in tables.items()}, default=str))
    return out


def inputs_digest(embedded: dict[str, Any]) -> str:
    """sha256 of embedded tables (canonical JSON): what ties the inputs a file carries to the data a run read."""
    text = json.dumps(embedded, sort_keys=True, separators=(",", ":"), allow_nan=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unembed_tables(embedded: dict[str, Any]) -> dict[str, Any]:
    """``embed_tables`` reversed, validated."""
    import pandas as pd

    return check_tables({name: pd.DataFrame(t["data"], columns=t["columns"]) for name, t in embedded.items()})


def check_tables(tables: dict[str, Any]) -> dict[str, Any]:
    """Raise unless the three tables have the columns and rows the estimators need; return them."""
    missing_tables = [t for t in TABLES if t not in tables]
    if missing_tables:
        raise ValueError(f"risk dataset is missing tables {missing_tables}")
    for name, cols in REQUIRED_COLUMNS.items():
        missing = [c for c in cols if c not in tables[name].columns]
        if missing:
            raise ValueError(f"risk table {name!r} is missing columns {missing}")
    factors = factor_columns(tables["factors"])
    if not factors:
        raise ValueError("risk table 'factors' has no r_<factor> return columns")
    unknown = sorted(set(tables["exposure"]["factor"].astype(str)) - {c[2:] for c in factors})
    if unknown:
        raise ValueError(f"exposure names factors with no return column in 'factors': {unknown}")
    if len(tables["losses"]) < 50:
        raise ValueError(f"risk table 'losses' has {len(tables['losses'])} rows; at least 50 are needed "
                         "for a tail estimate")
    return tables


def factor_columns(factors: Any) -> list[str]:
    return [c for c in factors.columns if str(c).startswith("r_")]


# ------------------------------------------------------------------ estimators


def var_es(x: Any, alpha: float) -> tuple[float, float]:
    """``(VaR, ES)`` of a loss sample at ``alpha``: the empirical quantile (linear interpolation) and the
    mean of the losses at or beyond it."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    q = float(np.quantile(x, alpha))
    tail = x[x >= q]
    return q, float(tail.mean()) if tail.size else q


def historical(losses: Any, opt: RiskOptions) -> dict[str, Any]:
    """Empirical quantiles of the recorded gross, defended and net losses."""
    import numpy as np

    gross = losses["gross"].to_numpy(dtype=np.float64)
    defended = np.minimum(losses["defended"].to_numpy(dtype=np.float64), gross)
    net = np.maximum(0.0, gross - defended)
    out: dict[str, Any] = {"n": int(gross.size)}
    for a in opt.alphas:
        g, g_es = var_es(gross, a)
        n, n_es = var_es(net, a)
        out[_key(a)] = {"gross": g, "net": n, "defended": g - n, "gross_es": g_es, "net_es": n_es}
    return out


def delta_normal(losses: Any, opt: RiskOptions) -> dict[str, Any]:
    """A normal distribution fitted to the recorded net losses: mean + z_alpha * std (ES in closed form).
    Has no gross/defended split: it only sees the net."""
    import numpy as np
    from scipy import stats

    gross = losses["gross"].to_numpy(dtype=np.float64)
    net = np.maximum(0.0, gross - np.minimum(losses["defended"].to_numpy(dtype=np.float64), gross))
    mu, sigma = float(net.mean()), float(net.std(ddof=1))
    out: dict[str, Any] = {"n": int(net.size), "mu": mu, "sigma": sigma}
    for a in opt.alphas:
        z = float(stats.norm.ppf(a))
        out[_key(a)] = {"net": mu + z * sigma, "net_es": mu + sigma * float(stats.norm.pdf(z)) / (1 - a)}
    return out


def gpd_tail(x: Any, threshold_q: float) -> dict[str, float]:
    """Fit a Generalised Pareto to the excesses of ``x`` over its ``threshold_q`` quantile (location 0)."""
    import numpy as np
    from scipy import stats

    x = np.asarray(x, dtype=np.float64)
    u = float(np.quantile(x, threshold_q))
    excess = x[x > u] - u
    if excess.size < 10:
        raise ValueError(f"only {excess.size} losses above the {threshold_q} quantile: too few for a GPD fit")
    xi, _, beta = stats.genpareto.fit(excess, floc=0)
    return {"u": u, "xi": float(xi), "beta": float(beta), "n_excess": int(excess.size), "n": int(x.size)}


def gpd_var_es(fit: dict[str, float], alpha: float) -> tuple[float, float]:
    """Peaks-over-threshold VaR and ES (Pickands-Balkema-de Haan). ``xi`` near 0 uses the exponential
    limit; ``xi >= 1`` has no finite ES (reported as inf)."""
    import math

    u, xi, beta = fit["u"], fit["xi"], fit["beta"]
    p_exceed = fit["n_excess"] / fit["n"]
    ratio = (1 - alpha) / p_exceed
    if abs(xi) < 1e-6:
        var = u - beta * math.log(ratio)
        return var, var + beta
    var = u + beta / xi * (ratio ** (-xi) - 1)
    es = (var + beta - xi * u) / (1 - xi) if xi < 1 else math.inf
    return var, es


def evt(losses: Any, opt: RiskOptions) -> dict[str, Any]:
    """GPD tails fitted separately to gross and net losses; defended = gross VaR - net VaR."""
    import numpy as np

    gross = losses["gross"].to_numpy(dtype=np.float64)
    net = np.maximum(0.0, gross - np.minimum(losses["defended"].to_numpy(dtype=np.float64), gross))
    fit_g, fit_n = gpd_tail(gross, opt.evt_threshold), gpd_tail(net, opt.evt_threshold)
    out: dict[str, Any] = {"n": int(gross.size), "gross_fit": fit_g, "net_fit": fit_n}
    for a in opt.alphas:
        g, g_es = gpd_var_es(fit_g, a)
        n, n_es = gpd_var_es(fit_n, a)
        out[_key(a)] = {"gross": g, "net": n, "defended": g - n, "gross_es": g_es, "net_es": n_es}
    return out


def correlation(factors: Any) -> Any:
    """The factors' correlation matrix, projected to the nearest positive-definite one (eigenvalues
    floored at 1e-4, then rescaled to a unit diagonal) so its Cholesky factor exists."""
    import numpy as np

    r = np.corrcoef(factors[factor_columns(factors)].to_numpy(dtype=np.float64), rowvar=False)
    w, v = np.linalg.eigh(r)
    r = v @ np.diag(np.maximum(w, 1e-4)) @ v.T
    d = np.diag(1.0 / np.sqrt(np.diag(r)))
    return d @ r @ d


def defence_cascade(gross: Any, layers: tuple[DefenceLayer, ...]) -> tuple[Any, dict[str, Any]]:
    """Apply the layers in order; ``(total defended, {layer name: amount per draw})``."""
    import numpy as np

    remaining = np.asarray(gross, dtype=np.float64).copy()
    per_layer: dict[str, Any] = {}
    for layer in layers:
        absorbed = np.minimum(remaining * layer.share, layer.cap)
        per_layer[layer.name] = absorbed
        remaining = remaining - absorbed
    return np.asarray(gross, dtype=np.float64) - remaining, per_layer


def simulate(tables: dict[str, Any], opt: RiskOptions, n: int, rng: Any,
             spend_override: dict[str, float] | None = None) -> dict[str, Any]:
    """One Monte Carlo sample of ``n`` horizons: ``{gross, defended, net, layers}`` arrays."""
    import numpy as np
    from scipy import stats

    factors = tables["factors"]
    cols = factor_columns(factors)
    names = [c[2:] for c in cols]
    k = len(cols)
    # One horizon's volatility per factor, from the recorded per-period returns.
    sigma = factors[cols].to_numpy(dtype=np.float64).std(axis=0, ddof=1) * np.sqrt(opt.periods_per_horizon)
    chol = np.linalg.cholesky(correlation(factors))

    exposure = np.zeros(k)
    for row in tables["exposure"].itertuples(index=False):
        exposure[names.index(str(row.factor))] += float(row.spend) * float(row.sensitivity)
    for name, spend in (spend_override or {}).items():
        if name not in names:
            raise ValueError(f"unknown factor {name!r}; factors are {names}")
        sens = tables["exposure"].loc[tables["exposure"]["factor"].astype(str) == name, "sensitivity"]
        exposure[names.index(name)] = float(spend) * (float(sens.mean()) if len(sens) else 1.0)

    z = rng.standard_normal((n, k)) @ chol.T
    w = rng.chisquare(opt.copula_nu, size=(n, 1))
    u = stats.t.cdf(z * np.sqrt(opt.copula_nu / w), df=opt.copula_nu)
    jumps = rng.poisson(opt.jump_rate, size=(n, k)) * rng.exponential(opt.jump_scale, size=(n, k))
    returns = stats.norm.ppf(np.clip(u, 1e-9, 1 - 1e-9)) * sigma + jumps
    gross = np.maximum(0.0, returns) @ exposure
    if opt.calibrate_gross_q99 is not None:
        gross = gross * (opt.calibrate_gross_q99 / float(np.quantile(gross, 0.99)))
    defended, layers = defence_cascade(gross, opt.defence)
    return {"gross": gross, "defended": defended, "net": np.maximum(0.0, gross - defended), "layers": layers}


def monte_carlo(tables: dict[str, Any], opt: RiskOptions) -> dict[str, Any]:
    """The copula simulation at every sample size in ``opt.iterations`` (one generator, drawn in that
    order), with a batch standard error on each net VaR and the defence breakdown at each alpha."""
    import time

    import numpy as np

    rng = np.random.Generator(np.random.PCG64(opt.seed))
    runs: list[dict[str, Any]] = []
    for n in opt.iterations:
        t0 = time.perf_counter()
        sim = simulate(tables, opt, n, rng)
        seconds = time.perf_counter() - t0
        row: dict[str, Any] = {"iterations": n, "seconds": round(seconds, 4)}
        for a in opt.alphas:
            g, g_es = var_es(sim["gross"], a)
            net, n_es = var_es(sim["net"], a)
            batches = np.array_split(sim["net"], opt.se_batches)
            se = float(np.std([np.quantile(b, a) for b in batches], ddof=1) / np.sqrt(opt.se_batches))
            # The layers' contribution in the draws that make up the gross tail: what each layer absorbed,
            # on average, when the gross loss was at or beyond its VaR.
            in_tail = sim["gross"] >= g
            layers = {name: float(v[in_tail].mean()) for name, v in sim["layers"].items()}
            row[_key(a)] = {"gross": g, "net": net, "defended": g - net, "gross_es": g_es, "net_es": n_es,
                            "net_se": se, "layers_in_tail": layers}
        runs.append(row)
    return {"runs": runs, **{k: v for k, v in runs[-1].items() if k.startswith("a")}}


def run_all(tables: dict[str, Any], opt: RiskOptions) -> dict[str, Any]:
    """All four estimates, plus the inputs' shape. JSON-serialisable."""
    losses = tables["losses"]
    return {
        "alphas": list(opt.alphas),
        "historical": historical(losses, opt),
        "delta_normal": delta_normal(losses, opt),
        "evt": evt(losses, opt),
        "monte_carlo": monte_carlo(tables, opt),
        "inputs": {"scenarios": int(len(losses)), "periods": int(len(tables["factors"])),
                   "factors": [c[2:] for c in factor_columns(tables["factors"])],
                   "total_spend": float(tables["exposure"]["spend"].sum())},
    }


def comparison_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per method and alpha: the table a reader compares."""
    rows = []
    for method in ("historical", "delta_normal", "evt", "monte_carlo"):
        for a in results["alphas"]:
            r = results[method][_key(a)]
            rows.append({"method": method, "alpha": a, "gross": r.get("gross"), "defended": r.get("defended"),
                         "net": r["net"], "net_es": r.get("net_es"), "net_se": r.get("net_se")})
    return rows


def results_digest(results: dict[str, Any]) -> str:
    """sha256 of the results without their wall-clock timings: what a bit-exact replay compares."""
    def strip(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k != "seconds"}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    text = json.dumps(strip(results), sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _key(alpha: float) -> str:
    """``a0.99`` -- a stable dict key for a confidence level."""
    return f"a{alpha:g}"


# ------------------------------------------------------------------ synthetic data


def make_synthetic_dataset(out_dir: Path, seed: int = 7, months: int = 360, fmt: str = "parquet") -> dict[str, Path]:
    """Write a synthetic risk dataset (the three tables) to ``out_dir``: five factors (three commodities,
    freight, FX) with fat-tailed, correlated monthly returns and occasional joint crises; an exposure
    table; and one-year scenario losses (gross, defended) built from rolling twelve-month windows of the
    same returns. For demos and tests: the numbers mean nothing about any real business."""
    import numpy as np
    import pandas as pd

    rng = np.random.Generator(np.random.PCG64(seed))
    names = ["wheat", "cocoa", "palm", "freight", "fx"]
    vol = np.array([0.075, 0.09, 0.08, 0.12, 0.03])
    corr = np.array([[1.0, 0.25, 0.45, 0.20, 0.15],
                     [0.25, 1.0, 0.30, 0.15, 0.10],
                     [0.45, 0.30, 1.0, 0.25, 0.20],
                     [0.20, 0.15, 0.25, 1.0, 0.30],
                     [0.15, 0.10, 0.20, 0.30, 1.0]])
    nu = 4.0
    z = rng.standard_normal((months, 5)) @ np.linalg.cholesky(corr).T
    t = z * np.sqrt((nu - 2) / rng.chisquare(nu, size=(months, 1)))
    crisis = rng.random(months) < 0.02  # roughly one joint shock every four years
    shock = crisis[:, None] * rng.exponential(0.12, size=(months, 5))
    returns = t * vol + shock
    dates = pd.date_range("1996-04-30", periods=months, freq="ME")
    factors = pd.DataFrame({"period": dates.strftime("%Y-%m-%d"), **{f"r_{n}": returns[:, i] for i, n in enumerate(names)}})

    spend = np.array([45.0, 28.5, 18.2, 34.2, 86.46])
    sens = np.array([0.22, 0.38, 0.25, 0.16, 0.12])
    exposure = pd.DataFrame({"factor": names, "spend": spend, "sensitivity": sens})

    # One-year scenarios: the cumulative return over each rolling twelve-month window, revalued on today's
    # exposure; the recorded defences absorb a share that saturates for large losses.
    windows = np.lib.stride_tricks.sliding_window_view(returns, 12, axis=0).sum(axis=-1)
    gross = np.maximum(0.0, windows) @ (spend * sens)
    defended = np.minimum(0.72 * gross, 6.0 + 0.35 * gross) * rng.uniform(0.9, 1.0, size=gross.size)
    losses = pd.DataFrame({"scenario_end": dates[11:].strftime("%Y-%m-%d"), "gross": gross, "defended": defended})

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in (("losses", losses), ("factors", factors), ("exposure", exposure)):
        p = out_dir / f"{name}.{fmt}"
        if fmt == "parquet":
            df.to_parquet(p, index=False)
        else:
            df.to_csv(p, index=False)
        paths[name] = p
    return paths
