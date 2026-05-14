import json
import sys
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console

from zakuro_poc.backends.docker_backend import docker_available
from zakuro_poc.config import ZakuroPocConfig, load_config
from zakuro_poc.execution.runner import execute_plan
from zakuro_poc.models import ExecutionPlan, ExecutionResult
from zakuro_poc.validation import validate_plan, validate_plan_or_raise

app = typer.Typer(help="Zakuro POC Plugin CLI")
console = Console()


def load_plan(plan_path: Path, config: ZakuroPocConfig | None = None) -> ExecutionPlan:
    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
        if config is not None:
            data.setdefault("backend", config.default_backend)
            data.setdefault("image", config.default_image)
        return ExecutionPlan(**data)
    except FileNotFoundError:
        console.print(f"[red]Error: Plan file not found at {plan_path}[/red]")
        raise typer.Exit(1) from None
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON in plan: {e}[/red]")
        raise typer.Exit(1) from e
    except ValidationError as e:
        console.print(f"[red]Error: Plan does not match schema: {e}[/red]")
        raise typer.Exit(1) from e


def print_plan(plan: ExecutionPlan) -> None:
    console.print(f"[bold]Job Name:[/bold] {plan.job_name}")
    console.print(f"[bold]Backend:[/bold] {plan.backend}")
    console.print(f"[bold]Image:[/bold] {plan.image}")
    console.print(f"[bold]Command:[/bold] {plan.command}")
    console.print(f"[bold]Network Enabled:[/bold] {plan.network_enabled}")
    console.print(f"[bold]Repository URL:[/bold] {plan.repo_url}")
    console.print(f"[bold]CPU Count:[/bold] {plan.resource_limits.cpu_count}")
    console.print(f"[bold]Memory (MB):[/bold] {plan.resource_limits.memory_mb}")
    console.print(f"[bold]Timeout (s):[/bold] {plan.resource_limits.timeout_seconds}")


def print_result(result: ExecutionResult) -> None:
    console.print("\n[bold]Execution Result[/bold]")
    console.print(f"Status: {result.status}")
    console.print(f"Exit Code: {result.exit_code}")
    console.print(f"Duration (ms): {result.duration_ms}")
    console.print(f"Artifact Dir: {result.artifact_dir}")
    console.print("\n[bold]Stdout:[/bold]")
    print(result.stdout)
    if result.stderr:
        console.print("\n[bold]Stderr:[/bold]")
        print(result.stderr)


@app.command()
def version():
    """Print package version."""
    import importlib.metadata

    try:
        v = importlib.metadata.version("zakuro-poc-plugin")
        print(f"zakuro-poc-plugin {v}")
    except importlib.metadata.PackageNotFoundError:
        print("zakuro-poc-plugin version unknown")


@app.command()
def validate(  # noqa: B008
    plan: Path = typer.Option(..., help="Path to the JSON plan"),  # noqa: B008
    config_path: Path | None = typer.Option(  # noqa: B008
        None, "--config", help="Path to custom config"
    ),
) -> None:
    """Validate a plan without executing it."""
    config = load_config(str(config_path) if config_path else None)
    p = load_plan(plan, config)
    try:
        validate_plan_or_raise(p, config)
        console.print("[green]Plan is valid.[/green]")
        raise typer.Exit(0) from None
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e


@app.command("plan-show")
def plan_show(plan: Path = typer.Option(..., help="Path to the JSON plan")) -> None:  # noqa: B008
    """Show human-readable plan details."""
    p = load_plan(plan)
    print_plan(p)


@app.command()
def execute(
    plan: Path = typer.Option(..., help="Path to the JSON plan"),  # noqa: B008
    config_path: Path | None = typer.Option(None, "--config", help="Path to custom config"),  # noqa: B008
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt"),  # noqa: B008
    json_output: bool = typer.Option(False, "--json", help="Output JSON result only"),  # noqa: B008
) -> None:
    """Execute a plan."""
    config = load_config(str(config_path) if config_path else None)
    p = load_plan(plan, config)

    violations = validate_plan(p, config)
    if violations:
        result = execute_plan(p, config)
        if json_output:
            print(result.model_dump_json(indent=2))
        else:
            console.print(f"[red]Validation rejected: {'; '.join(violations)}[/red]")
            print_result(result)
        raise typer.Exit(2) from None

    if not json_output:
        print_plan(p)
        console.print("-" * 40)

    if not yes:
        confirmation = typer.prompt("Type 'yes' to execute")
        if confirmation != "yes":
            console.print("[yellow]Execution aborted.[/yellow]")
            raise typer.Exit(1) from None

    result = execute_plan(p, config)

    if json_output:
        print(result.model_dump_json(indent=2))
    else:
        print_result(result)

    if result.status == "succeeded":
        raise typer.Exit(0)
    elif result.status == "timed_out":
        raise typer.Exit(124)
    elif result.status == "rejected":
        raise typer.Exit(2)
    else:
        raise typer.Exit(1)


@app.command()
def doctor() -> None:
    """Check system readiness."""
    all_ok = True

    # Python version
    v = sys.version_info
    if v.major >= 3 and v.minor >= 11:
        console.print("[green][OK] Python >= 3.11[/green]")
    else:
        console.print("[red][FAIL] Python >= 3.11[/red]")
        all_ok = False

    # Package import
    try:
        import zakuro_poc  # noqa

        console.print("[green][OK] Package imported[/green]")
    except ImportError:
        console.print("[red][FAIL] Package import failed[/red]")
        all_ok = False

    # Config load
    try:
        config = load_config()
        console.print("[green][OK] Config loaded[/green]")
    except Exception as e:
        console.print(f"[red][FAIL] Config load failed: {e}[/red]")
        all_ok = False

    # Artifact root writable
    if "config" in locals():
        artifact_root = Path(config.artifact_root)
        try:
            artifact_root.mkdir(parents=True, exist_ok=True)
            test_file = artifact_root / ".test"
            test_file.touch()
            test_file.unlink()
            console.print("[green][OK] Artifact root writable[/green]")
        except Exception as e:
            console.print(f"[red][FAIL] Artifact root not writable: {e}[/red]")
            all_ok = False

    # Docker checks
    if docker_available():
        console.print("[green][OK] Docker CLI available and daemon reachable[/green]")
    else:
        console.print("[yellow][WARN] Docker daemon not reachable or CLI not available[/yellow]")

    if not all_ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()  # pragma: no cover
