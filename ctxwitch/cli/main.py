"""ctxwitch CLI — the command-line interface for AI context versioning.

Primary command: witch (alias: ctxw)

Commands:
    witch init <name>        Initialize a new ctxwitch project
    witch checkout <branch>  Switch to a context branch
    witch commit             Commit context changes with version bump
    witch diff               Show semantic diff of context changes
    witch log                Show context change history
    witch pr                 Create, list, and manage context PRs
    witch eval               Run eval gates on current context
    witch rollback           Rollback to a specific version
    witch status             Show current context state
    witch inspect            Inspect and query context internals
    witch spell              Apply context transformations and recipes
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from ctxwitch.core.dimensions import Severity

from ctxwitch import __version__

console = Console()

BANNER = r"""
       _                _  _        _
   ___| |_ ____ __ __ _(_)| |_ __  | |__
  / __| __\ \/ /\ V  V / ||  _/ _| | '_ \
 | (__| |_ >  <  \_/\_/|_| \__\__| | | | |
  \___|\__/_/\_\                   |_| |_|
        context versioning for AI
"""


def get_store():
    from ctxwitch.engine.store import ContextStore
    return ContextStore(Path.cwd())


@click.group()
@click.version_option(__version__, prog_name="ctxwitch")
def cli():
    """ctxwitch — Version control for AI context.

    Manage prompts, RAG configs, tool definitions, and agent handovers
    with git-backed versioning, semantic diffs, eval gates, and context PRs.
    """
    pass


# ─── init ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("name")
@click.option("--owner", "-o", default="", help="Project owner name")
def init(name: str, owner: str):
    """Initialize a new ctxwitch project."""
    store = get_store()
    try:
        snapshot = store.init(name, owner)
        console.print(BANNER, style="bold cyan")
        console.print(
            Panel(
                f"[bold green]Initialized ctxwitch project:[/] {name}\n"
                f"[dim]Version:[/] {snapshot.version}\n"
                f"[dim]Context SHA:[/] {snapshot.sha}\n"
                f"[dim]Files created:[/] witch.yaml, .ctxwitch/, evals/",
                title="witch init",
                border_style="green",
            )
        )
        console.print("\n[dim]Next steps:[/]")
        console.print("  1. Edit [bold]witch.yaml[/] with your AI config")
        console.print("  2. Run [bold]witch commit -m 'initial context'[/]")
        console.print("  3. Create a branch: [bold]witch checkout -b feature-name[/]")
    except RuntimeError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


# ─── status ──────────────────────────────────────────────────────────────────

@cli.command()
def status():
    """Show current context state."""
    store = get_store()
    try:
        store._require_init()
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)

    from ctxwitch.core.context import load_context

    snapshot = load_context(store.context_path)
    branch = store.current_branch

    table = Table(title="Context Status", box=box.ROUNDED)
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Project", snapshot.name)
    table.add_row("Version", snapshot.version)
    table.add_row("Branch", f"[cyan]{branch}[/]")
    table.add_row("SHA", snapshot.sha)
    table.add_row("Model", snapshot.data.get("components", {}).get("model", "—"))
    table.add_row(
        "Temperature",
        str(snapshot.data.get("components", {}).get("temperature", "—")),
    )

    prompt = snapshot.data.get("components", {}).get("system_prompt", "")
    prompt_preview = prompt[:80].replace("\n", " ") + ("..." if len(prompt) > 80 else "")
    table.add_row("Prompt", prompt_preview)

    tools = snapshot.data.get("components", {}).get("tool_definitions", [])
    table.add_row("Tools", str(len(tools)))

    rag = snapshot.data.get("components", {}).get("rag_config", {})
    table.add_row("RAG", "enabled" if rag.get("enabled") else "disabled")

    console.print(table)


# ─── checkout ────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("branch")
@click.option("-b", "--create", is_flag=True, help="Create a new branch")
def checkout(branch: str, create: bool):
    """Switch to or create a context branch."""
    store = get_store()
    try:
        result = store.checkout(branch, create=create)
        action = "Created and switched to" if create else "Switched to"
        console.print(f"[green]{action}[/] branch [bold cyan]{result}[/]")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


# ─── commit ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("-m", "--message", required=True, help="Commit message")
@click.option("--bump", type=click.Choice(["major", "minor", "patch"]), default="patch")
@click.option("--author", default="", help="Override commit author")
def commit(message: str, bump: str, author: str):
    """Commit context changes with automatic version bump."""
    store = get_store()

    main_branch = store.current_branch
    try:
        config_path = store.config_path
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            if config.get("settings", {}).get("require_pr_for_main") and main_branch == "main":
                console.print(
                    "[yellow]Warning:[/] Direct commits to main are discouraged. "
                    "Consider using [bold]witch checkout -b <branch>[/] first."
                )
    except Exception:
        pass

    try:
        record = store.commit(message, author=author, bump=bump)
        console.print(
            Panel(
                f"[bold green]Committed:[/] {record.message}\n"
                f"[dim]Version:[/]  {record.version}\n"
                f"[dim]SHA:[/]      {record.sha}\n"
                f"[dim]Context:[/]  {record.context_sha}\n"
                f"[dim]Author:[/]   {record.author}\n"
                f"[dim]Branch:[/]   {record.branch}",
                title="witch commit",
                border_style="green",
            )
        )
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


# ─── diff ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--ref", default="HEAD~1", help="Reference to diff against")
@click.option("--behavioral/--no-behavioral", default=True, help="Show behavioral impact analysis")
@click.option("--judge", is_flag=True, default=False, help="Enable Tier 6 LLM-as-judge for deeper analysis")
@click.option("--strict", is_flag=True, default=False, help="Exit non-zero on SIGNIFICANT+ changes (for CI)")
def diff(ref: str, behavioral: bool, judge: bool, strict: bool):
    """Show semantic diff of context changes with behavioral impact analysis."""
    store = get_store()
    try:
        ctx_diff = store.diff(ref)

        if not ctx_diff.has_changes:
            console.print("[dim]No context changes.[/]")
            return

        console.print(
            Panel(
                f"[bold]{ctx_diff.old_version}[/] → [bold]{ctx_diff.new_version}[/]\n"
                f"[dim]{ctx_diff.summary}[/]",
                title="Context Diff",
                border_style="yellow",
            )
        )

        changed_paths = {e.path for e in ctx_diff.entries}
        for entry in ctx_diff.entries:
            if entry.change_type == "added":
                console.print(f"  [green]+ {entry.path}:[/] {_format_value(entry.new_value)}")
            elif entry.change_type == "removed":
                console.print(f"  [red]- {entry.path}:[/] {_format_value(entry.old_value)}")
            elif entry.change_type == "modified":
                console.print(f"  [yellow]~ {entry.path}:[/]")
                console.print(f"    [red]- {_format_value(entry.old_value)}[/]")
                console.print(f"    [green]+ {_format_value(entry.new_value)}[/]")

        _show_unchanged_fields(store, changed_paths)

        if behavioral:
            report = _show_behavioral_impact(store, ref, use_judge=judge)
            if strict and report and report.compound_severity >= Severity.SIGNIFICANT:
                sys.exit(2)

    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


def _show_behavioral_impact(store, ref: str, use_judge: bool = False):
    """Run CBIA and display the behavioral impact scorecard. Returns the report."""
    import yaml as _yaml
    from ctxwitch.core.behavioral import analyze_behavioral_impact
    from ctxwitch.core.context import load_context
    from ctxwitch.core.dimensions import Dimension

    current = load_context(store.context_path)

    try:
        old_content = store._git("show", f"{ref}:witch.yaml")
        old_data = _yaml.safe_load(old_content) or {}
    except Exception:
        old_data = {}

    if not old_data:
        return None

    report = analyze_behavioral_impact(old_data, current.data, use_judge=use_judge)

    severity_styles = {
        Severity.NO_CHANGE: ("dim", " "),
        Severity.COSMETIC: ("dim", "."),
        Severity.MINOR: ("yellow", "~"),
        Severity.SIGNIFICANT: ("bold yellow", "!"),
        Severity.BREAKING: ("bold red", "X"),
    }

    console.print()
    compound_style = severity_styles.get(report.compound_severity, ("white", "?"))
    console.print(
        Panel(
            f"[{compound_style[0]}]{report.compound_severity.label}[/{compound_style[0]}]\n"
            f"[dim]{report.summary}[/]",
            title="Behavioral Impact (CBIA)",
            border_style=compound_style[0].replace("bold ", ""),
        )
    )

    impact_by_dim = {i.dimension: i for i in report.impacts}

    table = Table(title="Dimension Scorecard", box=box.ROUNDED)
    table.add_column("", width=3)
    table.add_column("Dimension", style="bold")
    table.add_column("Severity")
    table.add_column("Reason")

    for dim in Dimension:
        impact = impact_by_dim.get(dim)
        if impact and impact.severity > Severity.NO_CHANGE:
            style, icon = severity_styles.get(impact.severity, ("white", "?"))
            table.add_row(
                f"[{style}]{icon}[/{style}]",
                dim.display_name,
                f"[{style}]{impact.severity.label}[/{style}]",
                impact.reason[:80],
            )
        else:
            table.add_row(
                "[dim]—[/]",
                f"[dim]{dim.display_name}[/]",
                "[dim]No Change[/]",
                "[dim]—[/]",
            )

    console.print(table)

    tier6_details = [d for d in report.details if d.startswith("[Tier 6")]
    if tier6_details:
        console.print()
        for detail in tier6_details:
            if "Skipped" in detail or "Token cost" in detail:
                console.print(f"  [dim]{detail}[/]")
            elif "LLM Judge" in detail or detail.startswith("[Tier 6 —"):
                console.print(f"  [cyan]{detail}[/]")
            else:
                console.print(f"  [yellow]{detail}[/]")

    if not use_judge:
        from ctxwitch.core.judge import needs_judge, SUBJECTIVE_DIMENSIONS
        if needs_judge(report):
            low_conf = [
                i for i in report.impacts
                if i.dimension in SUBJECTIVE_DIMENSIONS
                and i.severity >= Severity.SIGNIFICANT
                and i.confidence < 0.85
            ]
            dim_names = ", ".join(i.dimension.display_name for i in low_conf)
            console.print(
                f"\n[cyan]Tip:[/] Low heuristic confidence on [{dim_names}]. "
                f"Run [bold]witch diff --judge[/] for LLM-powered Tier 6 analysis."
            )

    breaking = report.breaking_changes
    if breaking:
        console.print(
            f"\n[bold red]WARNING:[/bold red] {len(breaking)} breaking behavioral change(s) detected. "
            f"Review carefully before merging."
        )

    console.print()
    if breaking:
        console.print(
            "[bold]Recommendation:[/] Merge blocked — resolve breaking changes, "
            "then run [bold]witch eval[/] before opening PR."
        )
    elif report.significant_changes:
        console.print(
            "[bold]Recommendation:[/] Requires eval gate — run [bold]witch eval[/] "
            "before opening PR."
        )
    elif report.changed_dimensions:
        console.print(
            "[bold]Recommendation:[/] Minor changes detected — consider running "
            "[bold]witch eval[/] to validate."
        )
    else:
        console.print(
            "[dim]Recommendation: No behavioral changes — safe to proceed.[/]"
        )

    return report


def _show_unchanged_fields(store, changed_paths: set):
    """Show key fields that were scanned but unchanged.

    Derives the field list from the schema so new fields are picked up
    automatically without manual updates.
    """
    from ctxwitch.core.context import load_context
    from ctxwitch.core.schema import CONTEXT_SCHEMA

    try:
        snapshot = load_context(store.context_path)
    except Exception:
        return

    comp_schema = CONTEXT_SCHEMA.get("properties", {}).get("components", {}).get("properties", {})
    components = snapshot.data.get("components", {})

    unchanged = []
    for field_name, field_def in comp_schema.items():
        path = f"components.{field_name}"
        if any(cp.startswith(path) for cp in changed_paths):
            continue

        val = components.get(field_name)
        if val is None:
            continue

        display = _summarize_field(field_name, val, field_def)
        if display is not None:
            unchanged.append((path, display))

    if unchanged:
        console.print()
        for path, val in unchanged:
            console.print(f"  [dim]  {path}: {val} (unchanged)[/]")


def _summarize_field(name: str, val, field_def: dict) -> str:
    """Produce a compact display value for an unchanged field."""
    field_type = field_def.get("type", "")

    if field_type == "object" and isinstance(val, dict):
        if "enabled" in val:
            return "enabled" if val["enabled"] else "disabled"
        count = len(val)
        return f"{count} key(s)" if count else "(empty)"
    if field_type == "array" and isinstance(val, list):
        return f"{len(val)} item(s)"
    if isinstance(val, (str, int, float, bool)):
        s = str(val)
        return s if len(s) <= 60 else s[:57] + "..."
    return str(val)[:60]


def _format_value(value) -> str:
    if isinstance(value, str):
        value = " ".join(value.split())
        if len(value) > 120:
            return value[:120] + "..."
        return value
    return str(value)


# ─── log ─────────────────────────────────────────────────────────────────────

@cli.command(name="log")
@click.option("-n", "--count", default=20, help="Number of entries to show")
@click.option("--since", "since_days", type=int, default=None, help="Show entries from last N days")
def log_cmd(count: int, since_days):
    """Show context change history."""
    store = get_store()
    try:
        records = store.log(count=count, since_days=since_days)

        if not records:
            console.print("[dim]No context history yet.[/]")
            return

        table = Table(title="Context History", box=box.ROUNDED)
        table.add_column("Version", style="bold cyan")
        table.add_column("Date", style="dim")
        table.add_column("Author")
        table.add_column("Message")
        table.add_column("Branch", style="dim")

        for r in records:
            date = r.timestamp[:10] if r.timestamp else "—"
            table.add_row(r.version, date, r.author, r.message, r.branch)

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


# ─── pr ──────────────────────────────────────────────────────────────────────

@cli.group()
def pr():
    """Manage context pull requests."""
    pass


@pr.command(name="create")
@click.option("-t", "--title", required=True, help="PR title")
@click.option("--base", default="main", help="Base branch to merge into")
def pr_create(title: str, base: str):
    """Create a new context PR from the current branch."""
    store = get_store()

    branch = store.current_branch
    if branch == base:
        console.print(f"[red]Error:[/] Cannot create PR from {base} to {base}. Switch to a feature branch first.")
        sys.exit(1)

    try:
        ctx_diff = store.diff(base)
    except Exception:
        ctx_diff = None

    from ctxwitch.engine.pr import PRStore
    pr_store = PRStore(store.root)
    author = store._git_user() or "unknown"
    context_pr = pr_store.create(title=title, author=author, branch=branch, base=base, diff=ctx_diff)

    console.print(
        Panel(
            f"[bold green]Context PR #{context_pr.number}[/]\n"
            f"[dim]Title:[/]   {context_pr.title}\n"
            f"[dim]Author:[/]  {context_pr.author}\n"
            f"[dim]Branch:[/]  {context_pr.branch} → {context_pr.base}\n"
            f"[dim]Status:[/]  {context_pr.status.value}",
            title="witch pr create",
            border_style="green",
        )
    )

    if ctx_diff and ctx_diff.has_changes:
        console.print(f"\n[dim]Diff: {ctx_diff.summary}[/]")
        for entry in ctx_diff.entries[:5]:
            if entry.change_type == "modified":
                console.print(f"  [yellow]~ {entry.path}[/]")
            elif entry.change_type == "added":
                console.print(f"  [green]+ {entry.path}[/]")
            elif entry.change_type == "removed":
                console.print(f"  [red]- {entry.path}[/]")

    console.print(f"\n[dim]Run [bold]witch eval[/] to trigger eval gate[/]")


@pr.command(name="list")
@click.option("--status", "status_filter", default=None, help="Filter by status")
def pr_list(status_filter):
    """List context PRs."""
    store = get_store()
    from ctxwitch.engine.pr import PRStore, PRStatus

    pr_store = PRStore(store.root)

    filter_status = None
    if status_filter:
        try:
            filter_status = PRStatus(status_filter)
        except ValueError:
            console.print(f"[red]Invalid status:[/] {status_filter}")
            sys.exit(1)

    prs = pr_store.list_prs(status=filter_status)

    if not prs:
        console.print("[dim]No context PRs found.[/]")
        return

    table = Table(title="Context PRs", box=box.ROUNDED)
    table.add_column("#", style="bold")
    table.add_column("Title")
    table.add_column("Author")
    table.add_column("Branch")
    table.add_column("Status")
    table.add_column("Created")

    status_colors = {
        "open": "yellow",
        "eval_passed": "green",
        "eval_failed": "red",
        "approved": "green",
        "merged": "cyan",
        "closed": "dim",
    }

    for p in prs:
        color = status_colors.get(p.status.value, "white")
        table.add_row(
            str(p.number),
            p.title,
            p.author,
            p.branch,
            f"[{color}]{p.status.value}[/{color}]",
            p.created_at[:10],
        )

    console.print(table)


@pr.command(name="show")
@click.argument("number", type=int)
def pr_show(number: int):
    """Show details of a context PR."""
    store = get_store()
    from ctxwitch.engine.pr import PRStore

    pr_store = PRStore(store.root)
    context_pr = pr_store.get(number)

    if not context_pr:
        console.print(f"[red]PR #{number} not found[/]")
        sys.exit(1)

    content = (
        f"[bold]{context_pr.title}[/]\n"
        f"[dim]Author:[/]   {context_pr.author}\n"
        f"[dim]Branch:[/]   {context_pr.branch} → {context_pr.base}\n"
        f"[dim]Status:[/]   {context_pr.status.value}\n"
        f"[dim]Created:[/]  {context_pr.created_at[:19]}\n"
    )

    if context_pr.approvals:
        content += f"[dim]Approved by:[/] {', '.join(context_pr.approvals)}\n"

    console.print(Panel(content, title=f"Context PR #{context_pr.number}", border_style="cyan"))

    if context_pr.diff and context_pr.diff.has_changes:
        console.print(f"\n[bold]Changes:[/] {context_pr.diff.summary}")
        for entry in context_pr.diff.entries:
            if entry.change_type == "modified":
                console.print(f"  [yellow]~ {entry.path}[/]")
                console.print(f"    [red]- {_format_value(entry.old_value)}[/]")
                console.print(f"    [green]+ {_format_value(entry.new_value)}[/]")
            elif entry.change_type == "added":
                console.print(f"  [green]+ {entry.path}: {_format_value(entry.new_value)}[/]")
            elif entry.change_type == "removed":
                console.print(f"  [red]- {entry.path}: {_format_value(entry.old_value)}[/]")

    if context_pr.comments:
        console.print(f"\n[bold]Comments ({len(context_pr.comments)}):[/]")
        for c in context_pr.comments:
            console.print(f"  [{c.author}] {c.body}")


# ─── eval ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--ref", default=None, help="Reference to compare against for behavioral analysis")
@click.option("--judge", is_flag=True, default=False, help="Enable Tier 6 LLM-as-judge")
def eval(ref: str, judge: bool):
    """Run eval gates and behavioral impact analysis on current context."""
    store = get_store()

    from ctxwitch.core.context import load_context
    from ctxwitch.eval.gate import create_default_gate

    snapshot = load_context(store.context_path)
    eval_config = snapshot.data.get("eval")
    golden_path = None

    if eval_config and eval_config.get("golden_dataset"):
        golden_path = store.root / eval_config["golden_dataset"]

    gate = create_default_gate()

    console.print("[bold]Running eval gate...[/]\n")

    result = gate.run(snapshot.data, eval_config=eval_config, golden_path=golden_path)

    verdict_colors = {
        "passed": "green",
        "failed": "red",
        "warning": "yellow",
        "skipped": "dim",
    }
    color = verdict_colors.get(result.verdict.value, "white")

    table = Table(title="Eval Results", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Score")
    table.add_column("Threshold")
    table.add_column("Result")

    for m in result.metrics:
        result_style = "green" if m.passed else "red"
        icon = "PASS" if m.passed else "FAIL"
        table.add_row(
            m.name,
            f"{m.score:.0f}",
            f"{m.threshold:.0f}",
            f"[{result_style}]{icon}[/{result_style}]",
        )

    console.print(table)
    console.print(
        f"\n[{color}]Verdict: {result.verdict.value.upper()}[/{color}]"
        f" — {result.summary}"
    )

    if result.golden_count:
        console.print(f"[dim]Tested against {result.golden_count} golden examples[/]")

    if ref or _has_previous_commit(store):
        actual_ref = ref or "HEAD~1"
        try:
            _show_behavioral_impact(store, actual_ref, use_judge=judge)
        except Exception:
            pass

    if not result.passed:
        console.print("\n[red]Eval gate failed. Merge blocked.[/]")
        sys.exit(1)


def _has_previous_commit(store) -> bool:
    try:
        store._git("rev-parse", "HEAD~1")
        return True
    except Exception:
        return False


# ─── rollback ────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("version")
def rollback(version: str):
    """Rollback to a specific context version."""
    store = get_store()
    try:
        snapshot = store.rollback(version)
        console.print(f"[green]Rolled back to[/] [bold]{version}[/] (SHA: {snapshot.sha})")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


# ─── inspect ─────────────────────────────────────────────────────────────────

@cli.group()
def inspect():
    """Inspect and query context internals."""
    pass


@inspect.command(name="prompt")
def inspect_prompt():
    """Show the full system prompt."""
    store = get_store()
    from ctxwitch.core.context import load_context

    snapshot = load_context(store.context_path)
    prompt = snapshot.data.get("components", {}).get("system_prompt", "")
    console.print(Panel(prompt, title="System Prompt", border_style="cyan"))


@inspect.command(name="tools")
def inspect_tools():
    """List all tool definitions."""
    store = get_store()
    from ctxwitch.core.context import load_context

    snapshot = load_context(store.context_path)
    tools = snapshot.data.get("components", {}).get("tool_definitions", [])

    if not tools:
        console.print("[dim]No tools defined.[/]")
        return

    table = Table(title="Tool Definitions", box=box.ROUNDED)
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Confirmation")

    for t in tools:
        table.add_row(
            t.get("name", "—"),
            t.get("description", "—"),
            "yes" if t.get("requires_confirmation") else "no",
        )

    console.print(table)


@inspect.command(name="rag")
def inspect_rag():
    """Show RAG configuration."""
    store = get_store()
    from ctxwitch.core.context import load_context

    snapshot = load_context(store.context_path)
    rag = snapshot.data.get("components", {}).get("rag_config", {})

    table = Table(title="RAG Config", box=box.ROUNDED)
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    for k, v in rag.items():
        table.add_row(k, str(v))

    console.print(table)


@inspect.command(name="env")
@click.argument("environment", default="prod")
def inspect_env(environment: str):
    """Show environment-specific overrides."""
    store = get_store()
    from ctxwitch.core.context import load_context

    snapshot = load_context(store.context_path)
    envs = snapshot.data.get("environments", {})
    env_data = envs.get(environment)

    if not env_data:
        console.print(f"[dim]No overrides for environment: {environment}[/]")
        return

    import yaml as _yaml
    content = _yaml.dump(env_data, default_flow_style=False)
    console.print(Panel(content, title=f"Environment: {environment}", border_style="cyan"))


# ─── spell (transforms) ─────────────────────────────────────────────────────

@cli.group()
def spell():
    """Apply context transformations and recipes."""
    pass


@spell.command(name="set")
@click.argument("key")
@click.argument("value")
def spell_set(key: str, value: str):
    """Set a context component value. Example: witch spell set temperature 0.5"""
    store = get_store()
    from ctxwitch.core.context import load_context

    snapshot = load_context(store.context_path)
    data = snapshot.data

    parts = key.split(".")
    target = data
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]

    try:
        import json as _json
        parsed = _json.loads(value)
    except (ValueError, TypeError):
        parsed = value

    old_value = target.get(parts[-1])
    target[parts[-1]] = parsed

    import yaml as _yaml
    with open(store.context_path, "w") as f:
        _yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    console.print(f"[green]Set[/] [bold]{key}[/]: {old_value} → {parsed}")


@spell.command(name="add-tool")
@click.argument("name")
@click.option("--description", "-d", default="", help="Tool description")
@click.option("--confirm", is_flag=True, help="Requires user confirmation")
def spell_add_tool(name: str, description: str, confirm: bool):
    """Add a tool definition to the context."""
    store = get_store()
    from ctxwitch.core.context import load_context

    snapshot = load_context(store.context_path)
    data = snapshot.data

    tools = data.get("components", {}).get("tool_definitions", [])
    tool = {"name": name}
    if description:
        tool["description"] = description
    if confirm:
        tool["requires_confirmation"] = True
    tools.append(tool)
    data.setdefault("components", {})["tool_definitions"] = tools

    import yaml as _yaml
    with open(store.context_path, "w") as f:
        _yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    console.print(f"[green]Added tool:[/] [bold]{name}[/]")


@spell.command(name="validate")
def spell_validate():
    """Validate the current witch.yaml against the schema."""
    store = get_store()
    from ctxwitch.core.context import load_context

    try:
        snapshot = load_context(store.context_path)
        console.print(f"[green]Valid[/] — {snapshot.name} {snapshot.version} (SHA: {snapshot.sha})")
    except Exception as e:
        console.print(f"[red]Invalid:[/] {e}")
        sys.exit(1)


@spell.command(name="export")
@click.option("--format", "fmt", type=click.Choice(["yaml", "json"]), default="yaml")
def spell_export(fmt: str):
    """Export the current context."""
    store = get_store()
    from ctxwitch.core.context import load_context

    snapshot = load_context(store.context_path)

    if fmt == "json":
        import json as _json
        console.print(_json.dumps(snapshot.data, indent=2))
    else:
        import yaml as _yaml
        console.print(_yaml.dump(snapshot.data, default_flow_style=False))


# ─── branches ───────────────────────────────────────────────────────────────

@cli.command()
def branches():
    """List all context branches."""
    store = get_store()
    try:
        branch_list = store.branches()
        for b in branch_list:
            prefix = "* " if b.is_current else "  "
            if b.is_current:
                console.print(f"{prefix}[bold cyan]{b.name}[/bold cyan]")
            else:
                console.print(f"{prefix}{b.name}")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


def main():
    cli()


if __name__ == "__main__":
    main()
