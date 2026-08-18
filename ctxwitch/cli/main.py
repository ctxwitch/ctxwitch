"""ctxwitch CLI — the command-line interface for AI context versioning.

Primary command: witch (alias: ctxw)

Commands:
    witch tour               Guided hands-on walkthrough (start here)
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
    """ctxwitch — Version control for AI agent behavior.

    Git tells you what changed in your prompt; ctxwitch tells you what
    the change will do — semantic diffs across 12 behavioral dimensions,
    eval gates, and Context PRs.

    New here? Run: witch tour
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
@click.option("--ref", default="HEAD", help="Reference to diff against (default: last commit, like git diff)")
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


@pr.command(name="merge")
@click.argument("number", type=int)
@click.option(
    "--allow-breaking",
    is_flag=True,
    default=False,
    help="Merge even when CBIA finds breaking behavioral changes",
)
def pr_merge(number: int, allow_breaking: bool):
    """Merge a context PR into its base branch.

    Re-runs the CBIA behavioral check (branch vs base) at merge time;
    breaking changes block the merge unless --allow-breaking is given.
    """
    import yaml as _yaml

    from ctxwitch.core.behavioral import analyze_behavioral_impact
    from ctxwitch.engine.pr import PRStatus, PRStore

    store = get_store()
    pr_store = PRStore(store.root)

    context_pr = pr_store.get(number)
    if not context_pr:
        console.print(f"[red]Error:[/] PR #{number} not found")
        sys.exit(1)
    if context_pr.status in (PRStatus.MERGED, PRStatus.CLOSED):
        console.print(f"[red]Error:[/] PR #{number} is already {context_pr.status.value}")
        sys.exit(1)

    # Behavioral gate at merge time — no checkout needed for the check.
    try:
        old_data = _yaml.safe_load(store._git("show", f"{context_pr.base}:witch.yaml")) or {}
        new_data = _yaml.safe_load(store._git("show", f"{context_pr.branch}:witch.yaml")) or {}
        report = analyze_behavioral_impact(old_data, new_data)
    except Exception:
        report = None

    if report is not None and report.breaking_changes and not allow_breaking:
        dims = ", ".join(i.dimension.display_name for i in report.breaking_changes)
        console.print(
            f"[red]Merge blocked — CBIA found breaking behavioral change(s) "
            f"in: {dims}.[/]"
        )
        console.print(
            "[dim]If this change is reviewed and intentional, rerun with "
            "--allow-breaking.[/]"
        )
        sys.exit(2)

    try:
        store.checkout(context_pr.base)
        sha = store.merge(
            context_pr.branch,
            message=f"witch merge: PR #{number} — {context_pr.title}",
        )
    except Exception as e:
        console.print(f"[red]Merge failed:[/] {e}")
        console.print("[dim]Resolve conflicts on the branch, then retry.[/]")
        sys.exit(1)

    pr_store.update_status(number, PRStatus.MERGED)

    override_note = " (breaking changes overridden)" if (
        report is not None and report.breaking_changes
    ) else ""
    console.print(
        Panel(
            f"[bold green]Merged PR #{number}[/]{override_note}\n"
            f"[dim]Branch:[/] {context_pr.branch} → {context_pr.base}\n"
            f"[dim]Commit:[/] {sha[:8]}",
            title="witch pr merge",
            border_style="green",
        )
    )


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
@click.option(
    "--allow-breaking",
    is_flag=True,
    default=False,
    help="Pass the gate even when CBIA finds breaking behavioral changes",
)
def eval(ref: str, judge: bool, allow_breaking: bool):
    """Run eval gates and behavioral impact analysis on current context.

    The gate has two parts: metric thresholds (scored on the new context)
    and the CBIA behavioral analysis (scored on the change). Both must be
    clean for the gate to pass; breaking behavioral changes block it
    unless --allow-breaking is given.
    """
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

    report = None
    if ref or _has_previous_commit(store):
        actual_ref = ref or "HEAD~1"
        try:
            report = _show_behavioral_impact(store, actual_ref, use_judge=judge)
        except Exception:
            report = None

    if not result.passed:
        console.print("\n[red]Gate verdict: FAILED — metric threshold(s) not met. Merge blocked.[/]")
        sys.exit(1)

    if report is not None and report.breaking_changes:
        if allow_breaking:
            console.print(
                "\n[yellow]Gate verdict: PASSED WITH OVERRIDE — metrics passed; "
                "breaking behavioral change(s) allowed via --allow-breaking.[/]"
            )
            return
        dims = ", ".join(i.dimension.display_name for i in report.breaking_changes)
        console.print(
            f"\n[red]Gate verdict: BLOCKED — metrics passed, but CBIA found "
            f"breaking behavioral change(s) in: {dims}.[/]"
        )
        console.print(
            "[dim]If this change is intentional and reviewed, rerun with "
            "--allow-breaking.[/]"
        )
        sys.exit(2)

    console.print("\n[green]Gate verdict: PASSED[/]")


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


def _render_cbia_report(report):
    """CBIA scorecard used by `scan --diff` (full 12-dimension view, shared style w/ diff)."""
    from ctxwitch.core.dimensions import Dimension

    severity_styles = {
        Severity.NO_CHANGE: ("dim", " "),
        Severity.COSMETIC: ("dim", "."),
        Severity.MINOR: ("yellow", "~"),
        Severity.SIGNIFICANT: ("bold yellow", "!"),
        Severity.BREAKING: ("bold red", "X"),
    }
    cstyle = severity_styles.get(report.compound_severity, ("white", "?"))[0]
    console.print()
    console.print(
        Panel(
            f"[{cstyle}]{report.compound_severity.label}[/{cstyle}]\n[dim]{report.summary}[/]",
            title="Behavioral Impact (CBIA)",
            border_style=cstyle.replace("bold ", ""),
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
                (impact.reason or "")[:80],
            )
        else:
            table.add_row(
                "[dim]—[/]",
                f"[dim]{dim.display_name}[/]",
                "[dim]No Change[/]",
                "[dim]—[/]",
            )

    console.print(table)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--framework", "-f", default=None,
              help="Force an adapter (adk, generic). Default: try all.")
@click.option("--agent", "-a", default=None,
              help="Select one agent by name/symbol when a file has several.")
@click.option("--diff", "ref", default=None, metavar="GIT_REF",
              help="Run CBIA between this git ref and the working tree.")
@click.option("--json", "as_json", is_flag=True, help="Emit snapshot(s) as JSON.")
@click.option("--judge", is_flag=True, help="Enable Tier-6 LLM judge (needs API key).")
def scan(path, framework, agent, ref, as_json, judge):
    """Extract the behavioral surface from agent CODE (ADK, LangGraph, raw SDK).

    Point ctxwitch at a Python agent instead of a witch.yaml:

        witch scan agent.py                 # show extracted snapshot
        witch scan agent.py --diff HEAD~1   # CBIA vs a previous git revision
    """
    import json as _json
    from pathlib import Path as _Path

    from ctxwitch.extract import extract_snapshots
    from ctxwitch.extract.extractor import diff_code

    p = _Path(path)
    new_source = p.read_text(encoding="utf-8")

    # ── --diff mode: this is how you "point at your own code + git diff" ──
    if ref is not None:
        old_source = _git_show(p, ref)
        if old_source is None:
            console.print(f"[red]Could not read {path} at {ref}[/]")
            sys.exit(1)
        report, old_snap, new_snap = diff_code(
            old_source, new_source, source_file=str(p),
            framework=framework, agent=agent, use_judge=judge,
        )
        if old_snap is None and new_snap is None:
            console.print("[yellow]No agent found in either revision.[/]")
            sys.exit(0)
        for label, snap in (("was", old_snap), ("now", new_snap)):
            if snap and snap.dynamic_fields:
                console.print(
                    f"[yellow]![/] unresolved in {label}: "
                    f"{', '.join(snap.dynamic_fields)} "
                    f"[dim](dynamic — not statically analyzable)[/]"
                )
        _render_cbia_report(report)
        # Where to go next.
        if _in_tour_sandbox():
            console.print(
                "\n[dim]↳ That's Path A ✓  See the governance workflow (Path B):[/] "
                "[bold cyan]witch tour[/][dim]  ·  start over:[/] [bold]witch tour --reset[/]"
            )
        else:
            console.print(
                "\n[dim]↳ Score every change automatically:[/] [bold cyan]witch ci[/] "
                "[dim]in your pipeline, or add the ctxwitch GitHub Action.[/]"
            )
        # exit code mirrors severity so scan is CI-usable
        sys.exit(2 if report.compound_severity >= Severity.BREAKING else 0)

    # ── plain scan: show what we extracted ───────────────────────────────
    snaps = extract_snapshots(new_source, source_file=str(p), framework=framework)
    if agent:
        snaps = [s for s in snaps if agent in (s.name, s.agent_symbol)]
    if not snaps:
        console.print("[yellow]No agent found. "
                      "Try --framework adk|generic, or check the file.[/]")
        sys.exit(0)

    if as_json:
        console.print(_json.dumps([s.to_context_dict() for s in snaps], indent=2))
        return

    for snap in snaps:
        comp = snap.to_context_dict()["components"]
        body = [
            f"[bold]model[/]        {comp.get('model') or '[dim]?[/]'}",
            f"[bold]temperature[/]  {comp.get('temperature', '[dim]—[/]')}",
            f"[bold]tools[/]        {', '.join(t['name'] for t in comp.get('tool_definitions', [])) or '[dim]none[/]'}",
            f"[bold]prompt[/]       {(snap.system_prompt[:80] + '…') if len(snap.system_prompt) > 80 else (snap.system_prompt or '[dim]—[/]')}",
        ]
        if snap.dynamic_fields:
            body.append(f"[yellow]unresolved[/]   {', '.join(snap.dynamic_fields)}")
        console.print(Panel(
            "\n".join(body),
            title=f"{snap.name}  [dim]({snap.source_framework} · {p.name}:{snap.source_line})[/]",
            border_style="cyan",
        ))

    # Nudge toward the scoring step. In the tour, offer to make the change for them.
    if _in_tour_sandbox() and p.name == "agent.py":
        console.print(
            "[dim]↳ Next: make a sample change —[/] [bold cyan]witch tour --edit-agent[/] "
            "[dim](or edit agent.py), then[/] [bold cyan]witch scan agent.py --diff HEAD[/][dim].[/]"
        )
    elif _git_show(p, "HEAD") is not None:
        console.print(
            f"[dim]↳ Next: edit {p.name}, then[/] "
            f"[bold]witch scan {p.name} --diff HEAD[/] "
            f"[dim]to score the change.[/]"
        )


def _in_tour_sandbox() -> bool:
    """True when the cwd is a `witch tour` sandbox (so scan can give tour-aware hints)."""
    from pathlib import Path as _Path
    return (_Path.cwd() / ".ctxwitch" / "tour.yaml").exists()


def _git_show(path, ref: str):
    """Return the file's contents at a git ref, or None."""
    import subprocess
    from pathlib import Path as _Path

    p = _Path(path)
    try:
        rel = p.resolve().relative_to(_Path.cwd().resolve())
    except ValueError:
        rel = p
    try:
        out = subprocess.run(
            ["git", "show", f"{ref}:{rel.as_posix()}"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout
    except Exception:
        return None


@cli.command()
@click.option("--base", required=True, help="Base git ref to compare against (e.g. the PR base SHA).")
@click.option("--head", default=None, help="Head git ref (default: working tree / HEAD).")
@click.option("--fail-on", type=click.Choice(["breaking", "significant", "minor", "never"]),
              default="breaking", help="Severity at/above which the check fails (exit 2).")
@click.option("--framework", default=None, help="Force an extractor adapter (adk, generic).")
@click.option("--format", "fmt", type=click.Choice(["markdown", "json"]), default="markdown")
@click.option("--output", "out_path", default=None, help="Write the report to this file instead of stdout.")
def ci(base, head, fail_on, framework, fmt, out_path):
    """Scan a PR's changed files with CBIA and emit a report (for CI / the Action).

    Runs entirely on local git history — no network, no telemetry. Exits 2 when
    the aggregated change is at/above --fail-on, so it can gate a merge.

        witch ci --base origin/main --fail-on breaking
    """
    import json as _json

    from ctxwitch.ci import run_ci
    from ctxwitch.core.dimensions import Severity

    report = run_ci(base=base, head=head, framework=framework, fail_on=fail_on)

    if fmt == "json":
        rendered = _json.dumps(report.to_dict(), indent=2)
    else:
        rendered = report.to_markdown(version=__version__)

    if out_path:
        Path(out_path).write_text(rendered, encoding="utf-8")
        console.print(f"[dim]wrote {out_path}[/] "
                      f"({report.compound_severity.label}, "
                      f"{'blocked' if report.blocked else 'passed'})")
    else:
        # plain print so the markdown is clean for piping into a PR comment
        print(rendered)

    sys.exit(2 if report.blocked else 0)


from ctxwitch.cli.tour import tour  # noqa: E402

cli.add_command(tour)


def main():
    cli()


if __name__ == "__main__":
    main()
