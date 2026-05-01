"""
agent-recall-ai CLI

Commands:
    list               List all saved checkpoints
    inspect <id>       Show full checkpoint details
    resume <id>        Print the resume prompt for a session
    export <id>        Export as JSON or agenttest fixture
    delete <id>        Delete a checkpoint
    status             Show overall statistics
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.state import SessionStatus
from ..storage.disk import DiskStore

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

app = typer.Typer(
    name="agent-recall-ai",
    help="Structured session checkpointing for AI agents.",
    add_completion=True,
)

console = Console(highlight=False)
logger = logging.getLogger(__name__)
_store_dir: str = ".agent-recall-ai"


def _get_store() -> DiskStore:
    return DiskStore(base_dir=_store_dir)


@app.command(name="list")
def list_sessions(
    status: str | None = typer.Option(None, "--status", "-s", help="Filter: active|completed|failed"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max sessions to show"),
):
    """List all saved checkpoints."""
    store = _get_store()
    status_filter: SessionStatus | None = None
    if status:
        try:
            status_filter = SessionStatus(status)
        except ValueError:
            valid = ", ".join(s.value for s in SessionStatus)
            console.print(f"[red]Invalid status '{status}'. Choose: {valid}[/red]")
            raise typer.Exit(1)
    sessions = store.list_sessions(status=status_filter, limit=limit)

    if not sessions:
        console.print("[dim]No checkpoints found.[/dim]")
        console.print("[dim]Create one with: Checkpoint('my-session') as cp: ...[/dim]")
        return

    t = Table(title=f"Agent Checkpoints ({len(sessions)} sessions)", box=box.SIMPLE_HEAD)
    t.add_column("Session ID", style="cyan", no_wrap=True)
    t.add_column("Status", style="bold")
    t.add_column("Checkpoint", style="dim")
    t.add_column("Cost", style="dim")
    t.add_column("Tokens", style="dim")
    t.add_column("Updated", style="dim")
    t.add_column("Goal", style="dim")

    status_colors = {
        "active": "yellow",
        "completed": "green",
        "failed": "red",
        "suspended": "blue",
    }

    for s in sessions:
        color = status_colors.get(s["status"], "white")
        updated = s["updated_at"][:16].replace("T", " ")
        goal = s["goal_summary"][:40] + ("…" if len(s["goal_summary"]) > 40 else "")
        t.add_row(
            s["session_id"],
            Text(s["status"], style=f"bold {color}"),
            f"#{s['checkpoint_seq']}",
            f"${s['cost_usd']:.4f}",
            f"{s['total_tokens']:,}",
            updated,
            goal,
        )

    console.print(t)


@app.command()
def inspect(
    session_id: str = typer.Argument(..., help="Session ID to inspect"),
    full: bool = typer.Option(False, "--full", "-f", help="Show all decisions and tool calls"),
):
    """Show full details for a checkpoint."""
    store = _get_store()
    state = store.load(session_id)

    if state is None:
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        raise typer.Exit(1)

    # Header
    status_color = {"active": "yellow", "completed": "green", "failed": "red"}.get(
        state.status.value, "white"
    )
    console.print(
        Panel(
            Text.assemble(
                (f"  {state.session_id}", "bold white"),
                ("  ·  ", "dim"),
                (state.status.value.upper(), f"bold {status_color}"),
                ("  ·  ", "dim"),
                (f"Checkpoint #{state.checkpoint_seq}", "dim"),
            ),
            box=box.DOUBLE_EDGE,
            style="blue",
        )
    )

    # Goals
    if state.goals:
        console.print("\n[bold cyan]Goals[/bold cyan]")
        for g in state.goals:
            console.print(f"  • {g}")

    # Constraints
    if state.constraints:
        console.print("\n[bold cyan]Constraints[/bold cyan]")
        for c in state.constraints:
            console.print(f"  [yellow]![/yellow] {c}")

    # Context summary
    if state.context_summary:
        console.print(f"\n[bold cyan]Context[/bold cyan]\n  {state.context_summary}")

    # Decisions
    if state.decisions:
        console.print(f"\n[bold cyan]Decisions ({len(state.decisions)} total)[/bold cyan]")
        show_decisions = state.decisions if full else state.decisions[-5:]
        if not full and len(state.decisions) > 5:
            console.print(f"  [dim]... {len(state.decisions)-5} earlier decisions omitted (use --full)[/dim]")
        for d in show_decisions:
            console.print(f"  [green]✓[/green] {d.summary}")
            if d.reasoning:
                console.print(f"    [dim italic]{d.reasoning}[/dim italic]")

    # Files
    if state.files_modified:
        unique_files = list({f.path: f for f in state.files_modified}.values())
        console.print(f"\n[bold cyan]Files Touched ({len(unique_files)} unique)[/bold cyan]")
        for f in unique_files[:20 if not full else 999]:
            console.print(f"  [dim]{f.action:10}[/dim]  {f.path}")

    # Next steps
    if state.next_steps:
        console.print("\n[bold cyan]Next Steps[/bold cyan]")
        for step in state.next_steps:
            console.print(f"  [bold white]→[/bold white] {step}")

    # Telemetry
    console.print("\n[bold cyan]Telemetry[/bold cyan]")
    console.print(f"  Tokens:   {state.token_usage.total:,}  (prompt: {state.token_usage.prompt:,}  completion: {state.token_usage.completion:,})")
    console.print(f"  Cost:     ${state.cost_usd:.4f}")
    console.print(f"  Context:  {state.context_utilization*100:.1f}% utilized")
    console.print(f"  Created:  {state.created_at.strftime('%Y-%m-%d %H:%M')} UTC")
    console.print(f"  Updated:  {state.updated_at.strftime('%Y-%m-%d %H:%M')} UTC")

    # Alerts
    if state.alerts:
        console.print(f"\n[bold cyan]Alerts ({len(state.alerts)})[/bold cyan]")
        severity_colors = {"info": "dim", "warn": "yellow", "error": "red", "critical": "bold red"}
        for alert in state.alerts[-10:]:
            color = severity_colors.get(alert.severity.value, "white")
            console.print(f"  [{color}]{alert.severity.value.upper():8}[/{color}]  {alert.message[:100]}")

    console.print()


@app.command()
def resume(
    session_id: str = typer.Argument(..., help="Session ID to resume"),
):
    """Print the resume prompt — paste this into a new agent session."""
    store = _get_store()
    state = store.load(session_id)

    if state is None:
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        raise typer.Exit(1)

    prompt = state.resume_prompt()
    console.print(
        Panel(
            prompt,
            title=f"[bold cyan]Resume Prompt: {session_id}[/bold cyan]",
            box=box.ROUNDED,
            style="dim",
        )
    )
    console.print("\n[dim]Copy the above and paste it at the start of your new agent session.[/dim]")


@app.command()
def export(
    session_id: str = typer.Argument(..., help="Session ID to export"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json | agenttest | handoff"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file (default: stdout)"),
):
    """Export a checkpoint as JSON, agenttest fixture, or handoff payload."""
    store = _get_store()
    state = store.load(session_id)

    if state is None:
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        raise typer.Exit(1)

    if format == "json":
        content = state.model_dump_json(indent=2)
    elif format == "handoff":
        content = json.dumps(state.as_handoff(), indent=2, default=str)
    elif format == "agenttest":
        content = _export_as_agenttest(state)
    else:
        console.print(f"[red]Unknown format '{format}'. Use: json | agenttest | handoff[/red]")
        raise typer.Exit(1)

    if output:
        Path(output).write_text(content, encoding="utf-8")
        console.print(f"[green]✓[/green] Exported to {output}")
    else:
        print(content)


def _export_as_agenttest(state) -> str:
    """Generate an agenttest-compatible test file from a checkpoint."""
    goals = "\n".join(f"#   - {g}" for g in state.goals)
    decisions_str = "\n".join(
        f'    assert_behavior(response, "{d.summary[:80]}")'
        for d in state.decisions[:5]
    )
    if not decisions_str:
        decisions_str = '    assert_behavior(response, "agent completes the task successfully")'

    seed_prompt = state.goals[0] if state.goals else "complete the task"
    test_func_name = state.session_id.replace("-", "_")
    created_str = state.created_at.strftime("%Y-%m-%d")
    return f'''\
"""
Behavioral tests generated from checkpoint: {state.session_id}
Checkpoint #{state.checkpoint_seq} — {created_str}

Goals:
{goals}

Run with:
    agenttest run ./test_{test_func_name}.py --evaluator keyword
"""
from agenttest import scenario, assert_behavior

# Replace with your actual agent:
# from my_agent import run_agent
def run_agent(prompt: str) -> str:
    return f"[stub] {{prompt}}"


@scenario(
    "{state.session_id} — replayed from checkpoint",
    seed_prompt="{seed_prompt}",
    variations=10,
    fail_under=0.85,
)
def test_{test_func_name}_behavior(prompt: str) -> str:
    """Behavioral assertions derived from recorded session decisions."""
    response = run_agent(prompt)
{decisions_str}
    return response
'''


@app.command()
def delete(
    session_id: str = typer.Argument(..., help="Session ID to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete a checkpoint."""
    store = _get_store()
    if not store.exists(session_id):
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        raise typer.Exit(1)

    if not yes:
        confirm = typer.confirm(f"Delete checkpoint '{session_id}'?", default=False)
        if not confirm:
            raise typer.Exit()

    if store.delete(session_id):
        console.print(f"[green]✓[/green] Deleted '{session_id}'")
    else:
        console.print(f"[red]Failed to delete '{session_id}'[/red]")
        raise typer.Exit(1)


@app.command()
def status():
    """Show overall checkpoint statistics."""
    store = _get_store()
    sessions = store.list_sessions(limit=1000)

    if not sessions:
        console.print("[dim]No checkpoints found.[/dim]")
        return

    total = len(sessions)
    by_status: dict[str, int] = {}
    total_cost = 0.0
    total_tokens = 0
    for s in sessions:
        by_status[s["status"]] = by_status.get(s["status"], 0) + 1
        total_cost += s["cost_usd"]
        total_tokens += s["total_tokens"]

    t = Table.grid(padding=(0, 3))
    t.add_column(style="dim")
    t.add_column(style="bold white")
    t.add_row("Total sessions:", str(total))
    for status_name, count in sorted(by_status.items()):
        t.add_row(f"  {status_name}:", str(count))
    t.add_row("Total cost:", f"${total_cost:.4f}")
    t.add_row("Total tokens:", f"{total_tokens:,}")

    console.print(Panel(t, title="[bold cyan]agent-recall-ai status[/bold cyan]", box=box.ROUNDED))


@app.command(name="install-hooks")
def install_hooks(
    tool: str = typer.Option(
        "claude-code",
        "--tool", "-t",
        help="Which AI tool to configure: claude-code | cursor | windsurf | generic",
    ),
    session_id: str = typer.Option(
        "auto",
        "--session", "-s",
        help="Checkpoint session name. 'auto' uses the git repo name or cwd name.",
    ),
    global_: bool = typer.Option(
        False, "--global", "-g",
        help="Install into global config (~/.claude/settings.json) instead of project-local.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print what would be changed without writing anything.",
    ),
):
    """
    Install agent-recall-ai hooks into your AI coding tool.

    For Claude Code, this adds a Stop hook to .claude/settings.json that
    auto-saves a checkpoint every time Claude finishes a response.
    Your sessions are protected with zero code changes.

    Examples:
        agent-recall-ai install-hooks
        agent-recall-ai install-hooks --tool cursor
        agent-recall-ai install-hooks --global
        agent-recall-ai install-hooks --session my-project --dry-run
    """
    import re

    # ── Resolve session name ──────────────────────────────────────────────────
    if session_id == "auto":
        # Use git repo name if we're in one, otherwise cwd name
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                session_id = Path(result.stdout.strip()).name
            else:
                session_id = Path.cwd().name
        except Exception:
            session_id = Path.cwd().name
        # Sanitize: replace spaces/special chars with hyphens
        session_id = re.sub(r"[^a-zA-Z0-9_\-]", "-", session_id).strip("-") or "session"

    # ── Build the hook command ────────────────────────────────────────────────
    hook_command = f"agent-recall-ai auto-save --session {session_id}"

    # ── Determine config file path ────────────────────────────────────────────
    config_paths: dict[str, Path] = {
        "claude-code":  (Path.home() / ".claude" / "settings.json") if global_
                        else (Path.cwd() / ".claude" / "settings.json"),
        "cursor":       Path.home() / ".cursor" / "settings.json",
        "windsurf":     Path.home() / ".codeium" / "windsurf" / "settings.json",
        "generic":      Path.cwd() / ".agent-recall-ai" / "hooks.json",
    }

    if tool not in config_paths:
        console.print(f"[red]Unknown tool '{tool}'. Choose: {', '.join(config_paths)}[/red]")
        raise typer.Exit(1)

    config_path = config_paths[tool]

    # ── Build or update the settings JSON ────────────────────────────────────
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            console.print(f"[red]Could not parse {config_path} as JSON.[/red]")
            raise typer.Exit(1)
    else:
        existing = {}

    # Hook structure (Claude Code format)
    new_hook = {"type": "command", "command": hook_command}

    if tool == "claude-code":
        hooks = existing.setdefault("hooks", {})
        stop_hooks = hooks.setdefault("Stop", [])
        # Check if already installed
        already_installed = any(
            h.get("hooks", [{}])[0].get("command", "").startswith("agent-recall-ai")
            for h in stop_hooks if isinstance(h, dict) and h.get("hooks")
        )
        if not already_installed:
            stop_hooks.append({
                "matcher": "",
                "hooks": [new_hook],
            })
        hooks["PostToolUse"] = hooks.get("PostToolUse", []) or []
    else:
        # Generic hooks.json format
        existing.setdefault("on_session_end", []).append(hook_command)

    updated_json = json.dumps(existing, indent=2)

    # ── Dry run: just print ───────────────────────────────────────────────────
    if dry_run:
        console.print(f"\n[bold cyan]Dry run — would write to:[/bold cyan] {config_path}\n")
        console.print(updated_json)
        console.print("\n[dim]Run without --dry-run to apply.[/dim]")
        return

    # ── Write ─────────────────────────────────────────────────────────────────
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(updated_json, encoding="utf-8")

    console.print("\n[bold green]✓ Hooks installed![/bold green]")
    console.print(f"  Tool:    [cyan]{tool}[/cyan]")
    console.print(f"  Session: [cyan]{session_id}[/cyan]")
    console.print(f"  Config:  [dim]{config_path}[/dim]")
    console.print(f"  Command: [dim]{hook_command}[/dim]")
    console.print()
    console.print("From now on, your sessions are auto-saved at the end of every response.")
    console.print(f"Resume any time with: [bold]agent-recall-ai resume {session_id}[/bold]")


@app.command(name="auto-save")
def auto_save(
    session_id: str = typer.Option(..., "--session", "-s", help="Session ID to save"),
    event: str = typer.Option("hook", "--event", "-e", help="Event name (for logging)"),
):
    """
    Auto-save the current checkpoint state (called by hooks).

    This command is normally invoked automatically by the hooks installed
    via 'install-hooks'. It reads the current session and bumps the checkpoint.
    You can also call it manually to force-save.
    """
    store = _get_store()
    state = store.load(session_id)

    if state is None:
        # No session yet — create a placeholder so the hook doesn't error
        logger.debug("auto-save: no session '%s' found, skipping.", session_id)
        return

    # store.save() increments checkpoint_seq internally — do NOT pre-increment here
    store.save(state)
    # Silent success — hooks must not pollute agent output
    logger.debug("auto-save: saved '%s' seq #%d (event=%s)", session_id, state.checkpoint_seq, event)


if __name__ == "__main__":
    app()
