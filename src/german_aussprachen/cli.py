"""Rich CLI for German Pronunciation Assessment."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from german_aussprachen.presets import DEMO_SAMPLES, PRACTICE_PRESETS, get_samples_dir
from german_aussprachen.scorer import get_default_scorer

app = typer.Typer(
    name="aussprache",
    help="Open-source German pronunciation assessment CLI (wav2vec2 + CTC forced alignment + GOP).",
    no_args_is_help=True,
)
console = Console()

STATUS_COLORS = {
    "good": "green",
    "fair": "yellow",
    "error": "red",
    "omitted": "bright_black",
}


@app.command()
def analyze(
    audio_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to audio file (.wav/.mp3/.webm/.ogg)."),
    reference_text: str = typer.Argument(..., help="Expected German sentence."),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON instead of formatted report."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full phoneme table with GOP and durations."),
) -> None:
    """Analyze a German audio recording against a reference sentence."""
    with console.status("Loading wav2vec2 phoneme model & analyzing audio...", spinner="dots"):
        scorer = get_default_scorer()
        result = scorer.score_file(audio_path, reference_text)

    if as_json:
        typer.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    # Header score banner
    score_color = "green" if result.overall_score >= 85 else ("yellow" if result.overall_score >= 65 else "red")
    word_line = Text()
    for i, w in enumerate(result.words):
        if i > 0:
            word_line.append("  ")
        color = STATUS_COLORS.get(w.status, "white")
        word_line.append(f"{w.word}", style=f"bold {color}")
        word_line.append(f" ({w.score:.0f})", style=color)

    summary = (
        f"[bold {score_color}]Overall Score: {result.overall_score:.1f} / 100[/bold {score_color}]\n"
        f"Accuracy (GOP): [bold]{result.accuracy_score:.1f}[/bold]   "
        f"Completeness: [bold]{result.completeness_score:.1f}[/bold]   "
        f"Vowel Length: [bold]{result.duration_score:.1f}[/bold]   "
        f"Free-Decode PER: [bold]{result.phoneme_error_rate * 100:.1f}%[/bold]\n\n"
        f"[dim]Expected IPA:[/dim]  /{result.expected_ipa}/\n"
        f"[dim]Heard IPA   :[/dim]  /{result.heard_ipa}/"
    )
    console.print(Panel(summary, title=f"German Aussprache — {audio_path.name}", border_style=score_color))
    console.print(Panel(word_line, title="Word Breakdown", border_style="cyan"))

    if result.top_issues:
        issue_table = Table(title="Pronunciation Feedback", show_lines=True)
        issue_table.add_column("Word", style="bold")
        issue_table.add_column("Sound")
        issue_table.add_column("Status")
        issue_table.add_column("Diagnosis & Tip")
        for iss in result.top_issues:
            c = STATUS_COLORS.get(iss.status, "white")
            heard_part = f" -> /{iss.heard_ipa}/" if iss.heard_ipa and iss.heard_ipa != iss.expected_ipa else ""
            msg = iss.diagnosis
            if iss.tip and iss.tip not in msg:
                msg += f"\n[dim]Tip: {iss.tip}[/dim]"
            issue_table.add_row(
                iss.word,
                f"/{iss.expected_ipa}/{heard_part}",
                f"[{c}]{iss.status} ({iss.score:.0f})[/{c}]",
                msg,
            )
        console.print(issue_table)
    else:
        console.print("[bold green]No pronunciation issues flagged — great German pronunciation![/bold green]")

    if verbose:
        ptable = Table(title="Phoneme Details (CTC Alignment + GOP)")
        ptable.add_column("Word")
        ptable.add_column("Exp")
        ptable.add_column("Heard")
        ptable.add_column("Score", justify="right")
        ptable.add_column("Logit GOP", justify="right")
        ptable.add_column("Dur (ms)", justify="right")
        ptable.add_column("Free Op")
        ptable.add_column("Status")
        for w in result.words:
            for p in w.phonemes:
                c = STATUS_COLORS.get(p.status, "white")
                extra = f" ({p.length_issue})" if p.length_issue else ""
                ptable.add_row(
                    w.word,
                    f"/{p.expected_ipa}/",
                    f"/{p.heard_ipa}/" if p.heard_ipa else "—",
                    f"[{c}]{p.score:.1f}[/{c}]",
                    f"{p.logit_gop:+.2f}",
                    f"{p.duration_ms:.0f}",
                    p.free_op,
                    f"[{c}]{p.status}{extra}[/{c}]",
                )
        console.print(ptable)


@app.command()
def presets() -> None:
    """List curated German practice sentences and built-in demo audio samples."""
    table = Table(title="German Practice Presets")
    table.add_column("ID", style="cyan")
    table.add_column("Difficulty")
    table.add_column("German Sentence", style="bold")
    table.add_column("Focus Sounds")
    for p in PRACTICE_PRESETS:
        table.add_row(p.id, p.difficulty, p.text, ", ".join(p.focus))
    console.print(table)

    samples_dir = get_samples_dir()
    stable = Table(title="Built-in Demo Audio Samples")
    stable.add_column("Sample ID", style="cyan")
    stable.add_column("Kind")
    stable.add_column("File")
    stable.add_column("Description")
    for s in DEMO_SAMPLES:
        exists = (samples_dir / s.filename).exists()
        kind_style = "green" if s.kind == "native" else "yellow"
        stable.add_row(
            s.id,
            f"[{kind_style}]{s.kind}[/{kind_style}]",
            s.filename if exists else f"[red]{s.filename} (missing)[/red]",
            s.description,
        )
    console.print(stable)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host interface to bind."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on."),
) -> None:
    """Launch the interactive German Pronunciation Assessment web demo."""
    console.print(
        Panel(
            f"Starting [bold cyan]German Aussprache Lab[/bold cyan]\n"
            f"Open [bold underline]http://{host}:{port}[/bold underline] in your browser.\n"
            f"[dim]Model: facebook/wav2vec2-xlsr-53-espeak-cv-ft (local CPU, no API keys needed)[/dim]",
            border_style="cyan",
        )
    )
    uvicorn.run("german_aussprachen.web.app:app", host=host, port=port, reload=False)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
