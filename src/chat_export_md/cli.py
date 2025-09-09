# Copyright (C) 2025 Torsten Knodt and contributors
# GNU General Public License
# SPDX-License-Identifier: GPL-3.0-or-later
"""CLI interface for ChatGPT to Markdown converter."""

import typer
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .extractor import extract_to_markdown

app = typer.Typer(
    name="chatgpt-saved-session-to-markdown",
    help="Convert ChatGPT HTML/MHTML/PDF exports to clean Markdown dialogues.",
    no_args_is_help=True,
)

console = Console()


@app.command()
def run(
    files: List[Path] = typer.Argument(..., help="Input files (HTML, MHTML, or PDF)"),
    outdir: Optional[Path] = typer.Option(
        None, "-o", "--outdir", help="Output directory for Markdown files"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
) -> None:
    """Convert ChatGPT session files to Markdown."""
    if outdir is None:
        outdir = Path.cwd() / "output"
    
    outdir.mkdir(exist_ok=True, parents=True)
    
    if verbose:
        console.print(f"Output directory: {outdir}")
        console.print(f"Processing {len(files)} files...")
    
    success_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for file_path in files:
            task = progress.add_task(f"Processing {file_path.name}...", total=None)
            
            try:
                if not file_path.exists():
                    console.print(f"[red]Error: File not found: {file_path}[/red]")
                    continue
                
                output_file = outdir / f"{file_path.stem}.md"
                markdown_content = extract_to_markdown(file_path)
                
                output_file.write_text(markdown_content, encoding="utf-8")
                success_count += 1
                
                if verbose:
                    console.print(f"[green]✓[/green] {file_path.name} → {output_file.name}")
                
            except Exception as e:
                console.print(f"[red]Error processing {file_path.name}: {e}[/red]")
            
            progress.remove_task(task)
    
    console.print(f"\n[green]Successfully converted {success_count}/{len(files)} files[/green]")
    
    if success_count < len(files):
        raise typer.Exit(1)


@app.command()
def headers_fix() -> None:
    """Fix license headers in source files."""
    console.print("[yellow]License header fix not implemented yet[/yellow]")


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"chatgpt-saved-session-to-markdown {__version__}")


if __name__ == "__main__":
    app()
