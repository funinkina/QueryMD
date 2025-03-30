import asyncio
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from check_state import check_files_state
from query_handler import query_with_llm

console = Console()

async def main():
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    )

    changes_detected = False
    try:
        with progress:
            task = progress.add_task("Checking file states & updating embeddings...", total=None)

            changes_detected = await asyncio.to_thread(check_files_state)

            progress.remove_task(task)

    except Exception as e:
        console.print(f"\n[bold red]Error during file checking/processing:[/bold red] {e}")
        return

    try:
        query = console.input("[bold blue]Enter your query (or press Ctrl+C to exit): [/bold blue]")
        if not query.strip():
            return

        response_content = ""
        with console.status("[bold cyan]Searching documents and asking the LLM...", spinner="dots"):
            response_content = query_with_llm(query)

        if response_content:
            response_md = Markdown(response_content)
            console.print(response_md)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"\n[bold red]Error during query processing:[/bold red] {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
