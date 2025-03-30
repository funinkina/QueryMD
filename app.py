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
            task = progress.add_task("Checking file states and updating embeddings if needed...", total=None)
            changes_detected = await asyncio.to_thread(check_files_state)

            progress.remove_task(task)

        if not changes_detected:
            console.print("No changes detected. Document embeddings are up-to-date.", style="green")
        else:
            console.print("Document embeddings have been updated based on file changes.", style="yellow")

    except Exception as e:
        console.print(f"\n[bold red]Error during file checking/processing:[/bold red] {e}")
        return
    try:
        query = console.input("[bold blue]Enter your query (or press Ctrl+C to exit): [/bold blue]")
        if not query.strip():
            console.print("No query entered. Exiting.", style="yellow")
            return

        with console.status("[bold cyan]Searching documents and asking the LLM...", spinner="dots"):
            response_content = query_with_llm(query)

        response_md = Markdown(response_content)
        console.print("\n[bold green]Response:[/bold green]")
        console.print(response_md)

    except KeyboardInterrupt:
        console.print("\nExiting.", style="yellow")
    except Exception as e:
        console.print(f"\n[bold red]Error during query processing:[/bold red] {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\nApplication interrupted. Exiting.", style="yellow")
