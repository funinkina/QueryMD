from rich.console import Console
from rich.markdown import Markdown
from check_state import check_files_state
from query_handler import query_with_llm

console = Console()
changes_detected = check_files_state()
if not changes_detected:
    console.print("No changes detected. Document embeddings are up-to-date.", style="green")
else:
    console.print("Document embeddings have been updated, you can now query your content.", style="yellow")

query = console.input("[blue]Enter your query: [/blue]")
response = Markdown(query_with_llm(query))
console.print(response, justify="center")
