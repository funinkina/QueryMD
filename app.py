from rich.console import Console
from rich.markdown import Markdown
from query_handler import query_with_llm

console = Console()
query = console.input("[blue]Enter your query: [/blue]")
print(f"Your query is: {query}")
response = Markdown(query_with_llm(query))
console.print(response, justify="center")
# console.print("Hello, [bold magenta]World[/bold magenta]!", ":vampire:", locals())
