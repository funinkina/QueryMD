# QueryMD
## Ask and query your markdown notes using AI

![Screenshot](Screenshot.png)

It uses ChromaDB as a vector database to store the embeddings of your notes and Groq AI or OpenAI interface to query them. Embeddings are generated using ChromaDB's built-in embedding model - SentenceTransformer and are saved locally only.

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/funinkina/QueryMD
```
### 2. Create a python virtual environment
Recommended Python version is 3.11
```bash
cd QueryMD
python -m venv .venv
source .venv/bin/activate
```
### 3. Install the requirements
```bash
pip install -r requirements.txt
```
### 4. Configure your config.toml
Can be used as it is, just change the `markdown_directory` to the path of your markdown files, as well for embeddings.
```toml
[embeddings]
embeddings_function = "all-MiniLM-L6-v2"
collection_name = "notes_collection"
embeddings_path = "/home/funinkina/Notes/.embeddings"

[files]
markdown_directory = "/home/funinkina/Notes/"
state_file = "/home/funinkina/Notes/.state.json"

[llm]
provider = "groq" # Options: "groq" or "openai"
model_name = "llama3-8b-8192"
# model_name = "gpt-4o"
temperature = 0.5
additonal_info = "True"
```
For additional models, you can check the [Groq](https://console.groq.com/keys) and [OpenAI](https://platform.openai.com/docs/models) documentation.

### 5. Set up your environment variables
You can use **Groq** or **OpenAI** as your LLM providers.
- Get Groq API keys [here](https://console.groq.com/keys).
- Get OpenAI API keys [here](https://platform.openai.com/account/api-keys).
And put it in `.env`

```bash
GROQ_API_KEY=<your_groq_api_key>
OPENAI_API_KEY=<your_openai_api_key>
```

## Usage
### Just Run the script 
It will automatically create the embeddings for your notes and store them in the specified path defined in the config. It will also create a state file to keep track of the embeddings.
```bash
python app.py
```
It will ask you for a query. You can enter any keyword or phrase related to your notes. It will return the most relevant notes based on the query.

## TODO
- [x] Build a TUI for easy access
- [ ] Native Linux Package
- [ ] Better query results
- [ ] Better embeddings model
- [ ] Improve Documentation
- [x] Include note references in the query results
- [ ] Give more AI inferences options

## You can read privacy policies of LLM providers here to know more about how your data is being used:
- [Groq](https://groq.com/privacy-policy/)
- [OpenAI](https://platform.openai.com/docs/guides/your-data)
  
## License
This project is licensed under the GNU GPL License. See the [LICENSE](LICENSE) file for details.