# QueryMD 📝
## Ask and query your markdown notes using AI 🤖

![Screenshot](Screenshot.png)

It uses ChromaDB as a vector database to store the embeddings of your notes and Groq AI or OpenAI interface to query them. Embeddings are generated using ChromaDB's built-in embedding model - SentenceTransformer and are saved locally only.

## Features ✨
- **Search your notes 🔍**: You can search your notes using keywords or phrases.
- **Totally local 🏠**: The embeddings are stored locally and not sent to any third-party service. The only thing that is sent to the LLM provider is the query and the context of your notes. You can read their privacy policies below to know more about how your data is being used.
- **Git integration 🛠️**: It uses git to track changes in your notes and update the embeddings accordingly.
- **State tracking 📂**: It uses a state file to keep track of the embeddings and notes, if git is not available.
- **AI-powered 🤖**: It uses AI to understand the context of your notes and provide relevant results.
- **Markdown support 📝**: It supports markdown files and can parse them to extract text.
- **TUI 🖥️**: It has a simple TUI to interact with the application.
- **Customizable ⚙️**: You can customize the configuration file to suit your needs.
- **Additional info ℹ️**: It provides additional info from the LLM provider to help you understand the context of your notes.
- **Note references 📌**: It includes note references in the query results to help you find the relevant notes easily.
- **Multi-provider support 🌐**: It supports multiple LLM providers like Groq and OpenAI. (More coming soon)

## Installation ⚙️
### 1. Clone the repository 📥
```bash
git clone https://github.com/funinkina/QueryMD
```
### 2. Create a python virtual environment 🐍
Recommended Python version is 3.11
```bash
cd QueryMD
python -m venv .venv
source .venv/bin/activate
```
### 3. Install the requirements 📦
```bash
pip install -r requirements.txt
```
### 4. Configure your config.toml ⚙️
```toml
[embeddings]
embeddings_function = "all-MiniLM-L6-v2"
collection_name = "notes_collection"
embeddings_path = "/home/funinkina/Notes/.embeddings"

[files]
markdown_directory = "/home/funinkina/Notes/"
state_file = "/home/funinkina/Notes/.state.json"

[state_tracking]
# choose "mtime" if you are not tracking changes with git
method = "git"  # Options: "mtime", "git"

[llm]
provider = "groq" # Options: "groq" or "openai"
model_name = "llama3-8b-8192"
# model_name = "gpt-4o"
temperature = 0.5
additonal_info = "True"
```

### Configuration Options Explained

#### [embeddings]
- **`embeddings_function`**: The embedding model to use (SentenceTransformer model name) (can be left as default)
- **`collection_name`**: Name for your ChromaDB collection `notes_collection` (can be anything)
- **`embeddings_path`**: Directory where embeddings will be stored locally `<absolutepath/to/your/notes/embedding_folder_name>`

#### [files]
- **`markdown_directory`**: Path to your markdown notes directory `<asbolutepath/to/your/notes>`
- **`state_file`**: Path where the state tracking file will be saved `<absolutepath/to/your/notes/.state.json>` 

#### [state_tracking]
- **`method`**: How to track changes in your notes
  - `git`: Uses git history to detect changes (recommended if your notes are in a git repository)
  - `mtime`: Uses file modification times to detect changes (use if git is not available)

#### [llm]
- **`provider`**: Which AI provider to use for querying notes
- **`model_name`**: The specific AI model to use
- **`temperature`**: Controls randomness of AI responses (lower = more deterministic)
- **`additional_info`**: Whether to include extra context from the AI in responses

For additional models, you can check the [Groq](https://console.groq.com/keys) and [OpenAI](https://platform.openai.com/docs/models) documentation.

### 5. Set up your environment variables 🔑
You can use **Groq** or **OpenAI** as your LLM providers.
- Get Groq API keys [here](https://console.groq.com/keys).
- Get OpenAI API keys [here](https://platform.openai.com/account/api-keys).

And put it in `.env`

```bash
GROQ_API_KEY=<your_groq_api_key>
OPENAI_API_KEY=<your_openai_api_key>
```

## Usage 🚀
### Just Run the script 🏃‍♂️
It will automatically create the embeddings for your notes and store them in the specified path defined in the config. It will also create a state file to keep track of the embeddings.
```bash
python app.py
```
It will ask you for a query. You can enter any keyword or phrase related to your notes. It will return the most relevant notes based on the query.

## TODO ✅
- [x] Build a TUI for easy access
- [ ] Native Linux Package
- [ ] Better query results
- [ ] Better embeddings model
- [ ] Improve Documentation
- [x] Include note references in the query results
- [ ] Add support for other LLM providers

- [ ] Add support for local models via ollama

## You can read privacy policies of LLM providers here to know more about how your data is being used: 🔒
- [Groq](https://groq.com/privacy-policy/)
- [OpenAI](https://platform.openai.com/docs/guides/your-data)
  
## License 📜
This project is licensed under the GNU GPL License. See the [LICENSE](LICENSE) file for details.