from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import os
import glob
import toml

config = toml.load("config.toml")
embeddings_config = config["embeddings"]
files_config = config["files"]

model = SentenceTransformer(embeddings_config["embeddings_function"])

chroma_client = chromadb.PersistentClient(path=embeddings_config["embeddings_path"])

collection_name = embeddings_config["collection_name"]
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(embeddings_config["embeddings_function"])
)

def create_and_store_embeddings(documents, ids, metadata_list):
    if len(documents) != len(ids) or len(documents) != len(metadata_list):
        raise ValueError("The number of documents, IDs, and metadata must match.")

    embeddings = model.encode(documents, show_progress_bar=True)

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadata_list,  # Add metadata to the collection
    )
    print(f"Successfully added {len(documents)} documents to the collection.")

def read_markdown_files(directory):
    markdown_files = glob.glob(os.path.join(directory, "**", "*.md"), recursive=True)
    documents = []
    ids = []
    metadata_list = []

    for file_path in markdown_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents.append(content)
            ids.append(os.path.relpath(file_path, directory))  # Use relative path as ID
            title = content.splitlines()[0].strip() if content else "Untitled"  # Extract the first line as title
            metadata_list.append({"title": title})  # Add title as metadata

    return documents, ids, metadata_list

if __name__ == "__main__":
    markdown_directory = files_config["markdown_directory"]

    documents, ids, metadata_list = read_markdown_files(markdown_directory)

    if not documents:
        print("No markdown files found in the specified directory.")
    else:
        create_and_store_embeddings(documents, ids, metadata_list)
