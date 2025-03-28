from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import os
import glob

model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path="./embeddings")

collection_name = "notes_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
)

def create_and_store_embeddings(documents, ids):
    if len(documents) != len(ids):
        raise ValueError("The number of documents must match the number of IDs.")

    embeddings = model.encode(documents, show_progress_bar=True)

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids
    )
    print(f"Successfully added {len(documents)} documents to the collection.")

def read_markdown_files(directory):
    markdown_files = glob.glob(os.path.join(directory, "**", "*.md"), recursive=True)
    documents = []
    ids = []

    for file_path in markdown_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())
            ids.append(os.path.relpath(file_path, directory))  # Use relative path as ID

    return documents, ids

if __name__ == "__main__":
    markdown_directory = "/home/funinkina/Notes/"

    documents, ids = read_markdown_files(markdown_directory)

    if not documents:
        print("No markdown files found in the specified directory.")
    else:
        create_and_store_embeddings(documents, ids)
        pass
