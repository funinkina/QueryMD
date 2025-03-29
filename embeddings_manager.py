from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import os
import glob
import toml
from pathlib import Path

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


def remove_document_from_collection(doc_id):
    """Remove a document from the collection by its ID."""
    try:
        collection.delete(ids=[doc_id])
        print(f"Successfully removed document with ID: {doc_id}")
    except Exception as e:
        print(f"Error removing document {doc_id}: {e}")

def process_file_for_embeddings(file_path, base_dir):
    """Process a single file and add its embeddings to the collection."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # Create a relative path for the document ID
            path_obj = Path(file_path)
            base_dir_obj = Path(base_dir).resolve()
            doc_id = str(path_obj.relative_to(base_dir_obj))

            # Extract title from the first line
            title = content.splitlines()[0].strip() if content else "Untitled"
            metadata = {"title": title}

            # Generate embedding
            embedding = model.encode([content])[0]

            # Add to collection
            collection.add(
                documents=[content],
                embeddings=[embedding],
                ids=[doc_id],
                metadatas=[metadata]
            )
            print(f"Successfully added document: {doc_id}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
