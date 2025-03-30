from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import toml
from pathlib import Path
import threading

config = toml.load("config.toml")
embeddings_config = config["embeddings"]
files_config = config["files"]

_model = None
_chroma_client = None
_collection = None
_lock = threading.Lock()

def get_embedding_model():
    """Lazily initializes and returns the SentenceTransformer model."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                print("Initializing SentenceTransformer model...")
                _model = SentenceTransformer(embeddings_config["embeddings_function"])
                print("SentenceTransformer model initialized.")
    return _model

def get_chroma_collection():
    """Lazily initializes and returns the ChromaDB client and collection."""
    global _chroma_client, _collection
    if _collection is None:
        with _lock:
            if _collection is None:
                print("Initializing ChromaDB client...")
                _chroma_client = chromadb.PersistentClient(path=embeddings_config["embeddings_path"])
                print("ChromaDB client initialized.")
                collection_name = embeddings_config["collection_name"]
                print(f"Getting or creating ChromaDB collection: {collection_name}...")
                _collection = _chroma_client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=embeddings_config["embeddings_function"]
                    )
                )
                print("ChromaDB collection ready.")
    return _collection

def remove_document_from_collection(doc_id):
    """Remove a document from the collection by its ID."""
    collection = get_chroma_collection()
    try:
        collection.delete(ids=[doc_id])
        print(f"Successfully removed document with ID: {doc_id}")
    except Exception as e:
        print(f"Error removing document {doc_id}: {e}")


def process_file_for_embeddings(file_path, base_dir):
    """Process a single file and add its embeddings to the collection."""
    model = get_embedding_model()
    collection = get_chroma_collection()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            path_obj = Path(file_path)
            base_dir_obj = Path(base_dir).resolve()
            try:
                doc_id = str(path_obj.relative_to(base_dir_obj))
            except ValueError:
                print(f"Warning: Could not make {path_obj} relative to {base_dir_obj}. Using absolute path as ID.")
                doc_id = str(path_obj.resolve())

            title = content.split('\n', 1)[0].strip('# ').strip() if content else "Untitled"
            metadata = {"title": title, "source": doc_id}

            if not content.strip():
                print(f"Skipping empty file: {file_path}")
                return
            try:
                embedding = model.encode([content])[0].tolist()
            except Exception as encode_err:
                print(f"Error encoding content from {file_path}: {encode_err}")
                return
            try:
                collection.add(
                    documents=[content],
                    embeddings=[embedding],
                    ids=[doc_id],
                    metadatas=[metadata]
                )
                print(f"Successfully added/updated document: {doc_id}")
            except Exception as add_err:
                print(f"Error adding document {doc_id} to collection: {add_err}")

    except FileNotFoundError:
        print(f"Error: File not found during processing: {file_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
