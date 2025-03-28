from sentence_transformers import SentenceTransformer
import chromadb
# from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
import glob

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./embeddings")

# Create a collection in ChromaDB
collection_name = "notes_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
)

def create_and_store_embeddings(documents, ids):
    """
    Create embeddings for a list of documents and store them in ChromaDB.

    :param documents: List of document texts
    :param ids: List of unique IDs corresponding to the documents
    """
    if len(documents) != len(ids):
        raise ValueError("The number of documents must match the number of IDs.")

    # Generate embeddings
    embeddings = model.encode(documents, show_progress_bar=True)

    # Add documents and embeddings to the collection
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids
    )
    print(f"Successfully added {len(documents)} documents to the collection.")

# Example usage
if __name__ == "__main__":

    # Function to read markdown files from a directory recursively
    def read_markdown_files(directory):
        """
        Reads all markdown files in the given directory and its subdirectories,
        and returns their content and filenames.

        :param directory: Path to the directory containing markdown files
        :return: A tuple of (list of file contents, list of filenames)
        """
        markdown_files = glob.glob(os.path.join(directory, "**", "*.md"), recursive=True)
        documents = []
        ids = []

        for file_path in markdown_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
                ids.append(os.path.relpath(file_path, directory))  # Use relative path as ID

        return documents, ids

    # Specify the directory containing markdown files
    markdown_directory = "/home/funinkina/Notes/"

    # Read markdown files
    documents, ids = read_markdown_files(markdown_directory)

    if not documents:
        print("No markdown files found in the specified directory.")
        # else:
        # Create and store embeddings
        # create_and_store_embeddings(documents, ids)

    results = collection.query(query_texts=["what is vim"], n_results=2)
    print(results)
