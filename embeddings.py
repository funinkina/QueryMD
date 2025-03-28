from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import os
import glob
import groq
import json
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path="./embeddings")

collection_name = "notes_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
)

groq_client = groq.Client(api_key=os.environ.get("GROQ_API_KEY"))

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

def relevant_documents(query_text, n_results=3):
    results = collection.query(query_texts=[query_text], n_results=n_results)
    documents = results.get('documents', [[]])[0]
    document_ids = results.get('ids', [[]])[0]

    if not documents:
        return None, None

    context = "\n\n".join([f"Document '{doc_id}':\n{doc}" for doc_id, doc in zip(document_ids, documents)])
    return context, document_ids

def query_with_llm(query_text, n_results=3, model_name="llama3-8b-8192"):
    context, _ = relevant_documents(query_text, n_results)
    print(f"Context: {context}\n\n")

    if not context:
        return "No relevant documents found for your query."

    prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below.
            Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
            However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
            strike a friendly and conversational tone.
            QUESTION: '{query_text}'
            PASSAGE: '{context}'

            ANSWER:
            """
    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1024
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    def read_markdown_files(directory):
        markdown_files = glob.glob(os.path.join(directory, "**", "*.md"), recursive=True)
        documents = []
        ids = []

        for file_path in markdown_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
                ids.append(os.path.relpath(file_path, directory))  # Use relative path as ID

        return documents, ids

    markdown_directory = "/home/funinkina/Notes/"

    documents, ids = read_markdown_files(markdown_directory)

    if not documents:
        print("No markdown files found in the specified directory.")
    else:
        # Uncomment to create embeddings
        # create_and_store_embeddings(documents, ids)
        pass

    user_query = input("Enter your query: ")
    # relevant_documents_response = relevant_documents(user_query)
    # print(f"Relevant Documents:\n{relevant_documents_response}")
    llm_response = query_with_llm(user_query)
    # print(f"Query: {user_query}")
    print("\nResponse:")
    print(llm_response)
