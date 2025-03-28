import chromadb
from chromadb.utils import embedding_functions
import os
import groq
from dotenv import load_dotenv
import toml

load_dotenv()

chroma_client = None
collection = None
groq_client = None
embedding_function = None
config = toml.load("config.toml")

def initialize_clients():
    """Lazily initialize clients when needed"""
    global chroma_client, collection, groq_client, embedding_function

    if chroma_client is None:
        chroma_client = chromadb.PersistentClient(path="./embeddings")

    if embedding_function is None:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(config["embeddings"]["embeddings_function"])

    if collection is None:
        collection_name = "notes_collection"
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

    if groq_client is None:
        groq_client = groq.Client(api_key=os.environ.get("GROQ_API_KEY"))

    return chroma_client, collection, groq_client

def relevant_documents(query_text, n_results=3):
    _, collection, _ = initialize_clients()

    results = collection.query(query_texts=[query_text], n_results=n_results)
    documents = results.get('documents', [[]])[0]
    document_ids = results.get('ids', [[]])[0]

    if not documents:
        return None, None

    context = "\n\n".join([f"Document '{doc_id}':\n{doc}" for doc_id, doc in zip(document_ids, documents)])
    return context, document_ids

def query_with_llm(query_text, n_results=3, model_name=config["llm"]["model_name"]):
    _, _, groq_client = initialize_clients()

    context, document_ids = relevant_documents(query_text, n_results)

    if not context:
        return "No relevant documents found for your query."

    prompt = f"""
            QUESTION: '{query_text}'
            ANSWER: ?
            """
    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful and informative bot that answers questions using text from the reference passage included below. "
                    "Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. "
                    "However, you are talking to a non-technical audience, so be sure to break down complicated concepts and strike a friendly and conversational tone. "
                    f"Relevant Context: '{context}'"
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1024
    )

    llm_output = response.choices[0].message.content
    # output_with_references = "\n\n".join(
    #     [f"From file '{doc_id}':\n{llm_output}" for doc_id in document_ids]
    # )

    return llm_output

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    llm_response = query_with_llm(user_query)
    print("\nResponse:")
    print(llm_response)
