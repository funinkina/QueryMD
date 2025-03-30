import os
import groq
from dotenv import load_dotenv
import toml
from embeddings_manager import get_chroma_collection, get_embedding_model

load_dotenv()

_groq_client = None
config = toml.load("config.toml")

def initialize_groq_client():
    """Initialize Groq client when needed"""
    global _groq_client
    if _groq_client is None:
        print("Initializing Groq client...")
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        _groq_client = groq.Client(api_key=api_key)
        print("Groq client initialized.")
    return _groq_client

def initialize_clients():
    """Ensure necessary clients are ready (Groq). Chroma is handled by its getter."""
    initialize_groq_client()
    get_chroma_collection()


def relevant_documents(query_text, n_results=3):
    collection = get_chroma_collection()
    print(f"Querying collection for: '{query_text}'")
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=['documents']
    )

    documents = results.get('documents', [[]])[0]
    document_ids = results.get('ids', [[]])[0]

    if not documents:
        print("No relevant documents found in collection.")
        return None, None

    if not document_ids:
        print("Warning: Query returned documents but no IDs.")

    print(f"Found {len(documents)} relevant documents: {document_ids}")
    context = "\n\n".join([f"Document '{doc_id}':\n{doc}" for doc_id, doc in zip(document_ids, documents)])
    return context, document_ids

def query_with_llm(query_text, n_results=3):
    groq_client = initialize_groq_client()

    context, _ = relevant_documents(query_text, n_results)

    if not context:
        return "I looked through the available documents, but couldn't find specific information related to your query."

    prompt = f"""
                You are a helpful and informative bot. Your task is to answer the user's QUESTION based *only* on the provided RELEVANT CONTEXT.
                Be comprehensive and include relevant background information found *within the context*.
                Speak to a non-technical audience: break down complex concepts using simple language, and adopt a friendly and conversational tone.
                Respond in complete sentences. If the context does not contain the answer, state that clearly.

                RELEVANT CONTEXT:
                ---
                {context}
                ---

                QUESTION: '{query_text}'

                ANSWER:"""

    print("Sending query to LLM...")
    try:
        response = groq_client.chat.completions.create(
            model=config["llm"]["model_name"],
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        print("LLM response received.")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return "Sorry, I encountered an error while trying to generate a response."


if __name__ == "__main__":
    user_query = input("Enter your query: ")
    llm_response = query_with_llm(user_query)
    print("\nResponse:")
    print(llm_response)
