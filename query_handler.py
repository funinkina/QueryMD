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
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        _groq_client = groq.Client(api_key=api_key)
    return _groq_client

def relevant_documents(query_text, n_results=3):
    collection = get_chroma_collection()
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=['documents']
    )

    documents = results.get('documents', [[]])[0]
    document_ids = results.get('ids', [[]])[0]

    if not documents:
        return None, None

    context = "\n\n".join([f"Document '{doc_id}':\n{doc}" for doc_id, doc in zip(document_ids, documents)])
    return context, document_ids

def query_with_llm(query_text, n_results=3):
    """
    Queries relevant documents and then the LLM.

    Returns:
        tuple: (llm_response_content, list_of_document_ids or None)
    """
    groq_client = initialize_groq_client()

    context, document_ids = relevant_documents(query_text, n_results)
    if not context:
        return "I looked through the available documents, but couldn't find specific information related to your query.", None

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
    try:
        response = groq_client.chat.completions.create(
            model=config["llm"]["model_name"],
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        llm_content = response.choices[0].message.content
        return llm_content, document_ids

    except Exception as e:
        return "Sorry, I encountered an error while trying to generate a response.", document_ids


if __name__ == "__main__":
    user_query = input("Enter your query: ")
    llm_response, ids = query_with_llm(user_query)
    print("\nReferenced IDs:", ids)
    print("\nResponse:")
    print(llm_response)
