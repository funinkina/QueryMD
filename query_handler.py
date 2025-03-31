import os
try:
    import groq
except ImportError:
    print("groq module not found. Please install it using 'pip install groq'")
try:
    from openai import OpenAI
except ImportError:
    print("openai module not found. Please install it using 'pip install openai'")
from dotenv import load_dotenv
import toml
from embeddings_manager import get_chroma_collection

load_dotenv()

_llm_client = None
config = toml.load("config.toml")

def initialize_client(provider):
    global _llm_client

    if _llm_client is None:
        keymap = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}

        if provider.lower() not in keymap:
            raise ValueError(f"Unsupported provider: {provider}")

        api_key = os.environ.get(keymap[provider.lower()])
        if not api_key:
            raise ValueError(f"{keymap[provider.lower()]} environment variable not set.")
        _llm_client = groq.Client(api_key=api_key) if provider.lower() == 'groq' else OpenAI(api_key=api_key)
    return _llm_client


def relevant_documents(query_text, n_results=2):
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
    provider = config["llm"].get("provider", "groq").lower()

    client = initialize_client(provider)

    context, document_ids = relevant_documents(query_text, n_results)
    if not context:
        return "I looked through the available documents, but couldn't find specific information related to your query.", None

    system_prompt = """
                You are a helpful assistant to provide user with the relevant information from the documents.
                You will be given the context below which is extarcted from the user's notes.
                Your task is to answer the user's question based on the context provided.
                The included context might include some irrelevant information, so extract the relevant information from the context and provide the answer to the user's question.
                If you do not get a specific query in a question format, just summarize the context.
                """

    if config["llm"]["additonal_info"] == "True":
        system_prompt += "Feel free to add any additional information from your knowledge base that might be relevant to the user's question."

    user_context = f"""
                RELEVANT CONTEXT:
                ---
                {context}
                ---

                QUESTION: '{query_text}'
                """
    try:
        model_name = config["llm"].get("model_name", "gpt-3.5-turbo" if provider == "openai" else None)
        if not model_name:
            raise ValueError("Model name must be specified in the configuration for the selected provider.")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context}
            ],
            temperature=config["llm"]["temperature"],
            max_tokens=1024
        )

        llm_content = response.choices[0].message.content
        return llm_content, document_ids

    except Exception as e:
        return f"An error occured: {e}", document_ids


if __name__ == "__main__":
    user_query = input("Enter your query: ")
    llm_response, ids = query_with_llm(user_query)
    print("\nReferenced IDs:", ids)
    print("\nResponse:")
    print(llm_response)
