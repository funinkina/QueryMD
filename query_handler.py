import os
try:
    import groq
except ImportError:
    print("groq module not found. Please install it using 'pip install groq'")
try:
    from openai import OpenAI
except ImportError:
    print("openai module not found. Please install it using 'pip install openai'")
# Add import for ollama
try:
    import ollama
except ImportError:
    print("ollama module not found. Please install it using 'pip install ollama'")

from dotenv import load_dotenv
import toml
from embeddings_manager import get_chroma_collection

load_dotenv()

_llm_client = None
config = toml.load("config.toml")

def initialize_client(provider):
    global _llm_client

    if _llm_client is None:
        provider_lower = provider.lower()
        keymap = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}

        if provider_lower == 'groq':
            api_key = os.environ.get(keymap[provider_lower])
            if not api_key:
                raise ValueError(f"{keymap[provider_lower]} environment variable not set.")
            _llm_client = groq.Client(api_key=api_key)
        elif provider_lower == 'openai':
            api_key = os.environ.get(keymap[provider_lower])
            if not api_key:
                raise ValueError(f"{keymap[provider_lower]} environment variable not set.")
            _llm_client = OpenAI(api_key=api_key)
        elif provider_lower == 'ollama':
            try:
                _llm_client = ollama.Client()  # Using default host: http://localhost:11434
                _llm_client.list()
            except NameError:
                raise ImportError("Ollama provider selected, but the 'ollama' library is not installed. Please run: pip install ollama")
            except Exception as e:
                raise ConnectionError(f"Failed to initialize or connect to Ollama client: {e}")
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers are 'groq', 'openai', 'ollama'.")

    return _llm_client


def relevant_documents(query_text, n_results=2):
    collection = get_chroma_collection()
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=['documents', 'ids']
    )

    documents = results.get('documents', [[]])[0]
    document_ids = results.get('ids', [[]])[0]

    if not documents:
        return None, None

    context = "\n\n".join([f"Document '{doc_id}':\n{doc}" for doc_id, doc in zip(document_ids, documents)])
    return context, document_ids


def query_with_llm(query_text, n_results=3):
    provider = config["llm"].get("provider", "groq").lower()

    try:
        client = initialize_client(provider)
    except (ValueError, ImportError, ConnectionError) as e:
        return f"Error initializing LLM client: {e}", None

    context, document_ids = relevant_documents(query_text, n_results)
    if not context:
        return "I looked through the available documents, but couldn't find specific information related to your query.", None

    system_prompt = """
                You are a helpful assistant designed to answer user questions based *only* on the provided context.
                The context below is extracted from the user's notes. It might contain irrelevant information.
                Your task is to:
                1. Carefully read the user's QUESTION.
                2. Find the relevant information *within* the RELEVANT CONTEXT provided.
                3. Synthesize an answer based *solely* on that information.
                4. If the context does not contain information to answer the question, state that clearly.
                5. If the user provides a statement or topic instead of a question, summarize the relevant parts of the context related to that topic.
                Do NOT use any prior knowledge or information outside the provided RELEVANT CONTEXT.
                """

    if config["llm"].get("additonal_info", "False").lower() == "true":
        system_prompt += "\nHowever, if the config allows, you may supplement your answer with general knowledge if relevant and clearly distinguish it from the context-based answer."

    user_context_prompt = f"""
                RELEVANT CONTEXT:
                ---
                {context}
                ---

                QUESTION: '{query_text}'

                Based *only* on the RELEVANT CONTEXT provided above, answer the QUESTION.
                """

    try:
        model_name = config["llm"].get("model_name")
        if not model_name:
            raise ValueError(f"LLM model_name must be specified in config.toml for the '{provider}' provider.")

        temperature = config["llm"].get("temperature", 0.7)
        max_tokens = config["llm"].get("max_tokens", 1024)

        llm_content = None

        message_data = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_context_prompt}]

        if provider == 'ollama':
            response = client.chat(
                model=model_name,
                messages=message_data,
                options={
                    'temperature': temperature,
                }
            )
            llm_content = response.get('message', {}).get('content', '')

        elif provider in ['groq', 'openai']:
            response = client.chat.completions.create(
                model=model_name,
                messages=message_data,
                temperature=temperature,
                max_tokens=max_tokens
            )
            llm_content = response.choices[0].message.content
        else:
            return f"Unsupported provider '{provider}' encountered during API call.", document_ids

        return llm_content.strip(), document_ids

    except Exception as e:
        error_message = f"An error occurred while querying the LLM ({provider}, model: {model_name or 'Not Specified'}): {e}"
        print(f"[Error] {error_message}")
        return error_message, document_ids


if __name__ == "__main__":
    user_query = input("Enter your query: ")
    if not config:
        config = toml.load("config.toml")
    llm_response, ids = query_with_llm(user_query)

    print("-" * 20)
    if ids:
        print("Referenced Document IDs:")
        for doc_id in ids:
            print(f"- {doc_id}")
    else:
        print("No specific documents were referenced.")
    print("-" * 20)
    print("LLM Response:")
    print(llm_response)
    print("-" * 20)
