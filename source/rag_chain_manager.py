import logging
import re
import requests
from source.vector_db_manager import get_embedding_model,index_search
#from vector_db_manager import get_embedding_model,index_search
import tiktoken


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models = ["deepseek-r1:14b" , "llama3-chatqa"]


MODEL_TOKENIZERS = {
    "deepseek-r1:14b": "cl100k_base",               # Approximation match becasue it's not public information 
    "llama3-chatqa": "cl100k_base"                 # Exact match
}

MAX_TOKEN_LIMIT_FOR_CONTEXT = {
    "deepseek-r1:14b": 125000,                        
    "llama3-chatqa": 3500                          
}

# MODEL_TOKENIZERS = {
#     "deepseek-r1:14b": "cl100k_base",              
#     "llama3-chatqa": "cl100k_base",                
#     "krith/meta-llama-3.1-8b-instruct": "cl100k_base",
#     "bobofrut/llama3-chatqa-2-8b-q8_0": "cl100k_base"
# }

def get_instructions():
     return (
        "Instruction: You are a helpful RAG based chat assistant. "
        "Answer in short and concise manner, ONLY based on the following context based on user request. "
        "Do not mention about the context or sources or documents in the answers at all , simply answer naturally. \n "
        "If the user asks an invalid question, no matter what you see in context, "
        "politely say it's an invalid question and to ask properly.\n\n"
        "If you need to think outloud, give the final answer explicitly.\n\n"
        "If you don’t know the answer, or the given context is empty, politely say it's out of your capacity right now.\n\n"
    )


def get_tokenizer(model_name):
    """
    Load tokenizer model
    """
    encoding_name = MODEL_TOKENIZERS.get(model_name)
    if encoding_name is None:
        raise ValueError(f"No tokenizer assigned for model: {model_name}")
    return tiktoken.get_encoding(encoding_name)



def count_tokens(text, tokenizer):
    """
    Returns the number of tokens in a given text.
    """
    return len(tokenizer.encode(text))



def format_prompt(chunks, question, model_name, max_tokens=3500):
    """
    Formats the context and question into a prompt for the model.
    """

    tokenizer = get_tokenizer(model_name)
    formatted = ""
    total_tokens = 0

    for i, chunk in enumerate(chunks, 1):
        chunk = chunk.strip()
        part = f"[Document {i}]\n{chunk}\n\n"
        part_tokens = count_tokens(part, tokenizer)

        if total_tokens + part_tokens > max_tokens:
            logger.warning(f"Truncating context at chunk {i} to stay within max_tokens limit ({max_tokens}). ")
            break

        formatted += part
        total_tokens += part_tokens

    # Final instruction and user question, NOT truncated
    instruction = get_instructions()
    
    prompt = f"{instruction} \n Context: {formatted} \n Question: {question} \n Answer:"

    return prompt



def build_question_answering_chain(quetion, model):
    """
    Builds a question-answering chain using the provided question.
    """
    embedding_model = get_embedding_model() 
    _,_,retrieved_context = index_search(query=quetion,embedding_model=embedding_model)
    
    if not retrieved_context:
        logger.warning("No relevant context found for the question")
        return "No relevant context found for the question."
    
    token_limit = MAX_TOKEN_LIMIT_FOR_CONTEXT.get(model, 3500)
    prompt = format_prompt(retrieved_context,quetion,model, token_limit) 
    answer,_ = ask_ollama(quetion, prompt,model)
    
    return answer



def clean_response(response_text):
    """
    Removes internal thinking and extracts only the final answer.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
    
    answer_match = re.search(r"\*\*Answer:\*\*\s*(.*)", cleaned, flags=re.DOTALL)
    if answer_match:
        cleaned = answer_match.group(1).strip()
    
    return cleaned.strip()



def ask_ollama(question, prompt, model=models[0]):
    if not isinstance(question, str) or not question.strip():
        logger.warning("Received invalid question input.")
        return "Invalid question."
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    
    try:
        model_response = response.json()["response"]
        logger.info("Ollama response received successfully.")
    except (KeyError, ValueError) as e:
        logging.error("Failed to parse Ollama response: %s", e)
        return "Sorry, something went wrong while getting the answer."
    
    formatted_response = clean_response(model_response)
    return formatted_response,model_response


# answer = build_question_answering_chain("who won at world war 1?",models[0])
# print(f"Answer: {answer}")