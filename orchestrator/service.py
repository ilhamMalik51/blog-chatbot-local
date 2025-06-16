import logging

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from config import config

logger = logging.getLogger(__name__)

# Glibal variable to hold chat history
CHAT_HISTORY = []

def init_basic_llm() -> ChatOpenAI:
    """
    Initialize the LLM with the OpenAI model.
    """
    return ChatOpenAI(
        base_url=config.OPENAI_ENDPOINT,
        api_key=config.OPENAI_API_KEY,
        temperature=0.7,
    )

def basic_chat_completion(
    messages: list[dict],
    model: ChatOpenAI = None,
) -> str:
    """
    Perform a chat completion using the provided messages and model.
    """
    if model is None:
        model = init_basic_llm()
    
    try:
        response = model.invoke(messages)
    
    except Exception as e:
        print(f"Error during chat completion: {e}")
        
        return "An error occurred while processing your request."
    
    logger.info(f"Response from model: {response.content}")
    
    return response.content