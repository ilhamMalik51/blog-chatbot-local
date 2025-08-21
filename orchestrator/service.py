import logging, json

from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage
from langchain_core.messages import AIMessage

from config import config
from constant import COLLECTION_NAME

logger = logging.getLogger(__name__)

# Global variable to hold chat history
CHAT_HISTORY = []

def init_basic_llm() -> ChatOpenAI:
    """
    Initialize the LLM with the OpenAI model.
    """
    return ChatOpenAI(
        base_url=config.OPENAI_ENDPOINT,
        api_key=config.OPENAI_API_KEY,
        temperature=0.0,
    )

def init_basic_embedding() -> OpenAIEmbeddings:
    """Initialize the embedding model."""
    return OpenAIEmbeddings(
        base_url=config.OPENAI_EMBEDDING_ENDPOINT,
        api_key=config.OPENAI_API_KEY,
    )

def search_vector_db(query: str, 
                     vector_db: QdrantClient = QdrantClient(url=config.QDRANT_ENDPOINT)) -> list[dict]:
    """Search the vector database for similar documents."""
    results = vector_db.query_points(
        collection_name=COLLECTION_NAME,
        query=query,
        limit=10,
        with_payload=True,
    ).points

    return results

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

async def basic_streaming_completion(
        message: str, 
        chat_history: list[dict], 
        model: ChatOpenAI = None):
    """
    Perform a streaming chat completion using the provided messages and model.
    """
    if model is None:
        model = init_basic_llm()
    
    messages = chat_history + [HumanMessage(content=message)]
    try:
        ai_responses = []
        for chunk in model.stream(messages):
            ai_responses.append(chunk.content)
            yield chunk.content
    except Exception as e:
        print(f"Error during streaming chat completion: {e}")
        yield "Error during streaming: " + str(e)

async def rag_streaming_completion(
        message: str, 
        chat_history: list[HumanMessage | AIMessage],
        model: ChatOpenAI = None
    ):
    """Perform a streaming chat completion using the provided messages and model."""
    if model is None:
        model = init_basic_llm()
    emb_model = init_basic_embedding()
    contexts = search_vector_db(query=emb_model.embed_query(message))

    prompt_input = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Chat History: {chat_history}
        Question: {question} 
        Context: {context} 
        Answer:"""

    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
    context_str = "\n\n".join([f"source:{ctx.payload['source']}\ncontent:{ctx.payload['text']}" for ctx in contexts])

    prompt_input = prompt_input.format(
        chat_history=chat_history_str,
        question=message,
        context=context_str
    )
    messages = [HumanMessage(content=prompt_input)]

    try:
        ai_responses = []
        for chunk in model.stream(messages):
            ai_responses.append(chunk.content)
            yield chunk.content
    
    except Exception as e:
        print(f"Error during streaming chat completion: {e}")
        yield "Error during streaming: " + str(e)

async def stream_generator(message: str, 
                           chat_history : list[dict],  
                           model: ChatOpenAI,
                           streaming_completion):
    """Yields streaming responses and updates chat history."""
    complete_response = []
    try:
        async for chunk in streaming_completion(message=message, chat_history=chat_history, model=model):
            if chunk:
                complete_response.append(chunk)
                yield f"{json.dumps({'content': chunk})}\n\n"
        
        # Once streaming is complete, store the full response in chat history
        if complete_response:
            full_response = "".join(complete_response)
            ai_message = AIMessage(content=full_response)
            chat_history.append(ai_message)
        
        # Send a done message
        yield f"{json.dumps({'done': True})}\n\n"
    
    except Exception as e:
        yield f"{json.dumps({'error': str(e)})}\n\n"