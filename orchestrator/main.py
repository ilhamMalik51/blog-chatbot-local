import os, logging, json

from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

from schema import MessageRequest, MessageResponse

from service import init_basic_llm, basic_chat_completion, basic_streaming_completion

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(openapi_prefix="/api")
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# ===== Backend Section =====

# In memory chat history for demonstration purposes
CHAT_HISTORY = []

@app.get("/")
async def root():
    return {"message": "Welcome to the Orchestrator API!"}

@app.post("/orchestrate")
async def orchestrate(request: MessageRequest, 
                      llm : ChatOpenAI = Depends(init_basic_llm)) -> MessageResponse:
    """
    Orchestrate the request.
    """
    if request.use_tool is True:
        llm.bind_tools([])

    if isinstance(request.content, str):
        CHAT_HISTORY.append(HumanMessage(content=request.content))
        messages = CHAT_HISTORY.copy()

    else:
        return HTTPException(
            status_code=400,
            detail="Invalid request format. 'content' must be a string."
        )
    
    logger.info(f"Messages: {messages}")

    # process chat history
    response_content = basic_chat_completion(messages=messages, model=llm)

    # handling message response
    ai_message = MessageResponse(content=response_content)

    # save ai response to chat history
    CHAT_HISTORY.append(AIMessage(content=response_content))

    return JSONResponse(
        content={
            "success": True,
            "data": ai_message.model_dump()
        },
        status_code=200
    )

@app.post("/orchestrate/stream")
async def orchestrate(request: MessageRequest, 
                      llm : ChatOpenAI = Depends(init_basic_llm)) -> MessageResponse:
    """
    Orchestrate the request.
    """
    if request.use_tool is True:
        llm.bind_tools([])

    if isinstance(request.content, str):
        CHAT_HISTORY.append(HumanMessage(content=request.content))
        messages = CHAT_HISTORY.copy()

    else:
        return HTTPException(
            status_code=400,
            detail="Invalid request format. 'content' must be a string."
        )
    
    logger.info(f"Messages: {messages}")

    complete_response = []
    async def stream_generator():
        nonlocal complete_response
        
        try:
            # Use the streaming completion function
            async for chunk in basic_streaming_completion(messages=messages, model=llm):
                if chunk:
                    complete_response.append(chunk)
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                    
            # Once streaming is complete, store the full response in chat history
            if complete_response:
                full_response = "".join(complete_response)
                ai_message = AIMessage(content=full_response)
                CHAT_HISTORY.append(ai_message)
                
            # Send a done message
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )

# ===== Frontend Section =====

@app.get("/chatroom")
async def get_chat_page(request: Request):  
    return templates.TemplateResponse("chatroom.html", {
        "request": request, 
        "api_key": os.environ.get("OPENAI_API_KEY")
    })