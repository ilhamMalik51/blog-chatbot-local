from langchain_openai import ChatOpenAI
from fastapi import FastAPI, Depends

from schema import MessageRequest, MessageResponse

from langchain.globals import set_llm_cache

from service import init_basic_llm, basic_chat_completion

app = FastAPI(openapi_prefix="/api")

app.post("/orchestrate")
def orchestrate(request: MessageRequest, llm : ChatOpenAI = Depends(init_basic_llm)) -> MessageResponse:
    """
    Orchestrate the request.
    """
    if request.use_tool is True:
        llm.bind_tools([])

    # process chat history
    response_content = basic_chat_completion()

    return {"message": "Orchestration started"}