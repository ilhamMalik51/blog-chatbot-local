from pydantic import BaseModel

# Request
class MessageRequest(BaseModel):
    role: str
    use_tool: bool
    use_rag: bool = False
    content: str

# Response
class MessageResponse(BaseModel):
    role: str = "assistant"
    content: str