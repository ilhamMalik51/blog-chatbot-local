from pydantic import BaseModel

# Request
class MessageRequest(BaseModel):
    role: str
    use_tool: bool
    content: str

# Response
class MessageResponse(BaseModel):
    role: str = "Assistant"
    content: str