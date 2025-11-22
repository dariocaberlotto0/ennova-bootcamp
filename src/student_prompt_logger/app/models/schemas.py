from pydantic import BaseModel

class LLMResponse(BaseModel):
    text: str
    model_info: str