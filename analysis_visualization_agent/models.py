from pydantic import BaseModel, Field


class VizResponse(BaseModel):
    code: str = Field(default=None, title="python code for visualization")
    explaination: str = Field(default=None, title="explaination of each step of code")
    refusal: str = Field(default=None, title="reason for not generating code")
