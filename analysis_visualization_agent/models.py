from typing import Optional

from pydantic import BaseModel, Field


class VizResponse(BaseModel):
    code: Optional[str] = Field(default=None, title="python code for visualization")
    explaination: Optional[str] = Field(default=None, title="explaination of each step of code")
    refusal: Optional[str] = Field(default=None, title="reason for not generating code")
