from typing import Optional

from pydantic import BaseModel


class PlannerResponse(BaseModel):
    mode: Optional[str] = None
    refusal: Optional[str] = None
