from typing import Literal

from pydantic import BaseModel


class ScanResponse(BaseModel):
    tag: Literal["healthy", "unhealthy"]
    confidence: float
