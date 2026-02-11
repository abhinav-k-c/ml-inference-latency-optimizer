from pydantic import BaseModel, Field
from typing import Dict, Literal


class ModelConfig(BaseModel):
    type: Literal["torch", "sklearn"]
    path: str
    artificial_delay_ms: int | None = 0


class RoutingConfig(BaseModel):
    strategy: str
    sla_ms: int
    window_size: int
    debug_mode: bool = False

class AppConfig(BaseModel):
    models: Dict[str, ModelConfig]
    routing: RoutingConfig
