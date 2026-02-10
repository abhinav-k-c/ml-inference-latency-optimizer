from pydantic import BaseModel


class LargeModelConfig(BaseModel):
    type: str
    path: str
    artificial_delay_ms: int = 0


class SmallModelConfig(BaseModel):
    type: str
    path: str


class ModelsConfig(BaseModel):
    large: LargeModelConfig
    small: SmallModelConfig


class RoutingConfig(BaseModel):
    strategy: str
    sla_ms: int
    window_size: int


class AppConfig(BaseModel):
    models: ModelsConfig
    routing: RoutingConfig
