from __future__ import annotations

from pydantic import BaseModel, Field


class ShipmentBase(BaseModel):
    shipment_id: str = Field(..., description="Shipment identifier")
    port_of_loading: str
    port_of_discharge: str
    booking_time: str


class ShipmentCreate(ShipmentBase):
    pass


class PredictionRequest(BaseModel):
    shipment_id: str
    booking_time: str
    port_of_loading: str
    port_of_discharge: str
    cargo_type: str | None = None
    container_count: float | None = None
    weight_tons: float | None = None


class PredictionResponse(BaseModel):
    shipment_id: str
    is_more_than_24h_late: bool
    probability_gt_24h_late: float
    predicted_delay_hours: float
