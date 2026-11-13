from pydantic import BaseModel, Field, field_validator

class SensorInput(BaseModel):
    pressure: float = Field(..., description="Pressure in bar")
    flow_rate: float = Field(..., description="Flow rate in L/min")
    temperature: float = Field(..., description="Temperature in °C")
    pump_vibration: float = Field(..., description="Pump vibration in mm/s")
    pump_current: float = Field(..., description="Pump current in A")
    ph: float = Field(..., description="pH level")
    tank_level: float = Field(..., description="Tank level in %")

    @field_validator('ph')
    def validate_ph(cls, v):
        if not (0.0 <= v <= 14.0):
            raise ValueError("pH value must be between 0.0 and 14.0")
        return v

    @field_validator('tank_level')
    def validate_tank_level(cls, v):
        if not (0.0 <= v <= 100.0):
            raise ValueError("Tank level percentage must be between 0.0 and 100.0")
        return v

class PredictionResult(BaseModel):
    fault_type: str
    confidence: float
    probabilities: dict
    status: str = "success"