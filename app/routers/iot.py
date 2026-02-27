"""
IoT Device Integration Router – ESP32 Sensor Data Collection.

Handles real-time sensor data from ESP32 IoT devices including:
- Temperature readings
- Humidity levels
- Soil moisture content
- Plant stress indices
- Integration with 3-day weather forecasts

Data validation, weather integration, and logging for dashboard display.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

from app.services.weather_service import (
    get_3_day_forecast,
    WeatherServiceError,
    LocationNotFoundError,
    APIKeyMissingError
)
from app.services.irrigation_advice_service import (
    get_irrigation_advice,
    GeminiAPIError,
    IrrigationAdviceError
)

logger = logging.getLogger("leaflens")


class SensorDataPayload(BaseModel):
    """Pydantic model for ESP32 sensor data validation."""
    
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Humidity percentage (0-100)")
    soil_moisture: float = Field(..., description="Soil moisture level (0-100)")
    stress_index: float = Field(..., description="Plant stress index (0-100). Calculated as: (temperature / 40.0) * (100 - soil_moisture)")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "temperature": 28.5,
                "humidity": 65.3,
                "soil_moisture": 58.2,
                "stress_index": 72.5
            }
        }


router = APIRouter(prefix="/api", tags=["iot", "sensor"])

# In-memory latest sensor snapshot for dashboard consumption.
# Structure: {"temperature": float, "humidity": float, "soil_moisture": float, "stress_index": float}
latest_sensor_data: Dict[str, Any] = {}


@router.post("/sensor-data")
async def receive_sensor_data(payload: SensorDataPayload) -> JSONResponse:
    """
    POST /api/sensor-data
    
    Receive and validate real-time sensor data from ESP32 IoT device.
    Stores sensor snapshot in-memory for dashboard consumption.
    Weather forecast is fetched separately via GET /api/insight-data.
    
    Request body (JSON):
    ```json
    {
        "temperature": <float>,
        "humidity": <float>,
        "soil_moisture": <float>,
        "stress_index": <float>
    }
    ```
    
    Returns (200):
    ```json
    {
        "status": "success",
        "message": "Sensor data received successfully",
        "timestamp": "2026-02-23T14:30:45.123456"
    }
    ```
    
    Args:
        payload: SensorDataPayload object with validated sensor readings
    
    Returns:
        JSONResponse with status, message, and timestamp
    
    Raises:
        HTTPException: If validation fails (422) or unexpected error occurs (500)
    """
    try:
        timestamp = datetime.utcnow().isoformat()
        
        # Log incoming sensor data
        logger.info(
            f"IoT Sensor Data Received - "
            f"Temperature: {payload.temperature}°C, "
            f"Humidity: {payload.humidity}%, "
            f"Soil Moisture: {payload.soil_moisture}%, "
            f"Stress Index: {payload.stress_index} | "
            f"Timestamp: {timestamp}"
        )
        
        # Store sensor snapshot for dashboard consumption
        try:
            latest_sensor_data.clear()
            latest_sensor_data.update(
                {
                    "temperature": float(payload.temperature),
                    "humidity": float(payload.humidity),
                    "soil_moisture": float(payload.soil_moisture),
                    "stress_index": float(payload.stress_index),
                }
            )
            logger.debug("Updated latest_sensor_data in-memory snapshot")
        except Exception:
            # Do not interrupt device ingestion on storage error; log and continue
            logger.exception("Failed updating latest_sensor_data snapshot")

        response_data = {
            "status": "success",
            "message": "Sensor data received successfully",
            "timestamp": timestamp,
        }

        return JSONResponse(status_code=200, content=response_data)
        
    except ValueError as e:
        logger.error(f"Validation error in sensor data: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Invalid sensor data: {str(e)}"
        )
    
    except Exception as e:
        logger.exception(f"Unexpected error processing sensor data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error processing sensor data"
        )


@router.get("/insight-data")
async def get_insight_data(location: str = Query(..., description="City or location name")) -> JSONResponse:
    """
    GET /api/insight-data?location=CityName

    Returns the latest sensor snapshot combined with a 3-day forecast for dashboard consumption.

    Response (200):
    {
      "status": "success",
      "location": "CityName",
      "sensor_data": { ... },
      "forecast_3_days": [ ... ],
      "timestamp": "ISO8601"
    }

    Error responses:
      404 - no sensor data available or location not found
      503 - weather service unavailable
      500 - unexpected server error
    """
    try:
        logger.info(f"Insight data requested for location: {location}")

        # Ensure we have a latest sensor snapshot
        if not latest_sensor_data:
            msg = "No sensor data available. Ingest data via POST /api/sensor-data first."
            logger.warning(msg)
            raise HTTPException(status_code=404, detail=msg)

        # Fetch forecast for the requested location
        try:
            forecast = get_3_day_forecast(city=location.strip())
        except LocationNotFoundError as e:
            logger.warning(f"Location not found when fetching forecast: {location}")
            raise HTTPException(status_code=404, detail=str(e))
        except APIKeyMissingError as e:
            logger.error("Weather API key missing for forecast request")
            raise HTTPException(status_code=500, detail=str(e))
        except WeatherServiceError as e:
            logger.error(f"Weather service error while fetching forecast: {e}")
            raise HTTPException(status_code=503, detail=str(e))

        response = {
            "status": "success",
            "location": location,
            "sensor_data": latest_sensor_data.copy(),
            "forecast_3_days": forecast,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(f"Providing insight data for {location}")
        return JSONResponse(status_code=200, content=response)

    except HTTPException:
        # Re-raise HTTPExceptions so FastAPI can handle them (and produce correct status codes)
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in get_insight_data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/irrigation-advice")
async def irrigation_advice(location: str = Query(..., description="City or location name")):
    """
    GET /api/irrigation-advice?location=CityName

    Generates smart irrigation advice using Gemini LLM
    based on latest sensor snapshot and 3-day weather forecast.
    """

    try:
        logger.info(f"Irrigation advice requested for location: {location}")

        # 1️⃣ Ensure sensor snapshot exists
        if not latest_sensor_data:
            msg = "No sensor data available. Ingest data via POST /api/sensor-data first."
            logger.warning(msg)
            raise HTTPException(status_code=404, detail=msg)

        # 2️⃣ Fetch forecast using existing weather service
        try:
            forecast = get_3_day_forecast(city=location.strip())
        except LocationNotFoundError as e:
            logger.warning(f"Location not found for irrigation advice: {location}")
            raise HTTPException(status_code=404, detail=str(e))
        except APIKeyMissingError as e:
            logger.error("Weather API key missing")
            raise HTTPException(status_code=500, detail=str(e))
        except WeatherServiceError as e:
            logger.error(f"Weather service error: {e}")
            raise HTTPException(status_code=503, detail=str(e))

        # 3️⃣ Call Gemini Irrigation Service
        try:
            advice = await get_irrigation_advice(
                sensor_data=latest_sensor_data.copy(),
                forecast_3_days=forecast,
                location=location
            )
        except GeminiAPIError as e:
            logger.error(f"Gemini API error: {e}")
            raise HTTPException(status_code=503, detail=str(e))
        except IrrigationAdviceError as e:
            logger.error(f"Irrigation advice error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        # 4️⃣ Return structured response
        response = {
            "status": "success",
            "location": location,
            "irrigation_advice": advice,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(
            f"Irrigation advice generated for {location} | "
            f"Required: {advice['irrigation_required']} | "
            f"Urgency: {advice['urgency']}"
        )

        return JSONResponse(status_code=200, content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in irrigation_advice route: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
