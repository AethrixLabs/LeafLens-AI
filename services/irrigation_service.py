from __future__ import annotations

import json
import logging
import os
from typing import Dict, Any, List
from urllib import request, error
from datetime import datetime

logger = logging.getLogger("leaflens.irrigation")

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-1.5-flash:generateContent"
)

MODEL_NAME = "models/gemini-pro"


class GeminiAPIError(Exception):
    pass


class IrrigationAdviceError(Exception):
    pass


def build_irrigation_prompt(
    sensor_data: Dict[str, float],
    forecast_3_days: List[Dict[str, Any]],
    location: str
) -> str:

    forecast_text = ""
    for i, day in enumerate(forecast_3_days, 1):
        forecast_text += (
            f"Day {i} ({day.get('date')}): "
            f"Temp={day.get('avg_temp')}°C, "
            f"Humidity={day.get('avg_humidity')}%, "
            f"Rain={day.get('total_rain_mm')}mm\n"
        )

    return f"""
You are an agricultural expert advising farmers in {location}.

CURRENT SENSOR DATA:
Temperature: {sensor_data.get('temperature')}°C
Humidity: {sensor_data.get('humidity')}%
Soil Moisture: {sensor_data.get('soil_moisture')}%
Stress Index: {sensor_data.get('stress_index')}

3-DAY WEATHER FORECAST:
{forecast_text}

Analyze whether irrigation is required.

Respond ONLY in JSON format:

{{
  "irrigation_required": true or false,
  "urgency": "Low" | "Medium" | "High",
  "explanation": "Short farmer-friendly explanation"
}}
"""


async def get_irrigation_advice(
    sensor_data: Dict[str, float],
    forecast_3_days: List[Dict[str, Any]],
    location: str
) -> Dict[str, Any]:

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiAPIError("GEMINI_API_KEY not configured")

    prompt = build_irrigation_prompt(sensor_data, forecast_3_days, location)

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 300
        }
    }

    url = f"{GEMINI_API_URL}?key={api_key}"
    data = json.dumps(body).encode("utf-8")

    req = request.Request(
        url=url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"}
    )

    try:
        with request.urlopen(req, timeout=10) as resp:
            response_body = resp.read().decode("utf-8")
    except error.HTTPError as e:
        logger.error(f"Gemini HTTP error: {e.code}")
        raise GeminiAPIError(f"Gemini API error: {e.code}")
    except error.URLError as e:
        logger.error(f"Gemini connection error: {e.reason}")
        raise GeminiAPIError("Gemini connection failed")

    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError:
        raise IrrigationAdviceError("Invalid JSON response from Gemini")

    text = _extract_text(payload)

    if not text:
        raise IrrigationAdviceError("Empty response from Gemini")

    # Clean markdown if present
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        advice = json.loads(text)
    except json.JSONDecodeError:
        raise IrrigationAdviceError("Gemini returned non-JSON output")

    required_fields = ["irrigation_required", "urgency", "explanation"]
    for field in required_fields:
        if field not in advice:
            raise IrrigationAdviceError(f"Missing field: {field}")

    return advice


def _extract_text(payload: Dict[str, Any]) -> str:
    try:
        candidates = payload.get("candidates", [])
        if not candidates:
            return ""
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return ""
        return parts[0].get("text", "")
    except Exception:
        return ""