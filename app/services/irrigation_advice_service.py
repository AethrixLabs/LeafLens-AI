"""
Irrigation Advice Service – REST-based Gemini Integration.

Provides smart irrigation recommendations based on real-time sensor data,
3-day weather forecast, and Gemini AI analysis.

Uses REST API calls to Gemini (no SDK), matching the architecture of llm_service.py.

Features:
- Structured prompt building from sensor and forecast data
- REST-based Gemini API integration with error handling
- JSON parsing and validation
- Farmer-friendly irrigation recommendations
- Production-grade error handling and logging
"""

import json
import logging
import os
from typing import Dict, Any, List
from urllib import request, error

logger = logging.getLogger("leaflens")

GEMINI_API_URL: str = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-1.5-flash:generateContent"
)
GEMINI_MODEL_NAME: str = "gemini-1.5-flash"


class GeminiAPIError(Exception):
    """Custom exception for Gemini API failures (network, HTTP errors)."""
    pass


class IrrigationAdviceError(Exception):
    """Custom exception for irrigation advice generation failures (parsing, validation)."""
    pass


def build_irrigation_prompt(
    sensor_data: Dict[str, float],
    forecast_3_days: List[Dict[str, Any]],
    location: str
) -> str:
    """
    Build a structured prompt for Gemini to generate irrigation advice.
    
    Args:
        sensor_data: Dictionary with keys: temperature, humidity, soil_moisture, stress_index
        forecast_3_days: List of 3-day forecast dicts with keys: date, avg_temp, avg_humidity, total_rain_mm
        location: City or location name
    
    Returns:
        Formatted prompt string for Gemini API.
    """
    # Format sensor data section
    sensor_section = f"""
CURRENT SENSOR DATA:
- Temperature: {sensor_data.get('temperature', 'N/A')}°C
- Humidity: {sensor_data.get('humidity', 'N/A')}%
- Soil Moisture: {sensor_data.get('soil_moisture', 'N/A')}%
- Plant Stress Index: {sensor_data.get('stress_index', 'N/A')}
"""
    
    # Format 3-day forecast section
    forecast_section = "3-DAY WEATHER FORECAST:\n"
    for i, day_forecast in enumerate(forecast_3_days, 1):
        date = day_forecast.get('date', 'N/A')
        temp = day_forecast.get('avg_temp', 'N/A')
        humidity = day_forecast.get('avg_humidity', 'N/A')
        rain = day_forecast.get('total_rain_mm', 'N/A')
        forecast_section += (
            f"Day {i} ({date}): "
            f"Temp={temp}°C, Humidity={humidity}%, Rain={rain}mm\n"
        )
    
    # Build complete prompt with clear instructions for JSON output
    prompt = f"""You are an agricultural expert providing irrigation advice to farmers in {location}.

Based on the following sensor data and weather forecast, determine if irrigation is needed.

{sensor_section}
{forecast_section}

TASK: Analyze the above data and respond with ONLY a valid JSON object (no markdown formatting, no additional text) in this exact format:

{{
  "irrigation_required": true or false,
  "urgency": "Low" or "Medium" or "High",
  "explanation": "Brief farmer-friendly explanation (1-2 sentences max) of why irrigation is or isn't needed and any recommendations"
}}

INSTRUCTIONS FOR YOUR ANALYSIS:
- Irrigation is likely needed if soil_moisture is below 40% AND stress_index is above 60
- Consider upcoming rainfall in the forecast
- Factor in current humidity levels
- Provide practical advice considering weather trends
- Set urgency to "High" if plants show high stress and no rain is forecast
- Set urgency to "Medium" if plants show moderate stress or uncertain weather
- Set urgency to "Low" if soil moisture is adequate or significant rain is forecast

Respond with ONLY the JSON object. No other text."""

    return prompt


async def get_irrigation_advice(
    sensor_data: Dict[str, float],
    forecast_3_days: List[Dict[str, Any]],
    location: str
) -> Dict[str, Any]:
    """
    Generate irrigation advice using Gemini REST API.
    
    Args:
        sensor_data: Current sensor readings (temperature, humidity, soil_moisture, stress_index)
        forecast_3_days: 3-day weather forecast data
        location: City or location name
    
    Returns:
        Dictionary with keys: irrigation_required, urgency, explanation
    
    Raises:
        GeminiAPIError: If Gemini API fails (HTTP error, network error)
        IrrigationAdviceError: If response parsing or validation fails
    """
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        error_msg = "GEMINI_API_KEY environment variable is not set"
        logger.error(error_msg)
        raise IrrigationAdviceError(error_msg)
    
    try:
        # Build structured prompt
        prompt = build_irrigation_prompt(sensor_data, forecast_3_days, location)
        logger.debug(f"Building irrigation advice prompt for location: {location}")
        
        # Call Gemini REST API
        try:
            response_text = _call_gemini_api(
                api_key=api_key,
                prompt=prompt,
                location=location,
                timeout_seconds=10.0
            )
        except GeminiAPIError:
            # Re-raise API errors (HTTP, network) as-is
            raise
        
        # Parse JSON response
        try:
            advice = _parse_irrigation_json(response_text)
        except IrrigationAdviceError:
            # Re-raise parsing/validation errors as-is
            raise
        
        logger.info(
            f"Generated irrigation advice for {location}: "
            f"irrigation_required={advice['irrigation_required']}, "
            f"urgency={advice['urgency']}"
        )
        
        return advice
        
    except (GeminiAPIError, IrrigationAdviceError):
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in get_irrigation_advice for {location}: {e}")
        raise IrrigationAdviceError(f"Unexpected error: {str(e)}")


def _call_gemini_api(
    api_key: str,
    prompt: str,
    location: str,
    timeout_seconds: float = 10.0,
) -> str:
    """
    Call the Gemini REST API and return the generated text.
    
    This helper constructs the request payload, performs an HTTPS POST with
    a sane timeout, and extracts the first candidate's first text part.
    
    Args:
        api_key: Gemini API key from environment
        prompt: The irrigation advice prompt
        location: City/location name (for logging only)
        timeout_seconds: HTTP request timeout
    
    Returns:
        Generated text from Gemini
    
    Raises:
        GeminiAPIError: If HTTP or network error occurs
        IrrigationAdviceError: If response is malformed
    """
    body: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.3,  # Lower for consistent JSON output
            "maxOutputTokens": 500,  # Limit output size
        },
    }
    
    # Use API key via query parameter; never log the full URL to avoid leaking it
    url = f"{GEMINI_API_URL}?key={api_key}"
    data = json.dumps(body).encode("utf-8")
    
    http_request = request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    
    try:
        logger.debug(f"Sending request to Gemini for location: {location}")
        with request.urlopen(http_request, timeout=timeout_seconds) as resp:
            resp_body = resp.read().decode("utf-8")
    except error.HTTPError as e:
        # Avoid logging response body, which could contain sensitive information
        error_msg = f"Gemini HTTP error for {location}: status={e.code}, reason={getattr(e, 'reason', '')}"
        logger.error(error_msg)
        raise GeminiAPIError(error_msg)
    except error.URLError as e:
        error_msg = f"Gemini request failed for {location}: {e.reason}"
        logger.error(error_msg)
        raise GeminiAPIError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error calling Gemini for {location}: {e}"
        logger.error(error_msg)
        raise GeminiAPIError(error_msg)
    
    # Parse response JSON
    try:
        payload = json.loads(resp_body)
    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode Gemini response JSON for {location}: {e}"
        logger.error(error_msg)
        raise IrrigationAdviceError(error_msg)
    
    # Extract text from response
    text = _extract_text_from_gemini_response(payload)
    if not text:
        error_msg = f"Gemini response did not contain any text for {location}"
        logger.error(error_msg)
        raise IrrigationAdviceError(error_msg)
    
    return text.strip()


def _extract_text_from_gemini_response(payload: Dict[str, Any]) -> str:
    """
    Extract the first text candidate from a Gemini generateContent response.
    
    Args:
        payload: Parsed JSON response from Gemini API
    
    Returns:
        Extracted text, or empty string if not found
    """
    try:
        candidates = payload.get("candidates") or []
        if not candidates:
            return ""
        first = candidates[0] or {}
        content = first.get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            return ""
        text = parts[0].get("text") or ""
        return text
    except Exception:
        # Be defensive: never let a parsing bug propagate
        logger.exception("Unexpected structure in Gemini response payload")
        return ""


def _parse_irrigation_json(response_text: str) -> Dict[str, Any]:
    """
    Parse and validate irrigation JSON response from Gemini.
    
    Args:
        response_text: Raw text response from Gemini
    
    Returns:
        Validated dictionary with keys: irrigation_required, urgency, explanation
    
    Raises:
        IrrigationAdviceError: If JSON is invalid or fields are missing/invalid
    """
    # Remove markdown code blocks if present
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    
    cleaned_text = cleaned_text.strip()
    
    # Parse JSON
    try:
        advice = json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in Gemini response: {e}"
        logger.error(error_msg)
        raise IrrigationAdviceError(error_msg)
    
    # Validate required fields exist
    required_fields = ["irrigation_required", "urgency", "explanation"]
    for field in required_fields:
        if field not in advice:
            error_msg = f"Missing required field in Gemini response: {field}"
            logger.error(error_msg)
            raise IrrigationAdviceError(error_msg)
    
    # Validate field types and values
    if not isinstance(advice["irrigation_required"], bool):
        error_msg = "irrigation_required must be a boolean"
        logger.error(error_msg)
        raise IrrigationAdviceError(error_msg)
    
    if advice["urgency"] not in ["Low", "Medium", "High"]:
        error_msg = f"urgency must be one of: Low, Medium, High (got: {advice['urgency']})"
        logger.error(error_msg)
        raise IrrigationAdviceError(error_msg)
    
    if not isinstance(advice["explanation"], str) or not advice["explanation"].strip():
        error_msg = "explanation must be a non-empty string"
        logger.error(error_msg)
        raise IrrigationAdviceError(error_msg)
    
    return advice
