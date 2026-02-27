"""
Weather and Mandi API routes – FastAPI implementation.

Reuses WeatherAPI, MandiPriceAPI, Translator, and LocationDetector.
Keeps response structure identical to the original Flask implementation.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger("leaflens")


def _translate_weather_labels(weather: Optional[Dict], language_code: str) -> Dict:
    """Translate weather parameter labels using Translator."""
    if not weather or language_code == "en":
        return weather or {}
    try:
        from i18n.translator import Translator

        translator = Translator()
        weather_data = {"weather": weather}
        translated = translator.translate_weather_response(weather_data, language_code)
        return translated.get("weather", weather)
    except Exception as e:
        logger.warning("Translation failed, using original weather labels: %s", e)
        return weather


router = APIRouter(prefix="/api", tags=["weather", "mandi"])


@router.get("/weather")
async def get_weather(
    location: Optional[str] = Query(None, alias="location"),
    language_code: str = Query("en", alias="language_code"),
) -> Dict[str, Any]:
    """
    GET /api/weather?location=<city>&language_code=<code>

    Optional query params: location (city name), language_code (default: en).
    Returns: success, location, weather, language.
    """
    try:
        language_code = (language_code or "en").strip().lower()

        from services.weather.location_detector import LocationDetector

        detector = LocationDetector()
        if location and isinstance(location, str) and location.strip():
            city = location.strip()
            loc = detector.get_location_by_city(city)
            if not loc:
                loc = {
                    "city": city,
                    "country": "Unknown",
                    "region": "",
                    "latitude": 0,
                    "longitude": 0,
                }
        else:
            loc = detector.detect_location_with_fallback()

        location_str = loc.get("city", "Delhi")

        try:
            from config.settings import OPENWEATHERMAP_API_KEY as api_key
        except ImportError:
            api_key = None

        if not api_key:
            logger.error("Weather API key not configured")
            raise HTTPException(
                status_code=500,
                detail="Weather API key not configured",
            )

        from services.weather.weather_api import WeatherAPI

        weather_api = WeatherAPI(api_key=api_key)
        raw_weather = weather_api.get_current_weather(location_str)

        if not raw_weather:
            logger.warning("Weather fetch failed for %s", location_str)
            raise HTTPException(
                status_code=500,
                detail="Weather service unavailable",
            )

        coord = raw_weather.get("coord") or {}
        sys_data = raw_weather.get("sys") or {}
        lat = coord.get("lat")
        lon = coord.get("lon")
        country = sys_data.get("country")
        if lat is not None:
            loc["latitude"] = lat
        if lon is not None:
            loc["longitude"] = lon
        if country is not None:
            loc["country"] = country

        weather_params = weather_api.extract_weather_params(raw_weather)
        weather_display = {k: v for k, v in weather_params.items() if v is not None}

        if language_code != "en":
            weather_display = _translate_weather_labels(weather_display, language_code)

        return {
            "success": True,
            "location": loc,
            "weather": weather_display,
            "language": language_code,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Weather request failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your request",
        )


@router.get("/mandi/crops")
async def get_mandi_crops():
    """GET /api/mandi/crops – Return all available crops."""
    try:
        from services.mandi.mandi_api import MandiPriceAPI

        api = MandiPriceAPI()
        crops = api.get_all_crops()
        return crops

    except Exception as e:
        logger.exception("Mandi crops fetch failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "An error occurred processing your request"},
        )


@router.get("/mandi/{crop_id}")
async def get_mandi_crop(
    crop_id: str,
    language_code: str = Query("en", alias="language_code"),
) -> Dict[str, Any]:
    """GET /api/mandi/<crop_id> – Return price comparison for crop."""
    try:
        from services.mandi.mandi_api import MandiPriceAPI

        api = MandiPriceAPI()
        result = api.get_price_comparison(crop_id.strip().lower())

        if not result.get("success"):
            return JSONResponse(status_code=404, content=result)

        return result

    except Exception as e:
        logger.exception("Mandi price fetch failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "An error occurred processing your request"},
        )
