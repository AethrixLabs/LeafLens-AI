from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger("leaflens.ui")

router = APIRouter()


def _get_templates(request: Request) -> Jinja2Templates:
	"""Get the shared Jinja templates instance from app state.

	`main.py` configures this in production. This fallback keeps the UI resilient
	in tests or alternate startup paths.
	"""
	templates = getattr(request.app.state, "templates", None)
	if isinstance(templates, Jinja2Templates):
		return templates
	logger.warning("Jinja2Templates not found in app.state; using fallback directory")
	return Jinja2Templates(directory="app/ui/templates")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> Any:
	templates = _get_templates(request)
	return templates.TemplateResponse(
		"index.html",
		{
			"request": request,
			"app_name": "LeafLens",
			"tagline": "AI-Powered Crop Disease Detection & Smart Explanation Engine",
		},
	)


@router.get("/weather", response_class=HTMLResponse)
async def weather(request: Request) -> Any:
	templates = _get_templates(request)
	return templates.TemplateResponse(
		"weather.html",
		{
			"request": request,
			"app_name": "LeafLens",
			"tagline": "AI-Powered Crop Disease Detection & Smart Explanation Engine",
		},
	)


@router.get("/mandi", response_class=HTMLResponse)
async def mandi(request: Request) -> Any:
	templates = _get_templates(request)
	return templates.TemplateResponse(
		"mandi.html",
		{
			"request": request,
			"app_name": "LeafLens",
			"tagline": "AI-Powered Crop Disease Detection & Smart Explanation Engine",
		},
	)


@router.get("/insight", response_class=HTMLResponse)
async def insight(request: Request) -> Any:
	templates = _get_templates(request)
	return templates.TemplateResponse(
		"leaflens_insight.html",
		{
			"request": request,
			"app_name": "LeafLens",
			"tagline": "AI-Powered Crop Disease Detection & Smart Explanation Engine",
		},
	)

