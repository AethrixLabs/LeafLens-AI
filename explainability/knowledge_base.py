"""
Deterministic farmer-friendly disease knowledge base for crop disease explanations.

This module loads disease knowledge from a JSON configuration file and provides
structured explanations (summary, cause, symptoms, spread, treatment, prevention)
without external APIs or LLMs. Designed for offline, production use and future
hybrid extension (e.g., optional LLM for ?detailed=true).

Usage:
    kb = KnowledgeBase("config/disease_knowledge.json")
    explanation = kb.get_explanation("rice", "Blast")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Type alias for the structured explanation returned by get_explanation.
ExplanationDict = Dict[str, str]

_REQUIRED_KEYS = frozenset({"summary", "cause", "symptoms", "spread", "treatment", "prevention"})


class KnowledgeBase:
    """
    Loads and serves disease knowledge from a JSON file.

    JSON is loaded once at initialization and held in memory (singleton-style
    per instance). Handles missing crop or disease with clear ValueErrors;
    does not expose raw stack traces or crash the server.

    Attributes:
        _data: Parsed JSON structure: { crop: { "diseases": { disease: { ... } } } }
    """

    def __init__(self, config_path: str | Path) -> None:
        """
        Initialize the knowledge base and load JSON from the given path.

        Args:
            config_path: Path to disease_knowledge.json (string or Path).

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the file is not valid JSON or has invalid structure.
        """
        self._path = Path(config_path)
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load and validate JSON from config path. Called once from __init__."""
        knowledge_dir = Path("knowledge")
        raw: Dict[str, Any] | None = None
        if knowledge_dir.exists() and knowledge_dir.is_dir():
            merged: Dict[str, Any] = {}
            if self._path.exists():
                try:
                    with open(self._path, "r", encoding="utf-8") as f:
                        base = json.load(f)
                        if isinstance(base, dict):
                            merged.update(base)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in disease knowledge config: %s", self._path)

            found_any = False
            for crop_dir in knowledge_dir.iterdir():
                if not crop_dir.is_dir():
                    continue
                crop_key = crop_dir.name
                diseases: Dict[str, Any] = {}
                for jf in crop_dir.glob("*.json"):
                    found_any = True
                    try:
                        with open(jf, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        diseases[jf.stem] = data
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in knowledge file: %s", jf)
                if diseases:
                    if crop_key in merged and isinstance(merged[crop_key], dict) and "diseases" in merged[crop_key]:
                        merged[crop_key]["diseases"].update(diseases)
                    else:
                        merged[crop_key] = {"diseases": diseases}

            if found_any:
                raw = merged

        if raw is None:
            if not self._path.exists():
                logger.error("Disease knowledge config not found: %s", self._path)
                raise FileNotFoundError(f"Disease knowledge config not found: {self._path}")

            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON in disease knowledge config: %s", e)
                raise ValueError(f"Invalid JSON in disease knowledge config: {e}") from e

        if not isinstance(raw, dict):
            logger.error("Disease knowledge root must be a JSON object")
            raise ValueError("Disease knowledge root must be a JSON object")

        for crop, crop_data in raw.items():
            if not isinstance(crop_data, dict):
                logger.warning("Skipping invalid crop entry '%s': not an object", crop)
                continue
            diseases = crop_data.get("diseases")
            if not isinstance(diseases, dict):
                logger.warning("Skipping crop '%s': missing or invalid 'diseases' object", crop)
                continue
            for disease_name, disease_info in diseases.items():
                if not isinstance(disease_info, dict):
                    logger.warning(
                        "Skipping disease '%s' under crop '%s': not an object",
                        disease_name, crop
                    )
                    continue
                missing = _REQUIRED_KEYS - set(disease_info.keys())
                if missing:
                    logger.warning(
                        "Disease '%s' under crop '%s' missing keys: %s",
                        disease_name, crop, missing
                    )

        self._data = raw
        crops = list(self._data.keys())
        logger.info(
            "KnowledgeBase loaded from %s: %d crop(s) available",
            self._path, len(crops)
        )
        logger.debug("Available crops: %s", crops)

    def get_explanation(self, crop: str, disease: str) -> ExplanationDict:
        """
        Return a structured explanation for the given crop and disease.

        Args:
            crop: Crop identifier (e.g., "rice").
            disease: Disease label (e.g., "Blast", "Bacterial_blight").

        Returns:
            Dictionary with keys: summary, cause, symptoms, spread, treatment, prevention.
            Values are non-empty strings where knowledge exists; missing keys
            are returned as empty string to preserve contract.

        Raises:
            ValueError: If crop is not in the knowledge base.
            ValueError: If disease is not found for the given crop.
        """
        if not crop or not isinstance(crop, str):
            logger.warning("get_explanation called with invalid crop: %r", crop)
            raise ValueError("Crop must be a non-empty string")

        if not disease or not isinstance(disease, str):
            logger.warning("get_explanation called with invalid disease: %r", disease)
            raise ValueError("Disease must be a non-empty string")

        crop_key = crop.strip()
        if crop_key not in self._data:
            available = list(self._data.keys())
            logger.warning("Crop '%s' not found in knowledge base. Available: %s", crop_key, available)
            raise ValueError(
                f"Crop '{crop_key}' not found in knowledge base. "
                f"Available crops: {available}"
            )

        crop_data = self._data[crop_key]
        diseases = crop_data.get("diseases") if isinstance(crop_data, dict) else None
        if not isinstance(diseases, dict):
            logger.warning("No diseases defined for crop '%s'", crop_key)
            raise ValueError(f"No disease knowledge available for crop '{crop_key}'")

        # Match disease case-insensitively for robustness; prefer exact match
        disease_clean = disease.strip()
        if disease_clean in diseases:
            entry = diseases[disease_clean]
        else:
            lower_map = {k.lower(): k for k in diseases}
            if disease_clean.lower() in lower_map:
                entry = diseases[lower_map[disease_clean.lower()]]
            else:
                available_diseases = list(diseases.keys())
                logger.warning(
                    "Disease '%s' not found for crop '%s'. Available: %s",
                    disease_clean, crop_key, available_diseases
                )
                raise ValueError(
                    f"Disease '{disease_clean}' not found for crop '{crop_key}'. "
                    f"Available diseases: {available_diseases}"
                )

        if not isinstance(entry, dict):
            logger.warning("Invalid entry for disease '%s' under crop '%s'", disease_clean, crop_key)
            raise ValueError(
                f"Disease '{disease_clean}' has invalid data for crop '{crop_key}'"
            )

        result: ExplanationDict = {
            "summary": _str_or_empty(entry.get("summary")),
            "cause": _str_or_empty(entry.get("cause")),
            "symptoms": _str_or_empty(entry.get("symptoms")),
            "spread": _str_or_empty(entry.get("spread")),
            "treatment": _str_or_empty(entry.get("treatment")),
            "prevention": _str_or_empty(entry.get("prevention")),
        }
        logger.debug("Returning explanation for crop=%s, disease=%s", crop_key, disease_clean)
        return result

    def has_crop(self, crop: str) -> bool:
        """Return True if the crop exists in the knowledge base."""
        return isinstance(crop, str) and crop.strip() in self._data

    def has_disease(self, crop: str, disease: str) -> bool:
        """Return True if the crop and disease exist (no ValueError)."""
        try:
            self.get_explanation(crop, disease)
            return True
        except ValueError:
            return False


def _str_or_empty(value: Any) -> str:
    """Return value as string, or empty string if None or not string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
