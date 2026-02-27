from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.sql import func

from database.db import Base


class Prediction(Base):
	"""ORM model representing a single prediction and its metadata.

	This table is designed for production logging of model predictions,
	with fields to support future analytics, feedback loops, and
	model versioning.
	"""

	__tablename__ = "predictions"

	id: int = Column(Integer, primary_key=True, index=True)
	crop: str = Column(String, index=True, nullable=False)
	image_path: str = Column(String, nullable=False)
	image_hash: str = Column(String(64), index=True, nullable=False)
	predicted_label: str = Column(String, nullable=False)
	confidence: float = Column(Float, nullable=False)
	model_version: str = Column(String, nullable=False)
	created_at: datetime = Column(
		DateTime(timezone=True),
		nullable=False,
		server_default=func.now(),
	)
	user_feedback: Optional[str] = Column(String, nullable=True)

	def __repr__(self) -> str:
		"""Return a concise string representation for debugging/logging."""
		return (
			f"Prediction(id={self.id!r}, crop={self.crop!r}, "
			f"predicted_label={self.predicted_label!r}, "
			f"confidence={self.confidence!r}, "
			f"model_version={self.model_version!r})"
		)


class LLMExplanation(Base):
	"""Cached LLM-generated explanations for crop+disease+language combinations.

	This table is intentionally separate from Prediction to keep
	explanation caching orthogonal to prediction logging.
	"""

	__tablename__ = "llm_explanations"
	__table_args__ = (
		UniqueConstraint("crop", "disease", "language", "model_name", name="uq_llm_explanation"),
	)

	id: int = Column(Integer, primary_key=True, index=True)
	crop: str = Column(String, index=True, nullable=False)
	disease: str = Column(String, index=True, nullable=False)
	language: str = Column(String, nullable=False, default="English", index=True)
	model_name: str = Column(String, nullable=False)
	explanation_text: str = Column(Text, nullable=False)
	created_at: datetime = Column(
		DateTime(timezone=True),
		nullable=False,
		server_default=func.now(),
	)

	def __repr__(self) -> str:
		"""Return a concise string representation for debugging/logging."""
		return (
			f"LLMExplanation(id={self.id!r}, crop={self.crop!r}, "
			f"disease={self.disease!r}, language={self.language!r}, model_name={self.model_name!r})"
		)
