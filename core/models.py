# --------------------------------------------------------------------------------------
# Pydantic models
# --------------------------------------------------------------------------------------
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class ReportOptions(BaseModel):
    detection_threshold: float = Field(0.5, ge=0.0, le=1.0)
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    top_k_retrieval: int = Field(10, ge=1, le=50)
    top_k_final: int = Field(5, ge=1, le=20)

class ReportArtifacts(BaseModel):
    thematic_plot_b64: str
    nlp_plot_b64: str
    # thematic_plot_data_url: str
    # nlp_plot_data_url: str

class ReportResult(BaseModel):
    patient_id: str
    metadata: Dict[str, Any]
    theme_scores: Dict[str, float]
    nlp_results: Dict[str, Any]
    rag_summary: str
    artifacts: ReportArtifacts

class GenerateJsonPayload(BaseModel):
    patient_data: Dict[str, Any]
    options: Optional[ReportOptions] = None