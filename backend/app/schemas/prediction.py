"""
Pydantic schemas for prediction requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule")
    include_shap: bool = Field(True, description="Include SHAP feature importance")
    include_heatmap: bool = Field(True, description="Include atom heatmap SVG")
    include_admet: bool = Field(True, description="Include ADMET properties")


class AssayPrediction(BaseModel):
    probability: float
    confidence_set: list[str] = Field(default_factory=lambda: ["SAFE"])
    uncertainty: str = "LOW"


class ShapFeature(BaseModel):
    feature: str
    value: float
    shap: float
    direction: str  # "toxic" or "protective"


class StructuralAlert(BaseModel):
    name: str
    severity: str  # "HIGH", "MED", "LOW"
    atoms: list[int] = Field(default_factory=list)
    description: str = ""
    smarts: str = ""


class ADMETProperties(BaseModel):
    qed: float = 0.0
    lipinski_violations: int = 0
    molecular_weight: float = 0.0
    logp: float = 0.0
    tpsa: float = 0.0
    hbd: int = 0
    hba: int = 0
    rotatable_bonds: int = 0
    aromatic_rings: int = 0
    herg_risk: str = "LOW"
    bbb_penetrant: bool = False
    cyp2d6_inhibitor: bool = False


class PredictResponse(BaseModel):
    smiles: str
    canonical_smiles: str = ""
    composite_risk: float = 0.0
    risk_level: str = "LOW"  # HIGH, MEDIUM, LOW
    predictions: dict[str, AssayPrediction] = Field(default_factory=dict)
    shap_top10: list[ShapFeature] = Field(default_factory=list)
    alerts: list[StructuralAlert] = Field(default_factory=list)
    atom_heatmap_svg: str = ""
    molecule_image_b64: str = ""
    admet: Optional[ADMETProperties] = None


class BatchPredictRequest(BaseModel):
    smiles_list: list[str] = Field(default_factory=list)
    risk_threshold: float = Field(0.5, ge=0.0, le=1.0)
    sort_by: str = "composite_risk"


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse] = Field(default_factory=list)
    total: int = 0
    flagged: int = 0
    mean_risk: float = 0.0
