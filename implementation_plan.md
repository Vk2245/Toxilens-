# ToxiLens — Full Implementation Plan

Drug toxicity prediction platform for CodeCure AI Hackathon (Track A, IIT BHU Spirit'26). Tri-modal ML ensemble + XAI + React frontend + FastAPI backend.

## User Review Required

> [!IMPORTANT]
> **This is an enormous project.** The full vision includes 3 ML models, an ensemble, 4 XAI techniques, 8 API endpoints, 5 React pages, LLM integration, PDF export, Docker, and HF Spaces deployment. We need to decide a practical scope.

> [!WARNING]
> **ML training requires GPU and datasets.** The Tox21 dataset must be downloaded from Kaggle. ChemBERTa fine-tuning and GNN training need a CUDA GPU. Do you have these available locally, or should we focus on the **web application shell** (backend + frontend) first and plug in models later?

> [!IMPORTANT]
> **API keys needed.** The LLM report feature needs an Anthropic API key (or Groq/Mistral fallback). Do you have one ready?

## Proposed Implementation Strategy

Given hackathon time constraints, I propose building in **priority waves** — each wave produces a working, demo-able product:

---

### Wave 1: Foundation (Backend + Frontend Shell) — ~3 hours
Get the project skeleton running end-to-end with mock predictions.

#### Backend (`backend/`)
- [NEW] `backend/app/main.py` — FastAPI entry with CORS, health check, model preload hooks
- [NEW] `backend/app/core/config.py` — Settings via pydantic-settings (env vars, model paths)
- [NEW] `backend/app/schemas/prediction.py` — Pydantic request/response models
- [NEW] `backend/app/preprocessing/rdkit_utils.py` — SMILES validation, standardization, 2D image generation
- [NEW] `backend/app/preprocessing/descriptors.py` — 200+ RDKit descriptor computation
- [NEW] `backend/app/preprocessing/fingerprints.py` — Morgan (ECFP4) + MACCS fingerprints
- [NEW] `backend/app/api/routes_predict.py` — `POST /predict` endpoint
- [NEW] `backend/app/features/structural_alerts.py` — 150+ SMARTS toxicophore patterns
- [NEW] `backend/app/features/admet_predictor.py` — ADMET property computation via RDKit
- [NEW] `requirements.txt` — Pinned Python dependencies

#### Frontend (`frontend/`)
- React 18 + Vite + TypeScript project
- [NEW] Pages: `SingleAnalysis.tsx` (main page with SMILES input, preset buttons)
- [NEW] Components: `MoleculeViewer`, `ToxicityRadar`, `ShapChart`, `AlertBadges`, `AdmetPanel`
- [NEW] `api/predict.ts` — API client
- [NEW] Global styles (dark theme, premium design)

#### Config Files
- [NEW] `.env.example`, `.gitignore`, `LICENSE`
- [NEW] `docker-compose.yml`, `docker/backend.Dockerfile`, `docker/frontend.Dockerfile`

---

### Wave 2: ML Models + Real Predictions — ~4 hours
Replace mocks with actual trained models.

#### ML Pipeline (`ml/`)
- [NEW] `ml/scripts/preprocess_tox21.py` — Download/process Tox21 with scaffold split
- [NEW] `ml/scripts/train_descriptor.py` — LightGBM 12-head training with SHAP
- [NEW] `backend/app/models/descriptor_model.py` — LightGBM inference wrapper
- [NEW] `backend/app/explainability/shap_utils.py` — TreeExplainer wrapper

---

### Wave 3: GNN + ChemBERTa + Ensemble — ~4 hours
- [NEW] `backend/app/models/gnn_model.py` — AttentiveFP with joint correlation loss
- [NEW] `backend/app/models/transformer_model.py` — ChemBERTa-2 fine-tuned wrapper
- [NEW] `backend/app/models/ensemble.py` — Weighted fusion + conformal prediction
- [NEW] `backend/app/preprocessing/graph_builder.py` — RDKit mol → PyG Data
- [NEW] `backend/app/explainability/captum_utils.py` — IntegratedGradients for GNN
- [NEW] `backend/app/explainability/heatmap_renderer.py` — Atom importance → colored SVG

---

### Wave 4: Advanced Features — ~3 hours  
- [NEW] LLM report generation (`report/llm_reporter.py`)
- [NEW] PDF export (`report/pdf_exporter.py`)
- [NEW] UMAP chemical space (`features/umap_search.py`, `ChemicalSpace.tsx`)
- [NEW] Batch screening (`routes_predict.py` batch endpoint, `BatchScreening.tsx`)
- [NEW] De-risking lab (`features/derisking.py`, `DeRiskLab.tsx`)
- [NEW] Multi-molecule compare (`MultiCompare.tsx`)

---

### Wave 5: Polish & Deploy — ~2 hours
- Docker compose testing
- HF Spaces deployment  
- Demo GIF recording
- README finalization

## Open Questions

> [!IMPORTANT]
> 1. **Do you have a CUDA GPU available locally?** This determines whether we train models or use pre-computed mock weights.
> 2. **Do you have the Tox21 dataset already?** Or should I add download steps?
> 3. **Do you have an Anthropic/Groq API key?** For LLM report generation.
> 4. **Should I start with Wave 1 immediately?** Building the full backend + frontend shell so you have a working demo ASAP, then layer in real ML?
> 5. **Tailwind vs Vanilla CSS?** The plan says Tailwind but my default is vanilla CSS for maximum control. Preference?

## Verification Plan

### Automated Tests
- `curl` smoke tests on all API endpoints
- Frontend dev server launches without errors
- Docker compose builds and runs

### Manual Verification
- Single molecule prediction end-to-end (Aspirin SMILES)
- Browser walkthrough of all 5 pages
- PDF report generation test
