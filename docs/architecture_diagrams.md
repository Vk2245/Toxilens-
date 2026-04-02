# ToxiLens Architecture Diagrams

This document provides detailed architecture diagrams for the ToxiLens platform.

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FRONTEND (React 18)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Single     │  │  Chemical    │  │    Batch     │  │   De-Risk    │   │
│  │  Analysis    │  │   Space      │  │  Screening   │  │     Lab      │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
└─────────┼──────────────────┼──────────────────┼──────────────────┼──────────┘
          │                  │                  │                  │
          └──────────────────┴──────────────────┴──────────────────┘
                                    │
                              HTTP/JSON REST API
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BACKEND (FastAPI)                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        API ENDPOINTS                                 │   │
│  │  /predict  /predict_batch  /generate_report  /derisk  /similar      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                   PREPROCESSING PIPELINE                             │   │
│  │  SMILES → Standardize → Descriptors + Fingerprints + Graph          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│         ┌──────────────────────────┼──────────────────────────┐             │
│         ▼                          ▼                          ▼             │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐         │
│  │ ChemBERTa-2 │          │     GNN     │          │  LightGBM   │         │
│  │ Transformer │          │ Multi-task  │          │  Gradient   │         │
│  │  768-dim    │          │  256-dim    │          │  Boosting   │         │
│  └──────┬──────┘          └──────┬──────┘          └──────┬──────┘         │
│         └──────────────────────┬─┴────────────────────────┘                │
│                                ▼                                            │
│                    ┌───────────────────────┐                                │
│                    │  Ensemble Fusion      │                                │
│                    │  + Conformal Predict  │                                │
│                    └───────────┬───────────┘                                │
│                                │                                            │
│         ┌──────────┬───────────┼───────────┬──────────┐                    │
│         ▼          ▼           ▼           ▼          ▼                    │
│     ┌──────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐             │
│     │ SHAP │  │Captum  │  │SMARTS  │  │ ADMET  │  │  UMAP  │             │
│     └──────┘  └────────┘  └────────┘  └────────┘  └────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

See README.md for full architecture details.


## Data Flow Diagram

### Single Molecule Prediction Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 1: User Input                                                          │
│   User submits SMILES via frontend → POST /predict                          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ Step 2: SMILES Validation & Standardization                                │
│   • Parse SMILES with RDKit                                                 │
│   • Neutralize charges                                                      │
│   • Remove salts                                                            │
│   • Canonicalize tautomers                                                  │
│   • Generate 2D coordinates                                                 │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ Step 3: Feature Engineering (Parallel)                                     │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│   │  Descriptors    │  │  Fingerprints   │  │  Graph Build    │           │
│   │  • 200+ RDKit   │  │  • Morgan 2048  │  │  • Node feats   │           │
│   │  • MW, logP     │  │  • MACCS 167    │  │  • Edge feats   │           │
│   │  • TPSA, etc.   │  │  • Radius=2     │  │  • Adjacency    │           │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘           │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ Step 4: Model Inference (Parallel)                                         │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│   │  ChemBERTa-2    │  │      GNN        │  │    LightGBM     │           │
│   │  Tokenize       │  │  AttentiveFP    │  │  12 classifiers │           │
│   │  → Encode       │  │  4 layers       │  │  Tree ensemble  │           │
│   │  → 12 logits    │  │  → 12 logits    │  │  → 12 probs     │           │
│   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘           │
└────────────┼─────────────────────┼─────────────────────┼────────────────────┘
             │                     │                     │
             └─────────────────────┼─────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────────┐
│ Step 5: Ensemble Fusion                                                    │
│   • Convert probs to logits                                                 │
│   • Weighted fusion: w₁·logit₁ + w₂·logit₂ + w₃·logit₃                    │
│   • Apply sigmoid → 12 probabilities                                        │
│   • MAPIE conformal wrapper → prediction sets + uncertainty                │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ Step 6: Explainability (Parallel)                                          │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│   │  Captum         │  │  SHAP           │  │  SMARTS         │           │
│   │  IntGrad on GNN │  │  TreeExplainer  │  │  Pattern match  │           │
│   │  → Atom scores  │  │  → Top 10 desc  │  │  → Alerts list  │           │
│   │  → Heatmap PNG  │  │  → Importance   │  │  → Severity     │           │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘           │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ Step 7: ADMET Computation                                                  │
│   • QED, Lipinski violations                                                │
│   • BBB penetration, oral bioavailability                                   │
│   • CYP inhibition, hERG risk                                               │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ Step 8: Response Assembly                                                  │
│   JSON response with:                                                       │
│   • 12 assay predictions + uncertainty                                      │
│   • Composite risk score + risk level                                       │
│   • SHAP top 10 features                                                    │
│   • Atom heatmap (base64 PNG)                                               │
│   • Structural alerts                                                       │
│   • ADMET properties                                                        │
│   • Conformal intervals                                                     │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ Step 9: Frontend Rendering                                                 │
│   • Radar chart (12 assays)                                                 │
│   • Heatmap overlay on molecule                                             │
│   • SHAP bar chart                                                          │
│   • Alert badges                                                            │
│   • ADMET panel                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## ML Pipeline Architecture

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Data Acquisition                                                         │
│    Tox21 dataset (Kaggle) → 11,764 compounds × 12 assays                   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ 2. Preprocessing                                                            │
│    • SMILES standardization (RDKit)                                         │
│    • Descriptor computation (200+ features)                                 │
│    • Fingerprint generation (Morgan + MACCS)                                │
│    • Graph construction (PyG Data objects)                                  │
│    • Feature scaling (StandardScaler)                                       │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ 3. Scaffold Splitting                                                       │
│    Bemis-Murcko scaffold extraction → 80/10/10 train/val/test              │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ 4. Model Training (Parallel)                                               │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│   │  LightGBM       │  │      GNN        │  │  ChemBERTa-2    │           │
│   │  • 12 binary    │  │  • AttentiveFP  │  │  • Fine-tune    │           │
│   │  • Class wts    │  │  • Joint loss   │  │  • LR: 2e-5     │           │
│   │  • 1000 trees   │  │  • 100 epochs   │  │  • 5-8 epochs   │           │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘           │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ 5. Ensemble Weight Optimization                                            │
│    Nelder-Mead on validation set → optimal weights [w₁, w₂, w₃]           │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ 6. Conformal Calibration                                                   │
│    MAPIE calibration on held-out set → α=0.15 (85% coverage)               │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ 7. Evaluation                                                               │
│    Test set AUROC per assay → Mean AUROC: 0.847                            │
└─────────────────────────────────────────────────────────────────────────────┘
```



## Inference Pipeline Architecture

```
Input: SMILES String
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ RDKit Preprocessing                                                         │
│   • Validate SMILES syntax                                                  │
│   • Standardize (neutralize, desalt, canonicalize)                          │
│   • Generate 2D coordinates                                                 │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
         ┌──────────────┐ ┌──────────┐ ┌──────────────┐
         │ Descriptors  │ │Fingerprnt│ │    Graph     │
         │   [200]      │ │ [2215]   │ │  PyG Data    │
         └──────┬───────┘ └────┬─────┘ └──────┬───────┘
                │              │              │
                ▼              ▼              ▼
         ┌──────────────┐ ┌──────────┐ ┌──────────────┐
         │  LightGBM    │ │ChemBERTa │ │     GNN      │
         │  12 trees    │ │ Encoder  │ │  AttentiveFP │
         │  → 12 probs  │ │→12 logits│ │  → 12 logits │
         └──────┬───────┘ └────┬─────┘ └──────┬───────┘
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Logit Fusion        │
                    │  w₁·L₁ + w₂·L₂ + w₃·L₃│
                    │  → Sigmoid           │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  MAPIE Conformal     │
                    │  → Prediction sets   │
                    │  → Uncertainty       │
                    └──────────┬───────────┘
                               ▼
Output: 12 Probabilities + Uncertainty + Explanations
```

## Explainability Pipeline

```
Prediction Results
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         ▼                 ▼                 ▼                 ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Captum          │ │ SHAP         │ │ SMARTS       │ │ ADMET        │
│ IntegratedGrad  │ │ TreeExplainer│ │ Pattern Match│ │ Properties   │
└────────┬────────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
         │                 │                 │                 │
         ▼                 ▼                 ▼                 ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Per-atom        │ │ Top 10       │ │ Detected     │ │ QED, Lipinski│
│ importance      │ │ descriptors  │ │ alerts with  │ │ BBB, hERG    │
│ scores [0,1]    │ │ with SHAP    │ │ severity     │ │ CYP, etc.    │
└────────┬────────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
         │                 │                 │                 │
         ▼                 │                 │                 │
┌─────────────────┐        │                 │                 │
│ Heatmap         │        │                 │                 │
│ Renderer        │        │                 │                 │
│ • Map to colors │        │                 │                 │
│ • Draw molecule │        │                 │                 │
│ • Export PNG    │        │                 │                 │
└────────┬────────┘        │                 │                 │
         │                 │                 │                 │
         └─────────────────┴─────────────────┴─────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  Combined Explanation    │
                    │  • Atom heatmap          │
                    │  • SHAP chart            │
                    │  • Alert badges          │
                    │  • ADMET panel           │
                    └──────────────────────────┘
```

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PreprocessingPipeline                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ rdkit_utils  │→ │ descriptors  │→ │ fingerprints │→ │graph_builder │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EnsembleModel                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│  │descriptor_mdl│  │   gnn_model  │  │transformer_mdl│                     │
│  └──────────────┘  └──────────────┘  └──────────────┘                     │
│                    ┌──────────────┐                                         │
│                    │ ensemble.py  │                                         │
│                    └──────┬───────┘                                         │
└───────────────────────────┼─────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ExplainabilityEngine                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│  │  shap_utils  │  │captum_utils  │  │heatmap_render│                     │
│  └──────────────┘  └──────────────┘  └──────────────┘                     │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FeatureModules                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │structural_   │  │admet_        │  │umap_search   │  │derisking     │   │
│  │alerts        │  │predictor     │  │              │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ReportGeneration                                   │
│  ┌──────────────┐  ┌──────────────┐                                        │
│  │llm_reporter  │→ │pdf_exporter  │                                        │
│  │(Claude API)  │  │(WeasyPrint)  │                                        │
│  └──────────────┘  └──────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Deployment Architecture

### Docker Compose Setup

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Docker Host                                        │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Frontend Container (nginx)                                            │  │
│  │   • React build artifacts                                             │  │
│  │   • Port 3000 → 80                                                    │  │
│  │   • Serves static files                                               │  │
│  └────────────────────────────┬──────────────────────────────────────────┘  │
│                               │                                             │
│                               │ HTTP proxy                                  │
│                               │                                             │
│  ┌────────────────────────────▼──────────────────────────────────────────┐  │
│  │ Backend Container (uvicorn)                                           │  │
│  │   • FastAPI application                                               │  │
│  │   • Port 8000                                                         │  │
│  │   • ML models loaded in memory                                        │  │
│  │   • Volume: ./ml/artifacts → /app/ml/artifacts                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Shared Volume                                                         │  │
│  │   • Model artifacts (.pt, .pkl)                                       │  │
│  │   • UMAP embeddings                                                   │  │
│  │   • SHAP background set                                               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Hugging Face Spaces Deployment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Hugging Face Spaces                                    │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Space Container (T4 GPU)                                              │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Gradio / FastAPI App                                            │ │  │
│  │  │   • Combined frontend + backend                                 │ │  │
│  │  │   • Models loaded on T4 GPU                                     │ │  │
│  │  │   • Port 7860 (Spaces default)                                  │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Secrets (Environment Variables)                                 │ │  │
│  │  │   • ANTHROPIC_API_KEY                                           │ │  │
│  │  │   • GROQ_API_KEY (fallback)                                     │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Persistent Storage                                              │ │  │
│  │  │   • Model artifacts (Git LFS)                                   │ │  │
│  │  │   • UMAP embeddings                                             │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Public URL: https://toxilens.hf.space                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Model Architecture Details

### ChemBERTa-2 Architecture

```
Input: "CC(=O)Oc1ccccc1C(=O)O"
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ SmilesTokenizer                                                             │
│   • Vocabulary: 591 tokens                                                  │
│   • Max length: 512                                                         │
│   • Special tokens: [CLS], [SEP], [PAD], [MASK]                            │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
         Token IDs: [101, 45, 67, 89, ..., 102]
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ RoBERTa Encoder                                                             │
│   • 12 transformer layers                                                   │
│   • 12 attention heads per layer                                            │
│   • Hidden size: 768                                                        │
│   • Intermediate size: 3072                                                 │
│   • Pretrained on 77M SMILES (PubChem)                                      │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
         CLS Token Embedding [768]
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Classification Head                                                         │
│   • Dropout(0.1)                                                            │
│   • Linear(768 → 12)                                                        │
│   • Sigmoid activation                                                      │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
         12 Toxicity Probabilities [0, 1]
```

### GNN Architecture (AttentiveFP)

```
Input: Molecular Graph
  Nodes: [atomic_num, degree, hybridization, aromatic, in_ring, charge, num_H]
  Edges: [bond_type, conjugated, in_ring, stereo]
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Graph Convolution Layers                                                    │
│   Layer 1: AttentiveFPConv(in → 256)  + ReLU                               │
│   Layer 2: AttentiveFPConv(256 → 256) + ReLU                               │
│   Layer 3: AttentiveFPConv(256 → 256) + ReLU                               │
│   Layer 4: AttentiveFPConv(256 → 256) + ReLU                               │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
         Node Embeddings [num_atoms, 256]
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Global Pooling                                                              │
│   • Global Mean Pool → [256]                                                │
│   • Global Max Pool  → [256]                                                │
│   • Concatenate      → [512]                                                │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Classification Head                                                         │
│   • Dropout(0.3)                                                            │
│   • Linear(512 → 12)                                                        │
│   • Sigmoid activation                                                      │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
         12 Toxicity Probabilities [0, 1]
```

### LightGBM Architecture

```
Input: Concatenated Features
  [200 descriptors] + [2048 Morgan bits] + [167 MACCS keys] = 2415 features
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Feature Scaling                                                             │
│   StandardScaler (fitted on training set)                                   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
         Scaled Features [2415]
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ LGBMClassifier  │ ... │ LGBMClassifier  │ ... │ LGBMClassifier  │
│   (NR-AR)       │     │   (NR-AhR)      │     │   (NR-PPAR)     │
│ • 1000 trees    │     │ • 1000 trees    │     │ • 1000 trees    │
│ • Max depth: 7  │     │ • Max depth: 7  │     │ • Max depth: 7  │
│ • Learning: 0.1 │     │ • Learning: 0.1 │     │ • Learning: 0.1 │
│ • pos_weight    │     │ • pos_weight    │     │ • pos_weight    │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                                 ▼
         12 Toxicity Probabilities [0, 1]
```

## API Request/Response Flow

```
Client (Browser/Python)
         │
         │ POST /predict
         │ {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FastAPI Router                                                              │
│   • Validate request (Pydantic)                                             │
│   • Extract SMILES                                                          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Prediction Service                                                          │
│   • Preprocess SMILES                                                       │
│   • Run ensemble inference                                                  │
│   • Compute explanations                                                    │
│   • Compute ADMET                                                           │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Response Builder                                                            │
│   • Assemble JSON response                                                  │
│   • Encode heatmap as base64                                                │
│   • Add metadata (timestamp, version)                                       │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 │ 200 OK
                                 │ {predictions, risk, shap, alerts, ...}
                                 ▼
Client receives JSON response
```

## Technology Stack Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PRESENTATION LAYER                                                          │
│   React 18 · TypeScript · Tailwind CSS · Vite                              │
│   Recharts · Plotly.js · Axios                                              │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ API LAYER                                                                   │
│   FastAPI · Uvicorn · Pydantic v2 · CORS                                   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ BUSINESS LOGIC LAYER                                                        │
│   Preprocessing · Model Inference · Explainability · Feature Modules        │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ ML FRAMEWORK LAYER                                                          │
│   PyTorch 2.x · PyTorch Geometric · Transformers · LightGBM                │
│   Captum · SHAP · MAPIE                                                     │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ CHEMISTRY LAYER                                                             │
│   RDKit · Molecular descriptors · Fingerprints · SMARTS                     │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│ INFRASTRUCTURE LAYER                                                        │
│   Docker · Docker Compose · CUDA · Linux                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*For more details, see the main [README.md](../README.md) and [design.md](../.kiro/specs/toxilens-platform/design.md)*
