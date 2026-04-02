# Implementation Plan: ToxiLens Platform

## Overview

This implementation plan breaks down the ToxiLens multi-modal toxicity prediction platform into discrete, incremental coding tasks. The platform combines three ML models (ChemBERTa-2 transformer, AttentiveFP GNN, and LightGBM) with comprehensive explainability features, a FastAPI backend, and a React TypeScript frontend.

Implementation follows a logical progression: project setup → data preprocessing → model training → backend core → explainability → advanced features → frontend → deployment. Each task builds on previous work, validates functionality early, and includes checkpoints at major milestones.

## Tasks

- [x] 1. Set up project infrastructure and configuration
  - Create directory structure: backend/, frontend/, ml/, docker/, ml/data/, ml/artifacts/
  - Create requirements.txt with core dependencies (fastapi, torch, rdkit, transformers, lightgbm, etc.)
  - Create .env.example documenting all environment variables
  - Set up Python virtual environment and install dependencies
  - Create backend/app/ structure with __init__.py files
  - _Requirements: 18.1-18.12, 16.12_

- [ ] 2. Implement SMILES preprocessing utilities
  - [x] 2.1 Create backend/app/preprocessing/rdkit_utils.py
    - Implement validate_smiles() function with RDKit parser
    - Implement standardize_smiles() with charge neutralization, salt removal, tautomer canonicalization
    - Implement smiles_to_mol() with error handling
    - Implement generate_2d_image() for PNG generation
    - _Requirements: 1.1, 1.2, 1.3, 1.4_
  
  - [ ]* 2.2 Write property test for SMILES standardization idempotence
    - **Property: Standardizing then canonicalizing then standardizing produces equivalent molecule**
    - **Validates: Requirements 1.9**
  
  - [ ]* 2.3 Write unit tests for SMILES validation
    - Test valid SMILES acceptance
    - Test invalid SMILES rejection with descriptive errors
    - _Requirements: 1.2, 17.1, 22.1_

- [ ] 3. Implement molecular feature computation
  - [x] 3.1 Create backend/app/preprocessing/descriptors.py
    - Implement compute_descriptors() for 200+ RDKit descriptors
    - Include MW, logP, TPSA, MolMR, BertzCT, Chi0-Chi4, Kappa1-Kappa3
    - Include NumHDonors, NumHAcceptors, NumRotatableBonds, NumAromaticRings
    - Return 200-dimensional numpy array
    - _Requirements: 1.5, 5.1-5.3_
  
  - [x] 3.2 Create backend/app/preprocessing/fingerprints.py
    - Implement compute_morgan_fingerprint() with radius=2, n_bits=2048
    - Implement compute_maccs_keys() returning 167-bit vector
    - _Requirements: 1.6, 1.7_
  
  - [x] 3.3 Create backend/app/preprocessing/graph_builder.py
    - Implement mol_to_graph() converting RDKit mol to PyTorch Geometric Data
    - Node features: atomic_num (one-hot), degree, hybridization, aromaticity, in_ring, formal_charge, num_Hs
    - Edge features: bond_type, is_conjugated, is_in_ring, stereo
    - _Requirements: 2.2_

  - [ ]* 3.4 Write property test for descriptor computation determinism
    - **Property: Computing descriptors twice for same molecule produces identical results**
    - **Validates: Requirements 22.10**
  
  - [ ]* 3.5 Write unit tests for feature computation
    - Test descriptor computation
    - Test fingerprint computation
    - Test graph construction
    - _Requirements: 22.2, 22.3, 22.4_

- [ ] 4. Create preprocessing pipeline integration
  - [x] 4.1 Create backend/app/preprocessing/pipeline.py
    - Implement PreprocessingPipeline class integrating all components
    - process() method returns dict with mol, canonical_smiles, descriptors, fingerprints, graph, image_png
    - Handle errors gracefully with descriptive messages
    - Measure and log processing time
    - _Requirements: 1.8, 17.1, 17.2, 20.1_
  
  - [ ]* 4.2 Write property test for preprocessing idempotence
    - **Property: Preprocessing then re-preprocessing produces identical feature vectors**
    - **Validates: Requirements 21.12**

- [ ] 5. Implement data preprocessing script for Tox21 dataset
  - [x] 5.1 Create ml/scripts/preprocess_tox21.py
    - Download Tox21 dataset from Kaggle or load from local file
    - Validate and standardize all SMILES strings
    - Compute descriptors, fingerprints, and graphs for all molecules
    - Implement scaffold_split() using Bemis-Murcko scaffolds (80/10/10 ratio)
    - Compute per-assay class weights for imbalanced labels
    - Compute label correlation matrix for GNN joint loss
    - Save processed features to ml/data/processed/ as pickle files
    - Save split indices for reproducibility
    - Log preprocessing statistics
    - _Requirements: 21.1-21.11, 12.1, 12.3_
  
  - [x] 5.2 Run preprocessing script
    - Execute preprocess_tox21.py to generate all training data
    - Verify output files created successfully
    - Verify processing completes within 10 minutes
    - _Requirements: 21.11_

- [x] 6. Checkpoint - Ensure preprocessing pipeline functional
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement LightGBM descriptor model
  - [x] 7.1 Create ml/scripts/train_lgbm.py
    - Load processed descriptors and fingerprints from ml/data/processed/
    - Concatenate descriptors + Morgan + MACCS into feature vector
    - Train 12 separate LGBMClassifier instances (one per assay)
    - Apply per-assay class weights to handle imbalance
    - Implement masked loss to handle missing labels
    - Use early stopping based on validation AUROC
    - Save trained models and StandardScaler to ml/artifacts/
    - _Requirements: 2.3, 12.3, 12.8, 12.9_

  - [x] 7.2 Train and evaluate LightGBM model
    - Execute train_lgbm.py to train all 12 classifiers
    - Compute per-assay AUROC on test set
    - Verify mean AUROC ≥ 0.80 target
    - Log training metrics and save results
    - _Requirements: 12.9, 12.11, 12.12_
  
  - [ ]* 7.3 Write property test for missing label masking
    - **Property: Predictions ignore missing labels in loss computation**
    - **Validates: Requirements 12.8**

- [ ] 8. Implement GNN model architecture
  - [x] 8.1 Create ml/models/gnn.py
    - Implement ToxGNN class with AttentiveFP architecture
    - Use 4 graph convolution layers with hidden_dim=256
    - Implement global mean and max pooling
    - Concatenate pooled embeddings (512-dim)
    - Add dropout (p=0.3) and 12 sigmoid output heads
    - _Requirements: 26.1-26.6_
  
  - [x] 8.2 Implement joint correlation loss
    - Create compute_correlation_matrix() function
    - Implement joint_correlation_loss() combining masked BCE and correlation consistency
    - Use lambda=0.1 weight for correlation loss
    - _Requirements: 26.7-26.9_
  
  - [x] 8.3 Create ml/scripts/train_gnn.py
    - Load processed graph data from ml/data/processed/
    - Initialize ToxGNN model
    - Use AdamW optimizer (lr=1e-3, weight_decay=1e-4)
    - Use CosineAnnealingLR scheduler (T_max=50)
    - Implement early stopping (patience=15) based on validation AUROC
    - Train for up to 100 epochs
    - Save best model checkpoint to ml/artifacts/
    - _Requirements: 26.10-26.13_
  
  - [~] 8.4 Train and evaluate GNN model
    - Execute train_gnn.py to train model
    - Compute per-assay AUROC on test set
    - Verify mean AUROC ≥ 0.80 target
    - _Requirements: 26.14_
  
  - [ ]* 8.5 Write property test for GNN loss composition
    - **Property: Joint loss equals BCE loss + lambda * correlation loss**
    - **Validates: Requirements 26.9**

- [ ] 9. Implement ChemBERTa-2 fine-tuning
  - [x] 9.1 Create ml/scripts/train_chemberta.py
    - Load pretrained "seyonec/ChemBERTa-zinc-base-v1" model
    - Add 12-class multi-label classification head
    - Implement masked loss for missing labels (set to -1)
    - Use AdamW optimizer (lr=2e-5) with linear warmup (10%) and cosine decay
    - Train for 5-8 epochs with batch_size=32
    - Use mixed precision training (fp16)
    - Implement early stopping based on validation mean AUROC
    - Save best checkpoint to ml/artifacts/chemberta_finetuned/
    - _Requirements: 27.1-27.12_

  - [~] 9.2 Train and evaluate ChemBERTa model
    - Execute train_chemberta.py to fine-tune model
    - Compute per-assay AUROC on test set
    - Verify mean AUROC ≥ 0.78 target
    - _Requirements: 27.13_

- [ ] 10. Implement ensemble fusion and weight optimization
  - [x] 10.1 Create ml/models/ensemble.py
    - Implement EnsembleModel class loading all three models
    - Implement weighted logit fusion (convert probs to logits, fuse, convert back)
    - Load ensemble weights from ensemble_weights.json
    - Return probabilities, logits, and individual model predictions
    - _Requirements: 2.4, 2.5, 2.6, 28.1, 28.2, 28.9_
  
  - [x] 10.2 Create ml/scripts/optimize_ensemble.py
    - Define objective function computing validation AUROC
    - Use Nelder-Mead optimization starting from [1, 1, 1]
    - Apply softmax normalization to ensure weights sum to 1
    - Save optimized weights to ml/artifacts/ensemble_weights.json
    - _Requirements: 28.3-28.7_
  
  - [~] 10.3 Optimize and evaluate ensemble
    - Execute optimize_ensemble.py to find optimal weights
    - Compute per-assay AUROC on test set
    - Verify mean AUROC ≥ 0.80 and improvement ≥ 0.02 over best individual model
    - _Requirements: 2.12, 28.10, 28.11_
  
  - [ ]* 10.4 Write property test for ensemble weight normalization
    - **Property: Ensemble weights sum to 1.0 after softmax**
    - **Validates: Requirements 28.6**
  
  - [ ]* 10.5 Write property test for probability bounds
    - **Property: All ensemble probabilities are in [0, 1]**
    - **Validates: Requirements 2.6**

- [ ] 11. Implement conformal prediction wrapper
  - [x] 11.1 Create ml/models/conformal.py
    - Wrap EnsembleModel with MAPIE MapieClassifier
    - Calibrate on held-out calibration set with alpha=0.15
    - Generate prediction sets: {SAFE}, {TOXIC}, or {SAFE, TOXIC}
    - _Requirements: 2.11, 29.1-29.6_
  
  - [~] 11.2 Evaluate conformal prediction
    - Compute empirical coverage on test set
    - Verify coverage ≥ 80% target
    - Measure additional latency (<50ms target)
    - _Requirements: 29.9, 29.10_

- [ ] 12. Implement composite risk score and classification
  - [~] 12.1 Create backend/app/models/risk_scorer.py
    - Implement compute_composite_risk() as weighted average of 12 assay probabilities
    - Implement classify_risk_level() returning HIGH (>0.6), MEDIUM (0.35-0.6), or LOW (<0.35)
    - _Requirements: 2.7-2.10_
  
  - [ ]* 12.2 Write property test for risk level classification consistency
    - **Property: Risk level classification boundaries are consistent**
    - **Validates: Requirements 2.8-2.10**

- [ ] 13. Checkpoint - Ensure all ML models trained and evaluated
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Implement backend model inference module
  - [~] 14.1 Create backend/app/models/descriptor_model.py
    - Implement DescriptorModel class loading LightGBM models and scaler
    - Implement predict() method returning 12 probabilities
    - _Requirements: 2.3_
  
  - [~] 14.2 Create backend/app/models/gnn_model.py
    - Copy ToxGNN architecture from ml/models/gnn.py
    - Implement model loading from checkpoint
    - Implement predict() method with GPU support
    - _Requirements: 2.2_
  
  - [~] 14.3 Create backend/app/models/transformer_model.py
    - Implement ChemBERTaModel class loading fine-tuned model and tokenizer
    - Implement predict() method with GPU support and proper tokenization
    - _Requirements: 2.1_
  
  - [~] 14.4 Create backend/app/models/ensemble_model.py
    - Copy EnsembleModel from ml/models/ensemble.py
    - Integrate all three models with weighted logit fusion
    - Load ensemble weights from artifacts
    - _Requirements: 2.4, 2.5_

- [ ] 15. Implement SHAP explainability
  - [~] 15.1 Create backend/app/explainability/shap_utils.py
    - Implement ShapExplainer class with TreeExplainer
    - Load SHAP background dataset from artifacts
    - Implement explain() method computing SHAP values
    - Return top 10 features by absolute SHAP value with names, values, contributions, directions
    - _Requirements: 3.5, 3.6_
  
  - [ ]* 15.2 Write unit tests for SHAP computation
    - Test SHAP value computation
    - Test top 10 feature extraction
    - _Requirements: 22.5_

- [ ] 16. Implement Captum atom attribution
  - [~] 16.1 Create backend/app/explainability/captum_utils.py
    - Implement CaptumExplainer class with IntegratedGradients
    - Implement explain() method computing per-atom attributions
    - Normalize scores to [0, 1] range
    - Complete computation within 800ms target
    - _Requirements: 3.1, 3.2, 3.8_
  
  - [~] 16.2 Create backend/app/explainability/heatmap_renderer.py
    - Implement HeatmapRenderer class
    - Implement render() method generating atom-colored PNG with RdYlBu_r colormap
    - Return base64-encoded image
    - Complete rendering within 100ms target
    - _Requirements: 3.3, 3.4, 20.6_
  
  - [ ]* 16.3 Write unit tests for Captum attribution
    - Test atom attribution computation
    - Test heatmap rendering
    - _Requirements: 22.5_

- [ ] 17. Implement structural alert detection
  - [~] 17.1 Create backend/app/explainability/structural_alerts.py
    - Implement StructuralAlertScanner class
    - Load 150+ SMARTS toxicophore patterns
    - Include quinones, nitro aromatics, epoxides, Michael acceptors, aldehydes, anilines, etc.
    - Implement scan() method detecting matching patterns
    - Return alerts with name, SMARTS, severity, description, atom_indices
    - Complete scanning within 50ms target
    - _Requirements: 4.1-4.8, 20.7_

  - [ ]* 17.2 Write unit tests for structural alert detection
    - Test alert pattern matching
    - Test severity classification
    - _Requirements: 22.5_

- [ ] 18. Implement ADMET property prediction
  - [~] 18.1 Create backend/app/features/admet_predictor.py
    - Implement ADMETPredictor class
    - Compute QED, Lipinski violations, TPSA, logP, MW
    - Compute BBB penetration and oral bioavailability estimates
    - Integrate ADMET-AI for CYP2D6, CYP3A4, hERG predictions (if available)
    - Compute water solubility (logS)
    - Return structured ADMET properties dict
    - Complete computation within 100ms target
    - _Requirements: 5.1-5.10, 20.8_

- [ ] 19. Implement UMAP chemical space search
  - [~] 19.1 Create ml/scripts/precompute_umap.py
    - Compute Morgan fingerprints for all Tox21 molecules
    - Fit UMAP with n_components=2, n_neighbors=15, min_dist=0.1, metric='jaccard'
    - Save fitted reducer to ml/artifacts/umap_reducer.pkl
    - Save coordinates, SMILES, fingerprints, labels to ml/artifacts/umap_data.json
    - Complete fitting within 10 minutes
    - _Requirements: 30.1-30.9_
  
  - [~] 19.2 Run UMAP precomputation
    - Execute precompute_umap.py to generate embeddings
    - Verify output files created successfully
  
  - [~] 19.3 Create backend/app/features/umap_search.py
    - Implement UMAPSearchEngine class
    - Load precomputed embeddings and fitted reducer
    - Implement project() method for new molecules (complete within 20ms)
    - Implement find_similar() computing Tanimoto similarity
    - Return top-k nearest neighbors with coordinates and labels
    - Complete similarity search within 100ms
    - _Requirements: 9.1-9.11, 30.7, 30.8, 30.10, 31.1-31.7_

- [ ] 20. Implement bioisostere generation for de-risking
  - [~] 20.1 Create backend/app/features/derisking.py
    - Implement BioisostereGenerator class
    - Define substitution rules: NO2→CN, Cl→F, aldehyde→alcohol, quinone→phenol, etc.
    - Implement generate_variants() applying rules to detected alerts
    - Validate generated SMILES with RDKit
    - Return 3-8 valid variants with modification descriptions
    - _Requirements: 10.1-10.11_
  
  - [ ]* 20.2 Write property test for bioisostere validity
    - **Property: All generated bioisostere SMILES are valid**
    - **Validates: Requirements 10.4, 10.5**

- [ ] 21. Implement LLM report generation
  - [~] 21.1 Create backend/app/report/llm_reporter.py
    - Implement LLMReporter class with Claude/Groq/Mistral API clients
    - Implement generate_report() formatting context and calling LLM
    - Structure report: Executive Summary, Pathway Analysis, Structural Drivers, De-Risking Recommendations, Regulatory Outlook, Confidence Assessment
    - Limit output to 1500 tokens
    - Handle API errors gracefully
    - Complete generation within 20 seconds
    - _Requirements: 6.1-6.8_

- [ ] 22. Implement PDF export
  - [~] 22.1 Create backend/app/report/pdf_exporter.py
    - Implement PDFExporter class using WeasyPrint
    - Implement export() method rendering HTML template
    - Include molecular structure, predictions table, SHAP chart, alerts, report text
    - Use professional styling with ToxiLens branding
    - Generate 2-page PDF with footer including timestamp
    - Complete generation within 5 seconds
    - _Requirements: 7.1-7.10_

- [ ] 23. Implement FastAPI backend application
  - [~] 23.1 Create backend/app/core/config.py
    - Implement Settings class using pydantic-settings
    - Load configuration from environment variables
    - Support MODEL_ARTIFACTS_PATH, API keys, CORS_ORIGINS, LOG_LEVEL, MAX_BATCH_SIZE, DEVICE
    - Provide default values and validation
    - _Requirements: 18.1-18.12_
  
  - [~] 23.2 Create backend/app/core/logging.py
    - Configure structured logging with JSON format
    - Include request IDs for tracing
    - Log requests, processing times, errors with stack traces
    - Respect LOG_LEVEL configuration
    - Do not log sensitive information (API keys, SMILES strings)
    - _Requirements: 19.1-19.12_
  
  - [~] 23.3 Create backend/app/main.py
    - Initialize FastAPI app with CORS middleware
    - Implement startup event handler for model preloading
    - Load all models, UMAP embeddings, SHAP background set
    - Log successful initialization
    - Fail fast if artifacts missing
    - Complete preloading within 30 seconds on CPU
    - _Requirements: 15.1-15.12_

- [ ] 24. Create Pydantic request/response schemas
  - [~] 24.1 Create backend/app/schemas/prediction.py
    - Define PredictionRequest with SMILES validation
    - Define PredictionResponse with all fields
    - Define BatchPredictionRequest and BatchPredictionResponse
    - Define ReportRequest and ReportResponse
    - Define DeriskRequest and DeriskResponse
    - Define WhatIfRequest and WhatIfResponse
    - Add field validation with descriptive error messages
    - _Requirements: 13.9, 13.10, 13.11_
  
  - [ ]* 24.2 Write property test for request validation error format
    - **Property: Validation errors return 422 status with detailed messages**
    - **Validates: Requirements 13.10**

- [ ] 25. Implement /predict endpoint
  - [~] 25.1 Create backend/app/api/predict.py
    - Implement POST /predict endpoint
    - Accept SMILES with optional flags (include_heatmap, include_shap, include_alerts, include_admet)
    - Run preprocessing pipeline
    - Run ensemble prediction with conformal intervals
    - Compute SHAP, Captum, alerts, ADMET if requested
    - Compute composite risk score and risk level
    - Return comprehensive prediction response
    - Complete within 200ms on CPU
    - _Requirements: 13.1, 13.11, 2.1-2.12, 3.1-3.8, 4.1-4.8, 5.1-5.10, 20.1_

  - [ ]* 25.2 Write integration test for /predict endpoint
    - Test with valid SMILES
    - Test with invalid SMILES
    - Test error handling
    - _Requirements: 22.6_

- [ ] 26. Implement /predict_batch endpoint
  - [~] 26.1 Create backend/app/api/batch.py
    - Implement POST /predict_batch endpoint
    - Accept CSV file upload with SMILES column
    - Parse and validate CSV
    - Process each molecule sequentially
    - Compute composite risk scores
    - Rank by risk score descending
    - Return batch results with flagged assays
    - Complete 100 molecules within 12 seconds on CPU
    - _Requirements: 13.2, 8.1-8.10, 20.3_
  
  - [ ]* 26.2 Write integration test for /predict_batch endpoint
    - Test with valid CSV
    - Test with missing SMILES column
    - Test batch size limits
    - _Requirements: 22.7_

- [ ] 27. Implement /generate_report endpoint
  - [~] 27.1 Create backend/app/api/report.py
    - Implement POST /generate_report endpoint
    - Accept SMILES and prediction data
    - Call LLMReporter to generate report text
    - Optionally generate PDF with PDFExporter
    - Return report text and PDF bytes
    - _Requirements: 13.3, 6.1-6.8, 7.1-7.10_
  
  - [ ]* 27.2 Write integration test for /generate_report endpoint
    - Test report generation
    - Test PDF export
    - Test LLM API error handling
    - _Requirements: 22.8_

- [ ] 28. Implement /derisk endpoint
  - [~] 28.1 Create backend/app/api/derisk.py
    - Implement POST /derisk endpoint
    - Accept SMILES string
    - Scan for structural alerts
    - Generate bioisostere variants
    - Predict toxicity for all variants
    - Compute delta risk scores
    - Rank variants by improvement
    - Complete within 3 seconds on GPU
    - _Requirements: 13.5, 10.1-10.12_

- [ ] 29. Implement /what_if endpoint
  - [~] 29.1 Create backend/app/api/whatif.py
    - Implement POST /what_if endpoint
    - Accept original and modified SMILES
    - Predict toxicity for both molecules
    - Compute per-assay deltas
    - Identify improved and worsened assays
    - Generate side-by-side heatmaps
    - Complete within 400ms on CPU
    - _Requirements: 13.4, 32.1-32.10, 20.9_

- [ ] 30. Implement /similar endpoint
  - [~] 30.1 Create backend/app/api/similar.py
    - Implement GET /similar endpoint
    - Accept SMILES query parameter
    - Project molecule into UMAP space
    - Compute Tanimoto similarity to all Tox21 molecules
    - Return top-k nearest neighbors
    - _Requirements: 13.6, 9.1-9.11, 31.1-31.8_

- [ ] 31. Implement /health and /docs endpoints
  - [~] 31.1 Create backend/app/api/health.py
    - Implement GET /health endpoint
    - Return status, models_loaded flag, version, uptime
    - Only return "ready" after all models loaded
    - _Requirements: 13.7, 15.11, 15.12_
  
  - [~] 31.2 Configure Swagger UI documentation
    - Enable auto-generated docs at /docs endpoint
    - _Requirements: 13.8_

- [ ] 32. Implement error handling and validation
  - [~] 32.1 Create backend/app/core/errors.py
    - Implement custom exception classes
    - Implement global exception handlers
    - Return appropriate HTTP status codes (422, 500, 503)
    - Return JSON with "error" and "detail" fields
    - Do not expose stack traces in production
    - _Requirements: 17.1-17.12, 13.14_
  
  - [~] 32.2 Implement rate limiting
    - Add slowapi rate limiter middleware
    - Limit to 100 requests/minute per IP
    - Return 429 status for rate limit exceeded
    - _Requirements: 24.8_

- [ ] 33. Checkpoint - Ensure backend API functional
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 34. Set up React TypeScript frontend project
  - [~] 34.1 Initialize Vite React TypeScript project
    - Create frontend/ directory with Vite template
    - Install dependencies: react, react-router-dom, axios, recharts, plotly.js, tailwindcss
    - Configure Tailwind CSS
    - Set up TypeScript configuration
    - Create src/ structure with components/, pages/, api/, types/
    - _Requirements: 14.14, 14.15_
  
  - [~] 34.2 Create API client module
    - Create frontend/src/api/client.ts
    - Implement predictToxicity() function
    - Implement predictBatch() function
    - Implement generateReport() function
    - Implement deriskMolecule() function
    - Implement whatIfAnalysis() function
    - Implement findSimilar() function
    - Configure base URL from environment variable
    - _Requirements: 13.1-13.7_

- [ ] 35. Implement frontend visualization components
  - [~] 35.1 Create frontend/src/components/MoleculeViewer.tsx
    - Display base molecular structure image
    - Overlay heatmap with opacity control
    - Position alert badges on structure
    - Provide zoom and pan controls
    - _Requirements: 14.6, 33.1, 33.8_
  
  - [~] 35.2 Create frontend/src/components/ToxicityRadar.tsx
    - Use Recharts RadarChart with 12 axes
    - Display assay probabilities
    - Color code by risk level (red/amber/green)
    - Show tooltips with exact values on hover
    - _Requirements: 14.7, 33.2, 33.9, 33.10_

  - [~] 35.3 Create frontend/src/components/ShapChart.tsx
    - Use Recharts BarChart for top 10 features
    - Color bars by direction (red=toxic, blue=protective)
    - Display feature names and SHAP values
    - Show tooltips on hover
    - _Requirements: 14.8, 33.3, 33.9_
  
  - [~] 35.4 Create frontend/src/components/AlertBadges.tsx
    - Display structural alerts with severity color coding
    - Position badges near flagged atoms
    - Show alert name and description on hover
    - _Requirements: 14.9, 33.4_
  
  - [~] 35.5 Create frontend/src/components/AdmetPanel.tsx
    - Display QED, Lipinski violations, TPSA, logP, MW
    - Display BBB penetration and oral bioavailability
    - Display CYP and hERG predictions
    - Use consistent color coding
    - _Requirements: 14.10, 33.5_
  
  - [~] 35.6 Create frontend/src/components/UmapPlot.tsx
    - Use Plotly scattergl for 12,000 points with WebGL acceleration
    - Color points by toxicity labels
    - Highlight query molecule with star marker
    - Enable hover to show SMILES
    - Enable click to load molecule
    - Maintain 60 FPS performance
    - _Requirements: 14.11, 33.7, 9.6-9.11, 20.9, 20.10, 33.11, 33.12_

- [ ] 36. Implement frontend pages
  - [~] 36.1 Create frontend/src/pages/SingleAnalysis.tsx
    - Add SMILES text input with validation
    - Add preset example molecule buttons (aspirin, ibuprofen, caffeine, bisphenol A, doxorubicin)
    - Display MoleculeViewer, ToxicityRadar, ShapChart, AlertBadges, AdmetPanel
    - Add "Generate Report" button
    - Add "Download PDF" button
    - Show loading animation during API calls
    - Display error messages in notification banner
    - _Requirements: 14.1, 14.11-14.13, 34.1-34.10_
  
  - [~] 36.2 Create frontend/src/pages/ChemicalSpace.tsx
    - Add SMILES input for query molecule
    - Display UmapPlot with 12,000 Tox21 points
    - Highlight query molecule and nearest neighbors
    - Enable click to load molecule into SingleAnalysis
    - _Requirements: 14.2, 9.6-9.11_
  
  - [~] 36.3 Create frontend/src/pages/BatchScreening.tsx
    - Add CSV file upload component
    - Display results table with sorting and filtering
    - Show SMILES, composite risk, risk level, flagged assays
    - Enable export of results
    - _Requirements: 14.3, 8.1-8.10_
  
  - [~] 36.4 Create frontend/src/pages/DeRiskingLab.tsx
    - Add SMILES input
    - Display original molecule and risk score
    - Display generated variants in grid
    - Show delta risk for each variant
    - Enable click to compare variants
    - _Requirements: 14.4, 10.1-10.12_

  - [~] 36.5 Create frontend/src/pages/MultiCompare.tsx
    - Add multi-SMILES input (2-5 molecules)
    - Display heatmap grid with assays as rows, molecules as columns
    - Color cells by toxicity probability
    - Display delta values relative to baseline
    - Show tooltips with exact values on hover
    - _Requirements: 14.5, 11.1-11.9_

- [ ] 37. Implement frontend styling and UX
  - [~] 37.1 Apply Tailwind CSS styling
    - Use dark theme with blue accents
    - Apply consistent spacing and card layouts
    - Add color-coded risk indicators (red/amber/green)
    - Add smooth transitions and hover effects
    - _Requirements: 14.15_
  
  - [~] 37.2 Implement loading states and error handling
    - Display loading animations during API calls
    - Show error messages in notification banners
    - Clear errors on new requests
    - _Requirements: 14.12, 17.10, 17.11_
  
  - [~] 37.3 Implement accessibility features
    - Use semantic HTML elements
    - Add alt text for images
    - Ensure keyboard accessibility
    - Add focus indicators
    - Add ARIA labels
    - Maintain 4.5:1 color contrast
    - Support browser zoom up to 200%
    - _Requirements: 35.1-35.12_

- [ ] 38. Checkpoint - Ensure frontend functional
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 39. Create Docker deployment configuration
  - [~] 39.1 Create docker/backend.Dockerfile
    - Use continuumio/miniconda3 base image
    - Install RDKit via conda
    - Install Python dependencies from requirements.txt
    - Copy backend code and model artifacts
    - Expose port 8000
    - Set CMD to run uvicorn
    - _Requirements: 16.2, 16.8, 16.9_
  
  - [~] 39.2 Create docker/frontend.Dockerfile
    - Use node:18 for build stage
    - Build React app with npm run build
    - Use nginx for serving static files
    - Copy nginx configuration
    - Expose port 3000
    - _Requirements: 16.3, 16.10_
  
  - [~] 39.3 Create docker-compose.yml
    - Define backend and frontend services
    - Configure ports (8000, 3000), environment variables, volumes
    - Add health checks
    - Configure restart policies
    - _Requirements: 16.1, 16.4-16.7, 16.11_
  
  - [~] 39.4 Test Docker deployment
    - Run docker-compose up
    - Verify platform accessible at http://localhost:3000 within 60 seconds
    - Test end-to-end prediction flow
    - _Requirements: 16.11_

- [ ] 40. Create Hugging Face Spaces deployment configuration
  - [~] 40.1 Create Spaces README.md with YAML frontmatter
    - Configure T4 GPU, Python 3.11 runtime
    - Add project description, demo GIF, usage instructions
    - _Requirements: 25.1-25.3, 25.12_
  
  - [~] 40.2 Create app.py entry point for Spaces
    - Adapt backend/app/main.py for Spaces deployment
    - Configure model artifact loading or download at startup
    - _Requirements: 25.5, 25.6_
  
  - [~] 40.3 Create Spaces-compatible requirements.txt
    - Ensure compatibility with Spaces environment
    - _Requirements: 25.4_
  
  - [~] 40.4 Configure Spaces secrets
    - Document ANTHROPIC_API_KEY configuration
    - _Requirements: 25.7_
  
  - [~] 40.5 Write deployment documentation
    - Provide step-by-step Spaces setup instructions
    - _Requirements: 25.10_

- [ ] 41. Create comprehensive documentation
  - [~] 41.1 Write README.md
    - Add project overview and features
    - Add installation instructions
    - Add usage examples with curl commands
    - Add architecture diagrams
    - Add troubleshooting guidance
    - Add contribution guidelines
    - Add license information (MIT)
    - Add citations for research papers and methods
    - _Requirements: 23.1, 23.5, 23.8, 23.9, 23.10, 23.11, 23.12_
  
  - [~] 41.2 Create model card
    - Document training data, performance metrics, limitations
    - _Requirements: 23.7_
  
  - [~] 41.3 Create example files
    - Add example SMILES strings for common drugs
    - Add example CSV file for batch screening
    - Add Jupyter notebook with API usage examples
    - _Requirements: 23.3, 23.4, 23.8_

- [ ] 42. Implement security measures
  - [~] 42.1 Verify security requirements
    - Confirm no SMILES stored in logs or disk
    - Verify rate limiting functional
    - Verify CORS restrictions
    - Verify input validation and sanitization
    - Verify API keys in environment variables only
    - Verify file upload size limits (10 MB)
    - _Requirements: 24.1-24.12_

- [ ] 43. Run end-to-end integration tests
  - [~] 43.1 Test complete prediction flow
    - Test SMILES input → preprocessing → prediction → explainability → report
    - Verify all visualizations render correctly
    - _Requirements: 20.1-20.12_
  
  - [~] 43.2 Test batch screening
    - Test CSV upload with 100 molecules
    - Verify processing time <12 seconds on CPU
    - _Requirements: 20.3_
  
  - [~] 43.3 Test de-risking workflow
    - Test variant generation and toxicity comparison
    - _Requirements: 10.1-10.12_

  - [ ] 43.4 Test chemical space exploration
    - Test UMAP visualization with 12,000 points
    - Verify 60 FPS rendering performance
    - _Requirements: 20.9, 20.10_
  
  - [ ]* 43.5 Run full test suite
    - Execute all unit tests, integration tests, property tests
    - Verify at least 70% code coverage
    - Ensure test suite completes within 5 minutes
    - _Requirements: 22.1-22.14_

- [ ] 44. Run performance benchmarks
  - [ ] 44.1 Measure single prediction latency
    - Verify <200ms on CPU
    - Verify <30ms on GPU (if available)
    - _Requirements: 20.1, 20.2_
  
  - [ ] 44.2 Measure batch prediction latency
    - Verify 100 molecules <12s on CPU
    - Verify 100 molecules <2s on GPU (if available)
    - _Requirements: 20.3, 20.4_
  
  - [ ] 44.3 Measure explainability computation time
    - Verify SHAP + Captum <800ms on CPU
    - Verify heatmap rendering <100ms
    - Verify structural alert scanning <50ms
    - Verify ADMET computation <100ms
    - _Requirements: 20.5, 20.6, 20.7, 20.8_
  
  - [ ] 44.4 Measure frontend performance
    - Verify UMAP plot renders at 60 FPS
    - Verify user interactions <16ms response time
    - _Requirements: 20.9, 20.10_

- [ ] 45. Final checkpoint - Ensure all tests pass and platform ready for deployment
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based and unit tests that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at major milestones (after preprocessing, after ML training, after backend API, after frontend, before deployment)
- Property tests validate universal correctness properties across randomized inputs
- Unit tests validate specific examples and edge cases
- Implementation follows incremental approach: setup → preprocessing → ML training → backend core → explainability → advanced features → frontend → deployment
- Model training tasks (7-11) can be run in parallel after data preprocessing completes
- Backend tasks (14-32) should be completed before frontend tasks (34-38)
- All model artifacts must be saved to ml/artifacts/ for backend to load at startup
- Frontend uses TypeScript for type safety and Tailwind CSS for styling
- Backend uses Python with FastAPI, PyTorch, RDKit, and related ML libraries
- Docker and Hugging Face Spaces deployment configurations enable easy deployment
