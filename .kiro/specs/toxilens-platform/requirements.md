# Requirements Document

## Introduction

ToxiLens is an interpretable multi-modal AI platform for drug toxicity prediction. The system enables medicinal chemists and drug discovery researchers to predict toxicity across 12 Tox21 assays, understand molecular-level drivers of toxicity through explainable AI, and generate actionable de-risking recommendations. The platform combines three complementary machine learning models (ChemBERTa-2 transformer, multi-task graph neural network, and LightGBM descriptor model) with multiple explainability techniques to provide atom-level heatmaps, descriptor importance rankings, structural alert detection, and LLM-generated assessment reports.

## Glossary

- **Platform**: The complete ToxiLens system including backend API, frontend web application, and ML models
- **Backend**: The FastAPI server that processes predictions, manages ML models, and generates reports
- **Frontend**: The React 18 web application providing user interface
- **ML_Ensemble**: The weighted fusion of ChemBERTa-2, GNN, and LightGBM models
- **ChemBERTa_Model**: The SMILES transformer model fine-tuned on Tox21 data
- **GNN_Model**: The graph neural network with AttentiveFP architecture and joint correlation loss
- **LightGBM_Model**: The gradient boosting model trained on molecular descriptors
- **Preprocessor**: The RDKit-based module that standardizes SMILES and computes molecular features
- **XAI_Engine**: The explainability module providing SHAP, Captum, and structural alerts
- **Report_Generator**: The LLM-powered module that creates toxicity assessment reports
- **PDF_Exporter**: The WeasyPrint-based module that renders reports as downloadable PDFs
- **SMILES**: Simplified Molecular Input Line Entry System string representation of molecules
- **Tox21_Assay**: One of 12 toxicity endpoints (NR-AR, NR-AhR, NR-AR-LBD, SR-ARE, SR-p53, NR-ER, SR-MMP, NR-AROMATASE, SR-ATAD5, SR-HSE, NR-ER-LBD, NR-PPAR)
- **AUROC**: Area Under Receiver Operating Characteristic curve, primary evaluation metric
- **Scaffold_Split**: Bemis-Murcko scaffold-based train/validation/test splitting strategy
- **Conformal_Prediction**: MAPIE-based uncertainty quantification providing calibrated prediction intervals
- **Structural_Alert**: SMARTS pattern matching known toxicophores (150+ patterns)
- **SHAP_Value**: Shapley Additive Explanation value indicating feature contribution to prediction
- **Atom_Attribution**: Per-atom importance score from Captum IntegratedGradients on GNN
- **ADMET_Property**: Absorption, Distribution, Metabolism, Excretion, Toxicity property
- **Bioisostere**: Structurally similar molecular fragment with potentially different toxicity profile
- **UMAP_Embedding**: 2D projection of chemical space using Uniform Manifold Approximation and Projection
- **Heatmap**: 2D molecular structure visualization with atoms colored by importance scores
- **Composite_Risk_Score**: Weighted average of 12 Tox21 assay probabilities
- **Risk_Level**: Categorical classification (HIGH/MEDIUM/LOW) based on composite risk score
- **Batch_Screening**: Processing multiple molecules from CSV upload with ranked results
- **De_Risking_Lab**: Feature for generating and evaluating bioisostere variants
- **Chemical_Space_Explorer**: Interactive UMAP visualization of Tox21 dataset with query molecule
- **User**: Medicinal chemist, drug discovery researcher, or computational chemist using the platform


## Requirements

### Requirement 1: SMILES Input Processing

**User Story:** As a User, I want to input molecular structures via SMILES strings, so that I can obtain toxicity predictions for compounds of interest.

#### Acceptance Criteria

1. WHEN a User submits a SMILES string, THE Preprocessor SHALL validate the SMILES syntax using RDKit
2. IF a SMILES string is invalid, THEN THE Preprocessor SHALL return a descriptive error message indicating the parsing failure
3. WHEN a valid SMILES string is received, THE Preprocessor SHALL standardize the molecule by neutralizing charges, removing salts, and canonicalizing tautomers
4. THE Preprocessor SHALL generate a 2D molecular structure image in PNG format with dimensions of 400x400 pixels
5. WHEN standardization is complete, THE Preprocessor SHALL compute 200+ RDKit molecular descriptors including logP, TPSA, molecular weight, hydrogen bond donors, hydrogen bond acceptors, and rotatable bonds
6. THE Preprocessor SHALL compute Morgan fingerprints with radius 2 and 2048 bits (ECFP4)
7. THE Preprocessor SHALL compute MACCS keys with 167 bits
8. WHEN a User provides a SMILES string, THE Platform SHALL process the request and return results within 200 milliseconds on CPU
9. FOR ALL valid SMILES strings, standardizing then canonicalizing then standardizing SHALL produce an equivalent molecule (idempotence property)


### Requirement 2: Multi-Modal Toxicity Prediction

**User Story:** As a User, I want to receive toxicity predictions across all 12 Tox21 assays simultaneously, so that I can assess comprehensive toxicity risk profiles.

#### Acceptance Criteria

1. WHEN a standardized molecule is provided, THE ChemBERTa_Model SHALL tokenize the SMILES string and generate a 768-dimensional CLS embedding
2. WHEN a standardized molecule is provided, THE GNN_Model SHALL construct a molecular graph with atom features (atomic number, degree, hybridization, aromaticity, ring membership, formal charge) and bond features (bond type, conjugation, ring membership, stereochemistry)
3. WHEN a standardized molecule is provided, THE LightGBM_Model SHALL predict toxicity probabilities using the computed descriptors and fingerprints
4. THE ML_Ensemble SHALL compute weighted logit-level fusion of predictions from ChemBERTa_Model, GNN_Model, and LightGBM_Model
5. THE ML_Ensemble SHALL apply learned ensemble weights (optimized on validation set via Nelder-Mead) to combine model outputs
6. THE ML_Ensemble SHALL generate probability predictions for all 12 Tox21_Assays in a single forward pass
7. THE ML_Ensemble SHALL compute a Composite_Risk_Score as the weighted average of 12 assay probabilities
8. WHEN the Composite_Risk_Score is greater than 0.6, THE Platform SHALL classify the Risk_Level as HIGH
9. WHEN the Composite_Risk_Score is between 0.35 and 0.6 inclusive, THE Platform SHALL classify the Risk_Level as MEDIUM
10. WHEN the Composite_Risk_Score is less than 0.35, THE Platform SHALL classify the Risk_Level as LOW
11. THE Conformal_Prediction module SHALL provide calibrated prediction intervals with 85% coverage (alpha=0.15)
12. FOR ALL molecules in the test set, THE ML_Ensemble SHALL achieve a mean AUROC of at least 0.80 across 12 Tox21_Assays


### Requirement 3: Atom-Level Explainability

**User Story:** As a User, I want to see which atoms in a molecule contribute most to toxicity predictions, so that I can understand structural drivers and guide molecular modifications.

#### Acceptance Criteria

1. WHEN a prediction is generated, THE XAI_Engine SHALL compute per-atom attribution scores using Captum IntegratedGradients on the GNN_Model
2. THE XAI_Engine SHALL normalize atom attribution scores to the range [0, 1]
3. THE XAI_Engine SHALL generate a Heatmap by coloring atoms on the 2D molecular structure where red indicates high toxic contribution and blue indicates protective contribution
4. WHEN a User requests a prediction, THE Backend SHALL return the Heatmap as a base64-encoded PNG image
5. THE XAI_Engine SHALL compute SHAP_Values for the top 10 molecular descriptors using TreeExplainer on the LightGBM_Model
6. THE Backend SHALL return SHAP_Values with feature names, feature values, SHAP contributions, and direction indicators (toxic or protective)
7. WHERE the ChemBERTa_Model is used, THE XAI_Engine SHALL compute token-level attention scores using Grad-CAM
8. FOR ALL predictions, THE XAI_Engine SHALL complete attribution computation within 800 milliseconds on CPU


### Requirement 4: Structural Alert Detection

**User Story:** As a User, I want to identify known toxicophore patterns in molecules, so that I can validate ML predictions against established medicinal chemistry knowledge.

#### Acceptance Criteria

1. THE XAI_Engine SHALL maintain a library of at least 150 SMARTS patterns representing known structural alerts
2. WHEN a molecule is analyzed, THE XAI_Engine SHALL scan for all matching structural alert patterns
3. THE XAI_Engine SHALL classify each detected alert with a severity level (HIGH, MEDIUM, or LOW)
4. THE XAI_Engine SHALL identify the specific atom indices matching each structural alert pattern
5. THE Backend SHALL return a list of detected alerts including alert name, SMARTS pattern, severity level, and matching atom indices
6. THE XAI_Engine SHALL include alerts for quinones, nitro aromatics, epoxides, Michael acceptors, aldehydes, anilines, acyl halides, and halogenated alkenes
7. WHEN no structural alerts are detected, THE Backend SHALL return an empty alerts list
8. THE XAI_Engine SHALL complete structural alert scanning within 50 milliseconds per molecule


### Requirement 5: ADMET Property Prediction

**User Story:** As a User, I want to assess drug-likeness and ADMET properties alongside toxicity predictions, so that I can evaluate overall compound viability.

#### Acceptance Criteria

1. WHEN a molecule is analyzed, THE Backend SHALL compute Quantitative Estimate of Drug-likeness (QED) score in the range [0, 1]
2. THE Backend SHALL evaluate Lipinski Rule of Five violations (molecular weight, logP, hydrogen bond donors, hydrogen bond acceptors)
3. THE Backend SHALL compute topological polar surface area (TPSA) in square angstroms
4. THE Backend SHALL predict blood-brain barrier (BBB) penetration likelihood
5. THE Backend SHALL estimate oral bioavailability based on Caco-2 permeability
6. WHERE ADMET-AI library is available, THE Backend SHALL predict CYP2D6 and CYP3A4 inhibition probabilities
7. WHERE ADMET-AI library is available, THE Backend SHALL predict hERG inhibition probability (cardiotoxicity risk)
8. THE Backend SHALL estimate water solubility (logS)
9. THE Backend SHALL return all ADMET_Properties in a structured format with property names and computed values
10. THE Backend SHALL compute ADMET_Properties within 100 milliseconds per molecule


### Requirement 6: LLM-Powered Assessment Reports

**User Story:** As a User, I want to generate comprehensive toxicity assessment reports in natural language, so that I can share findings with colleagues and document compound evaluations.

#### Acceptance Criteria

1. WHEN a User requests a report, THE Report_Generator SHALL construct a structured context including SMILES, molecular properties, Tox21 predictions, SHAP_Values, structural alerts, and Conformal_Prediction intervals
2. THE Report_Generator SHALL send the context to an LLM API (Claude, Groq, or Mistral) with a system prompt defining the role as a senior medicinal chemist
3. THE Report_Generator SHALL request a report structured with sections: Executive Summary, Pathway Analysis, Structural Drivers, De-Risking Recommendations, Regulatory Outlook, and Confidence Assessment
4. THE Report_Generator SHALL limit LLM output to 1500 tokens
5. WHEN the LLM response is received, THE Report_Generator SHALL return the report text to the Frontend
6. WHERE streaming is supported, THE Report_Generator SHALL stream report text progressively to the Frontend
7. THE Report_Generator SHALL complete report generation within 20 seconds
8. IF the LLM API request fails, THEN THE Report_Generator SHALL return an error message indicating the failure reason


### Requirement 7: PDF Report Export

**User Story:** As a User, I want to download toxicity assessment reports as PDF files, so that I can archive evaluations and include them in presentations or regulatory submissions.

#### Acceptance Criteria

1. WHEN a User requests PDF export, THE PDF_Exporter SHALL render the report text in a professional document template
2. THE PDF_Exporter SHALL include the 2D molecular structure image in the PDF
3. THE PDF_Exporter SHALL include the SHAP importance chart in the PDF
4. THE PDF_Exporter SHALL include the Tox21 assay predictions table in the PDF
5. THE PDF_Exporter SHALL include detected structural alerts in the PDF
6. THE PDF_Exporter SHALL use WeasyPrint to convert HTML to PDF format
7. THE PDF_Exporter SHALL generate a 2-page PDF document
8. WHEN PDF generation is complete, THE Backend SHALL return the PDF as downloadable binary content
9. THE PDF_Exporter SHALL complete PDF generation within 5 seconds
10. THE PDF_Exporter SHALL include ToxiLens branding and generation timestamp in the PDF footer


### Requirement 8: Batch Virtual Screening

**User Story:** As a User, I want to upload CSV files containing multiple SMILES strings, so that I can screen compound libraries and prioritize molecules for synthesis or testing.

#### Acceptance Criteria

1. WHEN a User uploads a CSV file, THE Backend SHALL parse the file and extract the SMILES column
2. IF the CSV file does not contain a SMILES column, THEN THE Backend SHALL return an error message indicating the missing column
3. THE Backend SHALL process each SMILES string in the batch sequentially
4. THE Backend SHALL compute Composite_Risk_Score for each molecule in the batch
5. THE Backend SHALL rank batch results by Composite_Risk_Score in descending order
6. THE Backend SHALL return batch results including SMILES, Composite_Risk_Score, Risk_Level, and flagged Tox21_Assays for each molecule
7. WHEN batch processing is complete, THE Backend SHALL provide results in JSON format suitable for table display
8. THE Backend SHALL process batches of 100 molecules within 12 seconds on CPU
9. WHERE GPU is available, THE Backend SHALL process batches of 100 molecules within 2 seconds
10. THE Frontend SHALL display batch results in a sortable table with filtering by risk threshold


### Requirement 9: Chemical Space Exploration

**User Story:** As a User, I want to visualize where my query molecule sits in chemical space relative to the Tox21 dataset, so that I can identify structurally similar compounds and assess prediction reliability.

#### Acceptance Criteria

1. THE Backend SHALL precompute UMAP_Embedding for all 12,000 Tox21 training molecules using Morgan fingerprints
2. THE Backend SHALL store UMAP_Embedding coordinates, SMILES strings, and toxicity labels in a JSON file
3. WHEN a User submits a query molecule, THE Backend SHALL project the molecule into the precomputed UMAP space
4. THE Backend SHALL compute Tanimoto similarity between the query molecule and all Tox21 molecules
5. THE Backend SHALL identify the 10 nearest neighbors in chemical space
6. THE Frontend SHALL display an interactive Plotly scatter plot with 12,000 Tox21 points
7. THE Frontend SHALL highlight the query molecule as a distinct marker (star or pulsing circle)
8. THE Frontend SHALL color points by toxicity label or first flagged assay
9. WHEN a User hovers over a point, THE Frontend SHALL display the SMILES string and compound identifier
10. WHEN a User clicks a point, THE Frontend SHALL load that molecule into the single analysis view
11. THE Frontend SHALL render the UMAP plot with 12,000 points without performance degradation (60 FPS)


### Requirement 10: De-Risking Lab with Bioisostere Generation

**User Story:** As a User, I want to automatically generate structurally modified variants of toxic molecules, so that I can explore de-risking strategies and identify safer alternatives.

#### Acceptance Criteria

1. WHEN a User submits a molecule to the De_Risking_Lab, THE Backend SHALL scan for detected structural alerts
2. WHEN a structural alert is detected, THE Backend SHALL apply bioisostere substitution rules to generate molecular variants
3. THE Backend SHALL apply substitution rules including: NO2 to CN, Cl to F, aldehyde to alcohol, and quinone to phenol
4. THE Backend SHALL validate each generated variant SMILES string using RDKit
5. THE Backend SHALL reject invalid variants that fail SMILES parsing
6. THE Backend SHALL generate at least 3 and at most 8 valid bioisostere variants per input molecule
7. THE Backend SHALL compute toxicity predictions for all generated variants
8. THE Backend SHALL compute delta risk scores (original Composite_Risk_Score minus variant Composite_Risk_Score) for each variant
9. THE Backend SHALL rank variants by delta risk score in descending order
10. THE Backend SHALL return variant results including modified SMILES, Composite_Risk_Score, delta risk, and applied modification description
11. WHERE LLM API is available, THE Report_Generator SHALL generate a rationale explaining each bioisostere modification
12. THE Backend SHALL complete de-risking analysis with 8 variants within 3 seconds on GPU


### Requirement 11: Multi-Molecule Comparison

**User Story:** As a User, I want to compare toxicity profiles of multiple molecules side-by-side, so that I can evaluate lead compound candidates and prioritize synthesis.

#### Acceptance Criteria

1. WHEN a User submits 2 to 5 SMILES strings, THE Backend SHALL process each molecule independently
2. THE Backend SHALL return predictions for all submitted molecules in a single response
3. THE Frontend SHALL display a heatmap grid with Tox21_Assays as rows and molecules as columns
4. THE Frontend SHALL color each cell in the heatmap grid based on toxicity probability (red for high, amber for medium, green for low)
5. THE Frontend SHALL display 2D molecular structures for all compared molecules
6. THE Frontend SHALL compute and display delta toxicity values relative to the first molecule (baseline)
7. WHEN a User hovers over a heatmap cell, THE Frontend SHALL display the exact probability value and assay name
8. THE Backend SHALL process multi-molecule comparison requests with 5 molecules within 1 second on GPU
9. IF a User submits fewer than 2 or more than 5 molecules, THEN THE Backend SHALL return an error message indicating the valid range


### Requirement 12: Model Training and Evaluation

**User Story:** As a developer, I want to train and evaluate ML models on Tox21 data with proper validation methodology, so that I can ensure robust generalization to unseen molecules.

#### Acceptance Criteria

1. THE Platform SHALL split the Tox21 dataset using Scaffold_Split (Bemis-Murcko) with 80% training, 10% validation, and 10% test
2. THE Platform SHALL NOT use random splitting for train/validation/test division
3. WHEN training the LightGBM_Model, THE Platform SHALL apply per-assay positive class weights computed as n_negative / n_positive
4. WHEN training the GNN_Model, THE Platform SHALL apply joint correlation loss combining masked BCE loss and correlation consistency loss
5. WHEN training the ChemBERTa_Model, THE Platform SHALL use a learning rate of 2e-5 with AdamW optimizer
6. THE Platform SHALL train the ChemBERTa_Model for 5 to 8 epochs with early stopping based on validation AUROC
7. THE Platform SHALL train the GNN_Model for up to 100 epochs with early stopping (patience=15) based on validation AUROC
8. THE Platform SHALL handle missing labels by masking them in the loss function
9. THE Platform SHALL evaluate all models using AUROC as the primary metric
10. THE Platform SHALL compute mean AUROC across all 12 Tox21_Assays as the final performance metric
11. THE Platform SHALL save trained model artifacts including model weights, scaler, SHAP background set, and ensemble weights
12. THE Platform SHALL achieve a mean AUROC of at least 0.80 on the scaffold-split test set


### Requirement 13: API Endpoints and Response Format

**User Story:** As a frontend developer or API consumer, I want well-defined REST endpoints with consistent response formats, so that I can integrate with the backend reliably.

#### Acceptance Criteria

1. THE Backend SHALL expose a POST /predict endpoint accepting JSON with a "smiles" field
2. THE Backend SHALL expose a POST /predict_batch endpoint accepting CSV file uploads
3. THE Backend SHALL expose a POST /generate_report endpoint accepting JSON with prediction context
4. THE Backend SHALL expose a POST /what_if endpoint accepting JSON with original and modified SMILES
5. THE Backend SHALL expose a POST /derisk endpoint accepting JSON with a SMILES string
6. THE Backend SHALL expose a GET /similar endpoint accepting a SMILES query parameter
7. THE Backend SHALL expose a GET /health endpoint returning service status
8. THE Backend SHALL expose a GET /docs endpoint serving auto-generated Swagger UI documentation
9. THE Backend SHALL validate all request payloads using Pydantic schemas
10. IF request validation fails, THEN THE Backend SHALL return a 422 status code with detailed validation errors
11. THE Backend SHALL return prediction responses in JSON format including predictions, composite_risk, risk_level, shap_top10, alerts, admet_properties, heatmap_image, and conformal_intervals
12. THE Backend SHALL include CORS headers allowing requests from the Frontend origin
13. THE Backend SHALL log all requests with timestamps, endpoints, and processing times
14. THE Backend SHALL return appropriate HTTP status codes (200 for success, 400 for bad request, 422 for validation error, 500 for server error)


### Requirement 14: Frontend User Interface

**User Story:** As a User, I want an intuitive web interface with multiple analysis modes, so that I can interact with the platform without command-line tools or API knowledge.

#### Acceptance Criteria

1. THE Frontend SHALL provide a Single Analysis page with SMILES text input and molecule drawing widget
2. THE Frontend SHALL provide a Chemical Space Explorer page with interactive UMAP visualization
3. THE Frontend SHALL provide a Batch Screening page with CSV upload and results table
4. THE Frontend SHALL provide a De-Risking Lab page with variant generation and comparison
5. THE Frontend SHALL provide a Multi-Molecule Comparison page with side-by-side heatmap grid
6. THE Frontend SHALL display 2D molecular structures with atom-level heatmap overlays
7. THE Frontend SHALL display a radar chart visualizing all 12 Tox21_Assay probabilities
8. THE Frontend SHALL display a horizontal bar chart showing top 10 SHAP_Values
9. THE Frontend SHALL display structural alert badges with severity color coding (red for HIGH, amber for MEDIUM, teal for LOW)
10. THE Frontend SHALL display an ADMET properties panel with drug-likeness metrics
11. THE Frontend SHALL provide a "Generate Report" button that triggers LLM report generation
12. THE Frontend SHALL display a loading animation during report generation
13. THE Frontend SHALL provide a "Download PDF" button after report generation completes
14. THE Frontend SHALL use TypeScript for type safety
15. THE Frontend SHALL use Tailwind CSS for styling
16. THE Frontend SHALL be responsive and functional on desktop browsers (minimum 1280px width)


### Requirement 15: Model Startup and Preloading

**User Story:** As a developer, I want ML models to be loaded into memory at application startup, so that prediction requests have minimal latency without model loading overhead.

#### Acceptance Criteria

1. WHEN the Backend starts, THE Backend SHALL load the LightGBM_Model from the artifacts directory
2. WHEN the Backend starts, THE Backend SHALL load the GNN_Model weights from the artifacts directory
3. WHEN the Backend starts, THE Backend SHALL load the ChemBERTa_Model from the artifacts directory
4. WHEN the Backend starts, THE Backend SHALL load the feature scaler from the artifacts directory
5. WHEN the Backend starts, THE Backend SHALL load the SHAP background set from the artifacts directory
6. WHEN the Backend starts, THE Backend SHALL load the ensemble weights from the artifacts directory
7. WHEN the Backend starts, THE Backend SHALL load the UMAP_Embedding data from the artifacts directory
8. IF any model artifact is missing, THEN THE Backend SHALL log an error and fail to start
9. THE Backend SHALL complete model preloading within 30 seconds on CPU
10. WHERE GPU is available, THE Backend SHALL load models onto GPU memory
11. WHEN model preloading is complete, THE Backend SHALL log a ready message indicating successful initialization
12. THE Backend SHALL respond to the /health endpoint with status "ready" only after all models are loaded


### Requirement 16: Docker Deployment

**User Story:** As a developer or deployment engineer, I want to run the entire platform using Docker Compose, so that I can deploy consistently across environments without dependency conflicts.

#### Acceptance Criteria

1. THE Platform SHALL provide a docker-compose.yml file defining backend and frontend services
2. THE Platform SHALL provide a backend Dockerfile that installs Python dependencies and copies application code
3. THE Platform SHALL provide a frontend Dockerfile that builds the React application and serves it via nginx
4. WHEN a developer runs docker-compose up, THE Platform SHALL start both backend and frontend services
5. THE backend service SHALL expose port 8000 for API requests
6. THE frontend service SHALL expose port 3000 for web access
7. THE docker-compose.yml SHALL define a shared volume for model artifacts
8. THE backend Dockerfile SHALL install RDKit via conda
9. THE backend Dockerfile SHALL install PyTorch with CUDA support where available
10. THE frontend Dockerfile SHALL build the React application using npm run build
11. WHEN Docker containers start, THE Platform SHALL be accessible at http://localhost:3000 within 60 seconds
12. THE Platform SHALL provide a .env.example file documenting required environment variables


### Requirement 17: Error Handling and Validation

**User Story:** As a User, I want clear error messages when something goes wrong, so that I can correct my input and successfully complete my analysis.

#### Acceptance Criteria

1. IF a User submits an invalid SMILES string, THEN THE Backend SHALL return an error message stating "Invalid SMILES: unable to parse molecular structure"
2. IF a User uploads a CSV file without a SMILES column, THEN THE Backend SHALL return an error message stating "CSV must contain a column named 'smiles'"
3. IF a prediction request fails due to a model error, THEN THE Backend SHALL return an error message stating "Prediction failed: internal model error" and log the full stack trace
4. IF the LLM API request fails, THEN THE Backend SHALL return an error message stating "Report generation failed: LLM API unavailable"
5. IF the LLM API key is missing, THEN THE Backend SHALL return an error message stating "Report generation unavailable: API key not configured"
6. IF a User submits a batch with more than 1000 molecules, THEN THE Backend SHALL return an error message stating "Batch size exceeds maximum of 1000 molecules"
7. IF a User submits an empty SMILES string, THEN THE Backend SHALL return an error message stating "SMILES string cannot be empty"
8. THE Backend SHALL validate all numeric inputs are within expected ranges
9. THE Backend SHALL sanitize all user inputs to prevent injection attacks
10. THE Frontend SHALL display error messages in a prominent notification banner
11. THE Frontend SHALL clear error messages when a User submits a new request
12. THE Backend SHALL return error responses with appropriate HTTP status codes and JSON format including "error" and "detail" fields


### Requirement 18: Configuration Management

**User Story:** As a developer, I want to configure the platform via environment variables, so that I can deploy to different environments without code changes.

#### Acceptance Criteria

1. THE Backend SHALL read configuration from environment variables using pydantic-settings
2. THE Backend SHALL support configuration of MODEL_ARTIFACTS_PATH for model file locations
3. THE Backend SHALL support configuration of ANTHROPIC_API_KEY for LLM report generation
4. THE Backend SHALL support configuration of GROQ_API_KEY as an alternative LLM provider
5. THE Backend SHALL support configuration of MISTRAL_API_KEY as an alternative LLM provider
6. THE Backend SHALL support configuration of CORS_ORIGINS for allowed frontend origins
7. THE Backend SHALL support configuration of LOG_LEVEL (DEBUG, INFO, WARNING, ERROR)
8. THE Backend SHALL support configuration of MAX_BATCH_SIZE for batch screening limits
9. THE Backend SHALL support configuration of DEVICE (cpu or cuda) for model inference
10. THE Backend SHALL provide default values for all configuration parameters
11. THE Backend SHALL validate configuration on startup and fail fast if required parameters are invalid
12. THE Platform SHALL provide a .env.example file documenting all configuration parameters with example values


### Requirement 19: Logging and Monitoring

**User Story:** As a developer or system administrator, I want comprehensive logging of platform operations, so that I can debug issues and monitor performance.

#### Acceptance Criteria

1. THE Backend SHALL log all incoming API requests with timestamp, endpoint, method, and client IP
2. THE Backend SHALL log prediction processing times for each request
3. THE Backend SHALL log model loading events during startup
4. THE Backend SHALL log errors with full stack traces at ERROR level
5. THE Backend SHALL log warnings for invalid inputs at WARNING level
6. THE Backend SHALL log successful predictions at INFO level
7. THE Backend SHALL log debug information including feature computation times at DEBUG level
8. THE Backend SHALL write logs to stdout in JSON format for structured logging
9. THE Backend SHALL include request IDs in all log entries for request tracing
10. THE Backend SHALL log LLM API calls including token usage and response times
11. THE Backend SHALL NOT log sensitive information including API keys or personal data
12. WHERE LOG_LEVEL is set to DEBUG, THE Backend SHALL log detailed model inference information


### Requirement 20: Performance and Scalability

**User Story:** As a User, I want fast response times even when analyzing complex molecules, so that I can iterate quickly during drug design.

#### Acceptance Criteria

1. THE Backend SHALL process single molecule predictions within 200 milliseconds on CPU
2. WHERE GPU is available, THE Backend SHALL process single molecule predictions within 30 milliseconds
3. THE Backend SHALL process batch predictions of 100 molecules within 12 seconds on CPU
4. WHERE GPU is available, THE Backend SHALL process batch predictions of 100 molecules within 2 seconds
5. THE Backend SHALL compute SHAP and Captum attributions within 800 milliseconds on CPU
6. THE Backend SHALL generate heatmap images within 100 milliseconds
7. THE Backend SHALL scan structural alerts within 50 milliseconds per molecule
8. THE Backend SHALL compute ADMET properties within 100 milliseconds per molecule
9. THE Frontend SHALL render UMAP plots with 12,000 points at 60 frames per second
10. THE Frontend SHALL respond to user interactions (clicks, hovers) within 16 milliseconds
11. THE Backend SHALL support at least 10 concurrent prediction requests without performance degradation
12. THE Backend SHALL use connection pooling for LLM API requests to minimize latency


### Requirement 21: Data Preprocessing Pipeline

**User Story:** As a developer, I want a reproducible data preprocessing pipeline, so that I can retrain models with updated datasets or different splitting strategies.

#### Acceptance Criteria

1. THE Platform SHALL provide a preprocess_tox21.py script that downloads Tox21 data from Kaggle
2. THE Platform SHALL standardize all SMILES strings by neutralizing charges, removing salts, and canonicalizing tautomers
3. THE Platform SHALL compute molecular descriptors, fingerprints, and graph representations for all molecules
4. THE Platform SHALL apply Scaffold_Split to create train/validation/test sets with 80/10/10 ratio
5. THE Platform SHALL save processed features to pickle files in ml/data/processed/
6. THE Platform SHALL save scaffold split indices to enable reproducible evaluation
7. THE Platform SHALL compute and save per-assay class weights for handling label imbalance
8. THE Platform SHALL compute and save label correlation matrix for joint correlation loss
9. THE Platform SHALL handle missing labels by marking them with NaN values
10. THE Platform SHALL log preprocessing statistics including number of valid molecules, descriptor ranges, and label distributions
11. THE Platform SHALL complete preprocessing of 12,000 Tox21 molecules within 10 minutes on CPU
12. FOR ALL molecules, preprocessing then re-preprocessing SHALL produce identical feature vectors (idempotence property)


### Requirement 22: Testing and Quality Assurance

**User Story:** As a developer, I want automated tests covering critical functionality, so that I can detect regressions and ensure platform reliability.

#### Acceptance Criteria

1. THE Platform SHALL provide unit tests for SMILES parsing and validation
2. THE Platform SHALL provide unit tests for descriptor computation
3. THE Platform SHALL provide unit tests for fingerprint computation
4. THE Platform SHALL provide unit tests for graph construction
5. THE Platform SHALL provide unit tests for structural alert scanning
6. THE Platform SHALL provide integration tests for the /predict endpoint
7. THE Platform SHALL provide integration tests for the /predict_batch endpoint
8. THE Platform SHALL provide integration tests for the /generate_report endpoint
9. THE Platform SHALL provide property-based tests verifying SMILES standardization idempotence
10. THE Platform SHALL provide property-based tests verifying descriptor computation determinism
11. THE Platform SHALL provide tests verifying error handling for invalid inputs
12. THE Platform SHALL achieve at least 70% code coverage for backend modules
13. THE Platform SHALL run all tests successfully before deployment
14. THE Platform SHALL provide a test suite that completes within 5 minutes


### Requirement 23: Documentation and Examples

**User Story:** As a new User or developer, I want comprehensive documentation with examples, so that I can quickly understand how to use the platform and integrate it into my workflow.

#### Acceptance Criteria

1. THE Platform SHALL provide a README.md file with project overview, features, and installation instructions
2. THE Platform SHALL provide API documentation via auto-generated Swagger UI at /docs endpoint
3. THE Platform SHALL provide example SMILES strings for common drug molecules (aspirin, ibuprofen, caffeine)
4. THE Platform SHALL provide example CSV files for batch screening demonstrations
5. THE Platform SHALL provide architecture diagrams showing system components and data flow
6. THE Platform SHALL provide code comments explaining complex algorithms and design decisions
7. THE Platform SHALL provide a model card documenting training data, performance metrics, and limitations
8. THE Platform SHALL provide usage examples for each API endpoint with curl commands
9. THE Platform SHALL provide troubleshooting guidance for common errors
10. THE Platform SHALL provide contribution guidelines for developers
11. THE Platform SHALL provide license information (MIT license)
12. THE Platform SHALL provide citations for research papers and methods used


### Requirement 24: Security and Data Privacy

**User Story:** As a User, I want my molecular data to be handled securely, so that I can analyze proprietary compounds without risk of data leakage.

#### Acceptance Criteria

1. THE Backend SHALL NOT store submitted SMILES strings or molecular structures in persistent storage
2. THE Backend SHALL NOT log SMILES strings or molecular structures in application logs
3. THE Backend SHALL process all predictions in-memory without writing temporary files
4. THE Backend SHALL validate and sanitize all user inputs to prevent injection attacks
5. THE Backend SHALL use HTTPS for all API communications in production deployments
6. THE Backend SHALL NOT expose internal error details or stack traces to API responses in production mode
7. THE Backend SHALL store API keys in environment variables and NOT in source code
8. THE Backend SHALL implement rate limiting to prevent abuse (100 requests per minute per IP)
9. THE Backend SHALL validate file uploads to prevent malicious file execution
10. THE Backend SHALL limit CSV upload file size to 10 MB
11. THE Backend SHALL implement CORS restrictions to allow requests only from authorized origins
12. THE Platform SHALL NOT collect or transmit user analytics or telemetry data


### Requirement 25: Hugging Face Spaces Deployment

**User Story:** As a hackathon participant, I want to deploy ToxiLens to Hugging Face Spaces, so that judges and users can access a live demo without local installation.

#### Acceptance Criteria

1. THE Platform SHALL provide a Hugging Face Spaces configuration file (README.md with YAML frontmatter)
2. THE Platform SHALL configure Spaces to use a T4 GPU for model inference
3. THE Platform SHALL configure Spaces to use Python 3.11 runtime
4. THE Platform SHALL provide a requirements.txt file compatible with Hugging Face Spaces
5. THE Platform SHALL provide an app.py entry point for Spaces deployment
6. THE Platform SHALL include model artifacts in the Spaces repository or download them at startup
7. THE Platform SHALL configure Spaces secrets for API keys (ANTHROPIC_API_KEY)
8. WHEN deployed to Spaces, THE Platform SHALL be accessible via a public URL
9. WHEN deployed to Spaces, THE Platform SHALL complete startup and model loading within 120 seconds
10. THE Platform SHALL provide deployment documentation with step-by-step Spaces setup instructions
11. THE Platform SHALL configure Spaces to automatically restart on failure
12. THE Platform SHALL display a custom Spaces README with project description, demo GIF, and usage instructions


### Requirement 26: Graph Neural Network Architecture

**User Story:** As a developer, I want a multi-task GNN with joint correlation loss, so that I can leverage molecular graph structure and cross-assay relationships for improved predictions.

#### Acceptance Criteria

1. THE GNN_Model SHALL use AttentiveFP or GIN architecture with 4 graph convolution layers
2. THE GNN_Model SHALL use a hidden dimension of 256
3. THE GNN_Model SHALL apply global mean pooling and global max pooling to node embeddings
4. THE GNN_Model SHALL concatenate pooled embeddings to create a 512-dimensional graph representation
5. THE GNN_Model SHALL apply dropout with probability 0.3 for regularization
6. THE GNN_Model SHALL use 12 independent sigmoid output heads (one per Tox21_Assay)
7. THE GNN_Model SHALL compute standard masked binary cross-entropy loss for labeled examples
8. THE GNN_Model SHALL compute correlation consistency loss based on precomputed label correlation matrix
9. THE GNN_Model SHALL combine standard loss and correlation loss with weight lambda=0.1
10. THE GNN_Model SHALL train using AdamW optimizer with learning rate 1e-3 and weight decay 1e-4
11. THE GNN_Model SHALL use CosineAnnealingLR scheduler with T_max=50
12. THE GNN_Model SHALL implement early stopping with patience=15 based on validation mean AUROC
13. THE GNN_Model SHALL train for up to 100 epochs
14. THE GNN_Model SHALL achieve a mean AUROC of at least 0.80 on the scaffold-split test set


### Requirement 27: ChemBERTa-2 Fine-Tuning

**User Story:** As a developer, I want to fine-tune ChemBERTa-2 on Tox21 data, so that I can leverage pre-trained chemical language understanding for toxicity prediction.

#### Acceptance Criteria

1. THE ChemBERTa_Model SHALL load pretrained weights from "seyonec/ChemBERTa-zinc-base-v1"
2. THE ChemBERTa_Model SHALL add a 12-class multi-label classification head
3. THE ChemBERTa_Model SHALL tokenize SMILES strings using the ChemBERTa tokenizer
4. THE ChemBERTa_Model SHALL extract the CLS token embedding (768-dimensional)
5. THE ChemBERTa_Model SHALL apply dropout with probability 0.1 before the classification head
6. THE ChemBERTa_Model SHALL train using AdamW optimizer with learning rate 2e-5
7. THE ChemBERTa_Model SHALL use linear warmup for 10% of training steps followed by cosine decay
8. THE ChemBERTa_Model SHALL train for 5 to 8 epochs
9. THE ChemBERTa_Model SHALL use batch size 32 with gradient accumulation if GPU memory is insufficient
10. THE ChemBERTa_Model SHALL use mixed precision training (fp16) via torch.cuda.amp.autocast
11. THE ChemBERTa_Model SHALL handle missing labels by setting them to -1 and masking in loss computation
12. THE ChemBERTa_Model SHALL save the best checkpoint based on validation mean AUROC
13. THE ChemBERTa_Model SHALL achieve a mean AUROC of at least 0.78 on the scaffold-split test set


### Requirement 28: Ensemble Weight Optimization

**User Story:** As a developer, I want to optimize ensemble weights on a validation set, so that I can maximize the combined predictive performance of all three models.

#### Acceptance Criteria

1. THE ML_Ensemble SHALL compute logits from LightGBM_Model probabilities using inverse sigmoid transformation
2. THE ML_Ensemble SHALL extract logits directly from GNN_Model and ChemBERTa_Model outputs
3. THE ML_Ensemble SHALL define an objective function that computes weighted logit fusion and evaluates mean AUROC on validation set
4. THE ML_Ensemble SHALL optimize ensemble weights using Nelder-Mead simplex algorithm
5. THE ML_Ensemble SHALL initialize weights to [1, 1, 1] before optimization
6. THE ML_Ensemble SHALL apply softmax normalization to ensure weights sum to 1
7. THE ML_Ensemble SHALL save optimized weights to ensemble_weights.json
8. THE ML_Ensemble SHALL load saved weights during inference
9. THE ML_Ensemble SHALL compute final predictions by applying sigmoid to weighted logit sum
10. THE ML_Ensemble SHALL achieve a mean AUROC improvement of at least 0.02 over the best individual model
11. THE ML_Ensemble SHALL complete weight optimization within 5 minutes on validation set


### Requirement 29: Conformal Prediction Integration

**User Story:** As a User, I want uncertainty estimates for predictions, so that I can assess confidence and identify molecules where the model is uncertain.

#### Acceptance Criteria

1. THE ML_Ensemble SHALL wrap the ensemble model with MAPIE MapieClassifier
2. THE ML_Ensemble SHALL calibrate conformal prediction on a held-out calibration set (separate from test set)
3. THE ML_Ensemble SHALL use alpha=0.15 to target 85% coverage
4. THE ML_Ensemble SHALL generate prediction sets for each Tox21_Assay
5. THE ML_Ensemble SHALL return prediction sets as one of: {SAFE}, {TOXIC}, or {SAFE, TOXIC}
6. WHEN a prediction set contains both SAFE and TOXIC, THE Backend SHALL mark the prediction as uncertain
7. THE Backend SHALL return conformal intervals in the prediction response
8. THE Backend SHALL compute the percentage of uncertain predictions across all 12 assays
9. THE ML_Ensemble SHALL achieve at least 80% empirical coverage on the test set
10. THE ML_Ensemble SHALL complete conformal prediction within 50 milliseconds additional latency per molecule


### Requirement 30: UMAP Precomputation and Projection

**User Story:** As a developer, I want to precompute UMAP embeddings for the Tox21 dataset, so that I can provide fast chemical space visualization without recomputing embeddings at query time.

#### Acceptance Criteria

1. THE Platform SHALL compute UMAP embeddings using Morgan fingerprints as input features
2. THE Platform SHALL configure UMAP with n_components=2, n_neighbors=15, min_dist=0.1, and random_state=42
3. THE Platform SHALL use Jaccard metric for binary fingerprint similarity
4. THE Platform SHALL fit UMAP on all Tox21 training molecules
5. THE Platform SHALL save the fitted UMAP reducer to enable projection of new molecules
6. THE Platform SHALL save embedding coordinates, SMILES strings, and toxicity labels to umap_data.json
7. WHEN a User submits a query molecule, THE Backend SHALL project the molecule into the precomputed UMAP space using the fitted reducer
8. THE Backend SHALL return the query molecule's 2D coordinates for visualization
9. THE Platform SHALL complete UMAP fitting on 12,000 molecules within 10 minutes
10. THE Backend SHALL project a new molecule into UMAP space within 20 milliseconds


### Requirement 31: Similarity Search

**User Story:** As a User, I want to find structurally similar molecules in the Tox21 dataset, so that I can compare toxicity profiles and validate predictions against known compounds.

#### Acceptance Criteria

1. WHEN a User requests similar molecules, THE Backend SHALL compute Tanimoto similarity between the query molecule and all Tox21 molecules using Morgan fingerprints
2. THE Backend SHALL rank Tox21 molecules by Tanimoto similarity in descending order
3. THE Backend SHALL return the top 10 most similar molecules
4. THE Backend SHALL include SMILES, Tanimoto similarity score, and toxicity labels for each similar molecule
5. THE Backend SHALL exclude the query molecule itself from similarity results if it exists in Tox21
6. THE Backend SHALL compute similarity scores within 100 milliseconds for 12,000 comparisons
7. THE Backend SHALL return similarity scores rounded to 3 decimal places
8. WHEN no molecules have Tanimoto similarity above 0.3, THE Backend SHALL return an empty results list with a message indicating no similar compounds found


### Requirement 32: What-If Analysis

**User Story:** As a User, I want to compare toxicity predictions before and after molecular modifications, so that I can evaluate the impact of structural changes.

#### Acceptance Criteria

1. WHEN a User submits original and modified SMILES strings, THE Backend SHALL validate both SMILES strings
2. THE Backend SHALL compute toxicity predictions for both the original and modified molecules
3. THE Backend SHALL compute delta toxicity (modified minus original) for each Tox21_Assay
4. THE Backend SHALL compute delta Composite_Risk_Score
5. THE Backend SHALL identify which assays improved (negative delta) and which worsened (positive delta)
6. THE Backend SHALL return a comparison response including both predictions, delta values, and improvement summary
7. THE Backend SHALL generate side-by-side heatmap images for visual comparison
8. THE Backend SHALL highlight atoms that differ between original and modified structures
9. THE Backend SHALL process what-if analysis within 400 milliseconds on CPU
10. IF the modified SMILES is identical to the original, THEN THE Backend SHALL return an error message stating "Modified structure is identical to original"


### Requirement 33: Frontend Visualization Components

**User Story:** As a User, I want rich interactive visualizations of toxicity data, so that I can quickly interpret results and identify patterns.

#### Acceptance Criteria

1. THE Frontend SHALL render 2D molecular structures with atom-level heatmap overlays
2. THE Frontend SHALL display a radar chart with 12 axes (one per Tox21_Assay) showing probability values
3. THE Frontend SHALL display a horizontal bar chart showing top 10 SHAP_Values with color coding (red for toxic contribution, blue for protective)
4. THE Frontend SHALL display structural alert badges positioned near flagged atoms on the molecular structure
5. THE Frontend SHALL display an ADMET properties panel with metrics including QED, Lipinski violations, TPSA, logP, and BBB penetration
6. THE Frontend SHALL use Recharts library for radar and bar charts
7. THE Frontend SHALL use Plotly.js for UMAP scatter plots with WebGL acceleration
8. THE Frontend SHALL provide zoom and pan controls for molecular structure viewer
9. THE Frontend SHALL provide tooltips on hover for all chart elements showing exact values
10. THE Frontend SHALL use color coding consistently: red for high risk (>0.7), amber for medium risk (0.4-0.7), green for low risk (<0.4)
11. THE Frontend SHALL render all visualizations within 100 milliseconds after receiving API response
12. THE Frontend SHALL maintain 60 frames per second during user interactions with visualizations


### Requirement 34: Preset Example Molecules

**User Story:** As a User, I want to quickly test the platform with example molecules, so that I can explore features without needing to find SMILES strings.

#### Acceptance Criteria

1. THE Frontend SHALL provide preset buttons for at least 5 example molecules
2. THE Frontend SHALL include aspirin (CC(=O)Oc1ccccc1C(=O)O) as an example molecule
3. THE Frontend SHALL include ibuprofen as an example molecule
4. THE Frontend SHALL include caffeine as an example molecule
5. THE Frontend SHALL include bisphenol A as an example molecule
6. THE Frontend SHALL include doxorubicin as an example molecule
7. WHEN a User clicks a preset button, THE Frontend SHALL populate the SMILES input field and automatically trigger prediction
8. THE Frontend SHALL display the molecule name alongside the preset button
9. THE Frontend SHALL provide a tooltip describing each example molecule (e.g., "Aspirin - common pain reliever")
10. THE Frontend SHALL organize preset buttons in a visually prominent location on the Single Analysis page


### Requirement 35: Responsive Design and Accessibility

**User Story:** As a User, I want the platform to be usable on different screen sizes and accessible to users with disabilities, so that everyone can benefit from the tool.

#### Acceptance Criteria

1. THE Frontend SHALL be fully functional on desktop browsers with minimum viewport width of 1280 pixels
2. THE Frontend SHALL use semantic HTML elements (header, nav, main, section, article)
3. THE Frontend SHALL provide alt text for all images including molecular structures
4. THE Frontend SHALL ensure all interactive elements are keyboard accessible
5. THE Frontend SHALL provide focus indicators for all interactive elements
6. THE Frontend SHALL use ARIA labels for complex interactive components
7. THE Frontend SHALL maintain color contrast ratios of at least 4.5:1 for text
8. THE Frontend SHALL provide text alternatives for color-coded information
9. THE Frontend SHALL support browser zoom up to 200% without breaking layout
10. THE Frontend SHALL use responsive font sizes that scale with viewport
11. THE Frontend SHALL provide loading states with accessible announcements for screen readers
12. THE Frontend SHALL ensure all form inputs have associated labels



## Requirements Summary

This requirements document defines 35 functional and non-functional requirements for the ToxiLens platform, organized into the following categories:

**Core Prediction Capabilities (Requirements 1-2, 12, 26-29)**
- SMILES input processing and validation
- Multi-modal ensemble prediction across 12 Tox21 assays
- Model training with scaffold splitting and proper validation
- GNN architecture with joint correlation loss
- ChemBERTa-2 fine-tuning
- Ensemble weight optimization
- Conformal prediction for uncertainty quantification

**Explainability and Interpretability (Requirements 3-5)**
- Atom-level attribution via Captum IntegratedGradients
- Descriptor importance via SHAP TreeExplainer
- Structural alert detection with 150+ SMARTS patterns
- ADMET property prediction

**Advanced Features (Requirements 6-11, 30-32)**
- LLM-powered assessment report generation
- PDF export with professional formatting
- Batch virtual screening with CSV upload
- Chemical space exploration via UMAP
- De-risking lab with bioisostere generation
- Multi-molecule comparison
- Similarity search
- What-if analysis for structural modifications

**API and Backend (Requirements 13, 15, 17-20)**
- RESTful API with 8 endpoints
- Model preloading at startup
- Comprehensive error handling
- Configuration management via environment variables
- Structured logging and monitoring
- Performance targets (<200ms single prediction on CPU)

**Frontend and User Experience (Requirements 14, 33-35)**
- Five specialized analysis pages
- Rich interactive visualizations (radar charts, heatmaps, UMAP plots)
- Preset example molecules
- Responsive design and accessibility compliance

**Infrastructure and Quality (Requirements 16, 21-25)**
- Docker Compose deployment
- Hugging Face Spaces deployment
- Data preprocessing pipeline
- Automated testing with property-based tests
- Comprehensive documentation
- Security and data privacy measures

**Key Correctness Properties for Property-Based Testing:**

1. **Idempotence**: Standardizing a SMILES string multiple times produces identical results (Requirement 1.9)
2. **Idempotence**: Preprocessing features multiple times produces identical vectors (Requirement 21.12)
3. **Determinism**: Descriptor computation is deterministic for the same molecule (Requirement 22.10)
4. **Invariant**: Ensemble weights sum to 1.0 after softmax normalization (Requirement 28.6)
5. **Metamorphic**: Modified molecules in what-if analysis produce different predictions than originals (Requirement 32.10)
6. **Error Conditions**: Invalid SMILES strings are properly rejected with descriptive errors (Requirements 1.2, 17.1)
7. **Bounds**: All probability predictions are in range [0, 1] (implicit in Requirements 2.6, 2.11)
8. **Coverage**: Conformal prediction achieves at least 80% empirical coverage (Requirement 29.9)

All requirements follow EARS patterns (Ubiquitous, Event-driven, State-driven, Unwanted event, Optional feature) and comply with INCOSE quality rules for clarity, testability, completeness, and positive statements.

