# ToxiLens Examples

This directory contains example files and notebooks to help you get started with ToxiLens.

## Contents

### 1. Example Molecules (`example_molecules.json`)

A curated collection of 10 drug molecules with known toxicity profiles for testing ToxiLens:

- **Aspirin** - Common NSAID, low-medium risk
- **Ibuprofen** - NSAID, low risk
- **Caffeine** - CNS stimulant, low risk
- **Bisphenol A** - Endocrine disruptor, medium-high risk
- **Doxorubicin** - Chemotherapy drug, high risk (cardiotoxic)
- **Tamoxifen** - SERM for breast cancer, medium risk
- **Paracetamol** - Analgesic, low-medium risk
- **Warfarin** - Anticoagulant, medium risk
- **Nicotine** - Alkaloid stimulant, medium risk
- **Benzene** - Known carcinogen, high risk

**Usage:**
```python
import json

with open('examples/example_molecules.json') as f:
    molecules = json.load(f)

for mol in molecules['molecules']:
    print(f"{mol['name']}: {mol['smiles']}")
```

### 2. Batch Screening CSV (`batch_screening_example.csv`)

A sample CSV file with 25 compounds for batch virtual screening demonstrations.

**Format:**
```csv
compound_id,smiles,compound_name,source
DRUG001,CC(=O)Oc1ccccc1C(=O)O,Aspirin,FDA Approved
...
```

**Usage:**
```bash
# Via API
curl -X POST http://localhost:8000/predict_batch \
  -F "file=@examples/batch_screening_example.csv" \
  -F "risk_threshold=0.5"

# Via Python
import requests

with open('examples/batch_screening_example.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict_batch',
        files={'file': f}
    )
    results = response.json()
```

### 3. API Usage Notebook (`api_usage_examples.ipynb`)

A comprehensive Jupyter notebook demonstrating all ToxiLens API endpoints:

**Topics Covered:**
1. Health check and connectivity
2. Single molecule prediction
3. Batch virtual screening
4. Chemical space exploration
5. What-if analysis
6. De-risking lab
7. LLM report generation
8. Multi-molecule comparison
9. Error handling

**Setup:**
```bash
# Install dependencies
pip install jupyter requests pandas matplotlib

# Start backend
cd backend
uvicorn app.main:app --reload --port 8000

# Launch notebook
jupyter notebook examples/api_usage_examples.ipynb
```

## Quick Start Examples

### Example 1: Predict Toxicity for Aspirin

```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    json={'smiles': 'CC(=O)Oc1ccccc1C(=O)O'}
)

data = response.json()
print(f"Risk Level: {data['risk_level']}")
print(f"Composite Risk: {data['composite_risk']:.3f}")
```

### Example 2: Find Similar Molecules

```python
response = requests.get(
    'http://localhost:8000/similar',
    params={'smiles': 'CC(=O)Oc1ccccc1C(=O)O', 'top_k': 5}
)

similar = response.json()
for mol in similar['similar_molecules']:
    print(f"Tanimoto: {mol['tanimoto_similarity']:.3f} - {mol['smiles']}")
```

### Example 3: Compare Before/After Modification

```python
response = requests.post(
    'http://localhost:8000/what_if',
    json={
        'original_smiles': 'CC(=O)Oc1ccccc1C(=O)O',  # Ester
        'modified_smiles': 'CC(=O)Nc1ccccc1C(=O)O'   # Amide
    }
)

comparison = response.json()
print(f"Delta Risk: {comparison['delta_composite_risk']:+.3f}")
```

### Example 4: Generate De-Risked Variants

```python
response = requests.post(
    'http://localhost:8000/derisk',
    json={
        'smiles': 'CC(C)(c1ccc(O)cc1)c1ccc(O)cc1',  # Bisphenol A
        'n_variants': 5
    }
)

variants = response.json()
for v in variants['variants']:
    print(f"{v['modification']}: Risk {v['composite_risk']:.3f}")
```

## Using with Frontend

All example molecules are also available in the ToxiLens web interface:

1. Navigate to http://localhost:3000
2. Click on preset buttons (Aspirin, Doxorubicin, etc.)
3. Or paste SMILES from `example_molecules.json`

## Troubleshooting

**Problem: "Connection refused" error**
```bash
# Ensure backend is running
curl http://localhost:8000/health
```

**Problem: "Invalid SMILES" error**
- Verify SMILES syntax using online validators
- Check for special characters or encoding issues
- Try canonical SMILES from RDKit

**Problem: CSV upload fails**
- Ensure CSV has a column named "smiles" (case-insensitive)
- Check for invalid SMILES in the CSV
- Verify file size is under 10 MB

## Additional Resources

- **API Documentation:** http://localhost:8000/docs
- **Model Card:** [docs/model_card.md](../docs/model_card.md)
- **Main README:** [README.md](../README.md)
- **GitHub Issues:** https://github.com/your-handle/toxilens/issues

## Contributing Examples

Have an interesting molecule or use case? Contribute to this directory!

1. Add molecule to `example_molecules.json`
2. Include description and expected risk level
3. Submit a pull request

**Good examples:**
- Molecules with known toxicity mechanisms
- Interesting structural alerts
- Edge cases that test model limits
- Molecules from recent drug approvals/failures
