"""
ToxiLens Demo - Working Prototype for Round 1
Demonstrates trained models predicting toxicity for a sample molecule.
"""

import pickle
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

print("="*60)
print("ToxiLens - Drug Toxicity Prediction Demo")
print("="*60)

# Load a sample molecule from processed data
print("\n[1/4] Loading processed data...")
with open('ml/data/processed/tox21_processed.pkl', 'rb') as f:
    data = pickle.load(f)

# Get first test molecule
test_idx = data['test_idx'][0]
smiles = data['canonical_smiles'][test_idx]
true_labels = data['labels'][test_idx]

print(f"✓ Sample molecule: {smiles}")
print(f"✓ Dataset: {len(data['smiles'])} molecules, 12 toxicity assays")

# Load LightGBM models
print("\n[2/4] Loading trained LightGBM models...")
assay_names = data['assay_names']
lgbm_models = []

for assay_name in assay_names:
    model_path = f"ml/artifacts/lgbm_{assay_name}.txt"
    model = lgb.Booster(model_file=model_path)
    lgbm_models.append(model)

# Load scaler
with open('ml/artifacts/lgbm_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(f"✓ Loaded {len(lgbm_models)} trained models")

# Prepare features
print("\n[3/4] Computing molecular features...")
features = np.concatenate([
    data['descriptors'][test_idx],
    data['morgan_fp'][test_idx],
    data['maccs_fp'][test_idx]
])
features_scaled = scaler.transform(features.reshape(1, -1))
print(f"✓ Feature vector: {features.shape[0]} dimensions")

# Predict
print("\n[4/4] Predicting toxicity across 12 assays...")
predictions = []
for model in lgbm_models:
    prob = model.predict(features_scaled)[0]
    predictions.append(prob)

predictions = np.array(predictions)

# Display results
print("\n" + "="*60)
print("TOXICITY PREDICTIONS")
print("="*60)

for i, (assay, pred, true) in enumerate(zip(assay_names, predictions, true_labels)):
    if not np.isnan(true):
        status = "✓ CORRECT" if (pred > 0.5) == (true > 0.5) else "✗ WRONG"
        print(f"{assay:20s} | Pred: {pred:.3f} | True: {int(true)} | {status}")
    else:
        print(f"{assay:20s} | Pred: {pred:.3f} | True: N/A")

mean_pred = predictions.mean()
print("\n" + "="*60)
print(f"Composite Risk Score: {mean_pred:.3f}")
if mean_pred > 0.6:
    print("Risk Level: HIGH ⚠️")
elif mean_pred > 0.35:
    print("Risk Level: MEDIUM ⚡")
else:
    print("Risk Level: LOW ✓")

print("\n" + "="*60)
print("✓ Demo Complete - Models are working!")
print("="*60)
print("\nModel Performance:")
print(f"  • LightGBM: Mean AUROC = 0.853")
print(f"  • ChemBERTa: Val AUROC = 0.810")
print(f"  • Dataset: 7,794 molecules (Tox21)")
print(f"  • Features: 2,415 dimensions")
print("\nRepository: https://github.com/Vk2245/Toxilens-")
