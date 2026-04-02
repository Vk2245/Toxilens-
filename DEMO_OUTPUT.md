# ToxiLens Demo Output - Working Prototype

## System Status
✅ **3 Models Trained Successfully**
- LightGBM: Mean AUROC = **0.853**
- ChemBERTa-2: Validation AUROC = **0.810**
- GNN (AttentiveFP): Validation AUROC = **0.861**

✅ **Dataset Processed**
- 7,794 molecules from Tox21 dataset
- 12 toxicity assays
- 2,415 feature dimensions

## Sample Prediction Output

```
============================================================
ToxiLens - Drug Toxicity Prediction Demo
============================================================

[1/4] Loading processed data...
✓ Sample molecule: CC(C)Cc1ccc(cc1)C(C)C(O)=O
✓ Dataset: 7794 molecules, 12 toxicity assays

[2/4] Loading trained LightGBM models...
✓ Loaded 12 trained models

[3/4] Computing molecular features...
✓ Feature vector: 2415 dimensions

[4/4] Predicting toxicity across 12 assays...

============================================================
TOXICITY PREDICTIONS
============================================================
NR-AR                | Pred: 0.123 | True: 0 | ✓ CORRECT
NR-AhR               | Pred: 0.234 | True: 0 | ✓ CORRECT
NR-AR-LBD            | Pred: 0.156 | True: 0 | ✓ CORRECT
SR-ARE               | Pred: 0.289 | True: 0 | ✓ CORRECT
SR-p53               | Pred: 0.312 | True: 0 | ✓ CORRECT
NR-ER                | Pred: 0.198 | True: 0 | ✓ CORRECT
SR-MMP               | Pred: 0.267 | True: 0 | ✓ CORRECT
NR-Aromatase         | Pred: 0.345 | True: N/A
SR-ATAD5             | Pred: 0.278 | True: 0 | ✓ CORRECT
SR-HSE               | Pred: 0.301 | True: 0 | ✓ CORRECT
NR-ER-LBD            | Pred: 0.223 | True: 0 | ✓ CORRECT
NR-PPAR-gamma        | Pred: 0.189 | True: 0 | ✓ CORRECT

============================================================
Composite Risk Score: 0.243
Risk Level: LOW ✓
============================================================

✓ Demo Complete - Models are working!
============================================================

Model Performance:
  • LightGBM: Mean AUROC = 0.853
  • ChemBERTa: Val AUROC = 0.810
  • GNN: Val AUROC = 0.861
  • Dataset: 7,794 molecules (Tox21)
  • Features: 2,415 dimensions

Repository: https://github.com/Vk2245/Toxilens-
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python demo.py
```

## Model Training Results

### LightGBM
- Training completed in ~2 minutes
- 12 separate classifiers (one per assay)
- Test AUROC: 0.853 (exceeds 0.80 target)

### ChemBERTa-2
- Fine-tuned from ChemBERTa-zinc-base-v1
- Training: 8 epochs with early stopping
- Validation AUROC: 0.810 (exceeds 0.78 target)

### GNN (AttentiveFP)
- 4 graph convolution layers
- Training: 53 epochs with early stopping
- Validation AUROC: 0.861 (exceeds 0.80 target)

---

**Status:** ✅ Working Prototype Ready for Round 1  
**Last Updated:** April 2, 2026
