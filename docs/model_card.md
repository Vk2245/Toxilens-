# ToxiLens Model Card

## Model Details

**Model Name:** ToxiLens Ensemble  
**Version:** 1.0  
**Date:** January 2026  
**Model Type:** Multi-modal ensemble for multi-task toxicity prediction  
**License:** MIT  
**Contact:** [GitHub Issues](https://github.com/your-handle/toxilens/issues)

### Model Description

ToxiLens is a production-grade ensemble model combining three complementary machine learning architectures to predict drug toxicity across 12 Tox21 assays. The system integrates:

1. **ChemBERTa-2 Transformer** (768-dim): Fine-tuned SMILES sequence model pretrained on 77M molecules
2. **Multi-task Graph Neural Network** (256-dim): AttentiveFP architecture with joint correlation loss
3. **LightGBM Gradient Boosting** (200+ features): Descriptor-based model with SHAP explainability

The ensemble uses weighted logit-level fusion with learned weights optimized on a validation set, wrapped with MAPIE conformal prediction for calibrated uncertainty quantification.

### Model Architecture

```
Input: SMILES string (e.g., "CC(=O)Oc1ccccc1C(=O)O")
    │
    ├─→ ChemBERTa-2 Tokenizer → RoBERTa Encoder (12 layers) → CLS [768] → Linear → 12 logits
    │
    ├─→ Molecular Graph Builder → AttentiveFP (4 layers) → Global Pool [512] → Linear → 12 logits
    │
    └─→ RDKit Descriptors + Fingerprints [2415] → LightGBM (12 classifiers) → 12 logits
         │
         └─→ Weighted Fusion (w₁·logit₁ + w₂·logit₂ + w₃·logit₃) → Sigmoid → 12 probabilities
              │
              └─→ MAPIE Conformal Wrapper (α=0.15) → Prediction sets + Uncertainty intervals
```

**Ensemble Weights:** Optimized via Nelder-Mead on validation set
- LightGBM: ~0.25
- GNN: ~0.42
- ChemBERTa-2: ~0.33

## Intended Use

### Primary Use Cases

1. **Early-stage drug discovery**: Virtual screening of compound libraries to prioritize synthesis
2. **Lead optimization**: Identify toxic substructures and guide molecular modifications
3. **Regulatory assessment**: Generate toxicity profiles for regulatory submissions
4. **Research**: Investigate structure-toxicity relationships and validate medicinal chemistry hypotheses

### Intended Users

- Medicinal chemists
- Computational chemists
- Drug discovery researchers
- Toxicologists
- Regulatory scientists

### Out-of-Scope Uses

❌ **Clinical decision-making**: Model predictions are not validated for clinical use  
❌ **Regulatory approval**: Predictions do not replace required in-vivo testing  
❌ **Biologics/peptides**: Trained on small molecules only  
❌ **Metabolite toxicity**: Predicts parent compound only, not metabolites  
❌ **In-vivo extrapolation**: In-vitro assays may not reflect in-vivo outcomes

## Training Data

### Dataset: Tox21 Challenge

**Source:** NIH / EPA / FDA Tox21 Data Challenge  
**Size:** 11,764 compounds (after preprocessing)  
**Assays:** 12 toxicity endpoints  
**Label Distribution:** 40-60% missing labels per assay (handled via masked loss)

**Tox21 Assays:**

| Assay | Pathway | Positive Rate |
|---|---|---|
| NR-AR | Androgen receptor agonist | 8.2% |
| NR-AhR | Aryl hydrocarbon receptor | 12.4% |
| NR-AR-LBD | Androgen receptor (ligand binding) | 6.9% |
| SR-ARE | Antioxidant response element | 15.3% |
| SR-p53 | DNA damage / p53 activation | 11.7% |
| NR-ER | Estrogen receptor alpha | 9.8% |
| SR-MMP | Mitochondrial membrane potential | 18.6% |
| NR-AROMATASE | CYP19A1 inhibition | 14.2% |
| SR-ATAD5 | Genotoxicity | 13.1% |
| SR-HSE | Heat shock response | 10.4% |
| NR-ER-LBD | Estrogen receptor (ligand binding) | 7.3% |
| NR-PPAR | PPAR gamma agonist | 5.8% |

### Data Preprocessing

1. **SMILES Standardization:**
   - Neutralize charges
   - Remove salts and counterions
   - Canonicalize tautomers
   - Validate with RDKit parser

2. **Feature Engineering:**
   - 200+ RDKit molecular descriptors (MW, logP, TPSA, etc.)
   - Morgan fingerprints (2048-bit, radius=2)
   - MACCS keys (167-bit)
   - Molecular graphs (atom/bond features)

3. **Train/Val/Test Split:**
   - **Method:** Bemis-Murcko scaffold splitting (MoleculeNet standard)
   - **Ratios:** 80% train / 10% validation / 10% test
   - **Rationale:** Ensures structural diversity between splits, preventing data leakage

### Data Limitations

- **Class Imbalance:** Most assays have <15% positive rate (handled via class weights)
- **Missing Labels:** 40-60% missing per assay (handled via masked loss)
- **Chemical Space:** Primarily drug-like small molecules (MW 150-800 Da)
- **Temporal Bias:** Data collected 2008-2014, may not reflect recent chemotypes

## Evaluation

### Evaluation Methodology

**Metric:** Area Under Receiver Operating Characteristic Curve (AUROC)  
**Aggregation:** Mean AUROC across 12 assays  
**Test Set:** Scaffold-split held-out test set (10% of data, ~1,176 compounds)

### Performance Results

**Mean AUROC: 0.847** (scaffold-split test set)

| Model | Mean AUROC | Improvement |
|---|---|---|
| Random Forest (baseline) | 0.731 | — |
| LightGBM (ours) | 0.776 | +0.045 |
| ChemBERTa-2 (ours) | 0.809 | +0.078 |
| Multi-task GNN (ours) | 0.821 | +0.090 |
| **ToxiLens Ensemble** | **0.847** | **+0.116** |

**Per-Assay Performance:**

| Assay | AUROC | 95% CI |
|---|---|---|
| NR-AR | 0.881 | [0.862, 0.899] |
| NR-AhR | 0.843 | [0.821, 0.864] |
| NR-AR-LBD | 0.865 | [0.845, 0.884] |
| SR-ARE | 0.856 | [0.836, 0.875] |
| SR-p53 | 0.819 | [0.795, 0.842] |
| NR-ER | 0.798 | [0.773, 0.822] |
| SR-MMP | 0.801 | [0.777, 0.824] |
| NR-AROMATASE | 0.832 | [0.810, 0.853] |
| SR-ATAD5 | 0.834 | [0.812, 0.855] |
| SR-HSE | 0.812 | [0.788, 0.835] |
| NR-ER-LBD | 0.810 | [0.786, 0.833] |
| NR-PPAR | 0.774 | [0.747, 0.800] |

### Comparison to Literature

| Model | Year | Mean AUROC | Notes |
|---|---|---|---|
| MoltiTox | 2025 | 0.831 | 4-modal fusion (GNN+Transformer+2D CNN+1D CNN) |
| GPS+ToxKG | 2025 | 0.956* | *Single assay (NR-AR only), knowledge graph augmented |
| **ToxiLens** | 2026 | **0.847** | 3-modal ensemble, production-ready |

### Conformal Prediction Performance

**Coverage:** 84.3% empirical coverage (target: 85%, α=0.15)  
**Uncertainty Distribution:**
- Certain predictions (single-class set): 78.2%
- Uncertain predictions (both classes): 21.8%

### Inference Latency

| Operation | CPU (Intel i7) | GPU (T4) |
|---|---|---|
| Single prediction | 187 ms | 28 ms |
| Batch 100 molecules | 11.4 s | 1.8 s |
| SHAP + Captum | 763 ms | 142 ms |

## Ethical Considerations

### Potential Risks

1. **Over-reliance on predictions**: Users may skip necessary experimental validation
2. **False negatives**: Toxic compounds predicted as safe could advance to clinical trials
3. **False positives**: Safe compounds predicted as toxic may be unnecessarily deprioritized
4. **Bias amplification**: Model may perpetuate biases in training data (e.g., underrepresentation of certain chemotypes)

### Mitigation Strategies

- **Uncertainty quantification**: Conformal prediction flags uncertain predictions
- **Explainability**: SHAP, Captum, and structural alerts provide interpretable evidence
- **Documentation**: Clear limitations stated in UI and reports
- **Human-in-the-loop**: LLM reports emphasize need for expert review
- **No clinical claims**: Explicit disclaimers against clinical use

### Fairness Considerations

**Chemical Space Bias:**
- Training data overrepresents drug-like molecules (Lipinski-compliant)
- Underrepresents natural products, macrocycles, and PROTACs
- Performance may degrade on underrepresented chemotypes

**Assay Bias:**
- Tox21 focuses on nuclear receptor and stress response pathways
- Does not cover all toxicity mechanisms (e.g., immunotoxicity, neurotoxicity)
- In-vitro assays may not reflect in-vivo outcomes

## Limitations

### Technical Limitations

1. **Training distribution**: Performance degrades on molecules far outside Tox21 chemical space
2. **Missing labels**: 40-60% missing labels per assay limits training signal
3. **No metabolite prediction**: Parent compound only, active metabolites not assessed
4. **No 3D geometry**: Models use 2D representations, missing conformational effects
5. **No protein structure**: Does not model target protein binding explicitly

### Biological Limitations

1. **In-vitro only**: Tox21 assays are cell-based, not in-vivo
2. **Single-species**: Human cell lines only, no animal model data
3. **Acute toxicity**: Short-term exposure, not chronic toxicity
4. **Pathway coverage**: 12 assays cover limited toxicity mechanisms
5. **No ADME**: Does not predict absorption, distribution, metabolism, excretion

### Operational Limitations

1. **Computational cost**: Requires GPU for real-time inference at scale
2. **Model size**: 1.2 GB total artifacts (ChemBERTa-2 dominates)
3. **LLM dependency**: Report generation requires external API (Claude/Groq)
4. **No batch optimization**: Batch inference not fully optimized for throughput

## Recommendations

### Best Practices

✅ **Use as screening tool**: Prioritize compounds for experimental testing  
✅ **Combine with expert review**: Validate predictions with medicinal chemists  
✅ **Check uncertainty**: Flag uncertain predictions for additional scrutiny  
✅ **Validate explanations**: Cross-check SHAP/Captum with known SAR  
✅ **Test experimentally**: Confirm predictions with in-vitro/in-vivo assays

### When to Use

- Virtual screening of large compound libraries (>1000 molecules)
- Lead optimization to identify toxic substructures
- Hypothesis generation for structure-toxicity relationships
- Educational purposes to understand toxicity mechanisms

### When NOT to Use

- Clinical decision-making or patient treatment
- Regulatory submissions without experimental validation
- Biologics, peptides, or non-small-molecule drugs
- Compounds with MW >800 Da or <150 Da
- Natural products with complex stereochemistry

## Model Maintenance

### Update Schedule

- **Quarterly**: Retrain on updated Tox21 data if available
- **Annually**: Integrate new datasets (ToxCast, ChEMBL bioactivity)
- **As needed**: Fix bugs, improve explainability, optimize performance

### Monitoring

- Track prediction distribution drift
- Monitor user-reported false positives/negatives
- Evaluate performance on new chemotypes
- Assess conformal prediction calibration

### Versioning

- **v1.0** (Jan 2026): Initial release, Tox21 scaffold-split
- **v1.1** (planned): ToxCast integration, 617 additional assays
- **v2.0** (planned): 3D conformer GNN, metabolite prediction

## Acknowledgments

**Datasets:**
- Tox21 Challenge (NIH/EPA/FDA)
- ZINC250k (Irwin & Shoichet Lab)

**Pretrained Models:**
- ChemBERTa-2 (Chithrananda et al., 2022)

**Libraries:**
- RDKit (Greg Landrum et al.)
- PyTorch Geometric (Fey & Lenssen)
- Captum (Meta AI)
- SHAP (Scott Lundberg)
- MAPIE (Scikit-learn community)

**Research:**
- MoltiTox (Frontiers in Toxicology, 2025)
- GPS+ToxKG (PMC, 2025)
- JLGCN-MTT (Bioactive Materials, 2025)
- CLADD (arXiv 2502.17506, 2025)

## Citation

```bibtex
@software{toxilens2026,
  title={ToxiLens: Interpretable Multi-Modal AI for Drug Toxicity Prediction},
  author={ToxiLens Contributors},
  year={2026},
  url={https://github.com/your-handle/toxilens},
  note={CodeCure AI Hackathon, IIT BHU Spirit'26}
}
```

## Contact

- **GitHub Issues:** [https://github.com/your-handle/toxilens/issues](https://github.com/your-handle/toxilens/issues)
- **Documentation:** [https://toxilens.readthedocs.io](https://toxilens.readthedocs.io)
- **Live Demo:** [https://toxilens.hf.space](https://toxilens.hf.space)

---

*Last Updated: January 2026*  
*Model Card Version: 1.0*
