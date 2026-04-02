# ToxiLens Documentation Index

Welcome to the ToxiLens documentation! This index helps you navigate all available documentation resources.

## 📚 Core Documentation

### [README.md](../README.md)
**Main project documentation** - Start here!

Covers:
- Project overview and problem statement
- Feature overview and capabilities
- Architecture diagrams
- ML pipeline details
- Installation and setup instructions
- API reference with curl examples
- UI walkthrough
- Performance metrics
- Troubleshooting guide
- Contributing guidelines
- License and citations

### [Model Card](model_card.md)
**Comprehensive model documentation**

Covers:
- Model architecture and design
- Training data and preprocessing
- Performance metrics and benchmarks
- Intended use cases and limitations
- Ethical considerations
- Maintenance and versioning

### [Architecture Diagrams](architecture_diagrams.md)
**Visual system architecture**

Includes:
- System architecture overview
- Data flow diagrams
- ML pipeline architecture
- Inference pipeline
- Explainability pipeline
- Component interactions
- Deployment architecture
- Model architecture details

## 🧪 Examples and Tutorials

### [Examples Directory](../examples/)
**Hands-on examples and sample data**

Contains:
- `example_molecules.json` - 10 curated drug molecules with known toxicity profiles
- `batch_screening_example.csv` - 25 compounds for batch screening demos
- `api_usage_examples.ipynb` - Comprehensive Jupyter notebook with API examples
- `README.md` - Guide to using the examples

### [API Usage Notebook](../examples/api_usage_examples.ipynb)
**Interactive tutorial**

Demonstrates:
1. Health check and connectivity
2. Single molecule prediction
3. Batch virtual screening
4. Chemical space exploration
5. What-if analysis
6. De-risking lab
7. LLM report generation
8. Multi-molecule comparison
9. Error handling

## 🔧 Technical Documentation

### [Design Document](../.kiro/specs/toxilens-platform/design.md)
**Detailed technical design**

Covers:
- System architecture
- Component interfaces
- Data structures
- Algorithm details
- Technology choices

### [Requirements Document](../.kiro/specs/toxilens-platform/requirements.md)
**Functional and non-functional requirements**

Includes:
- 35 detailed requirements
- Acceptance criteria
- Property-based testing specifications
- Performance targets

### [Tasks Document](../.kiro/specs/toxilens-platform/tasks.md)
**Implementation task breakdown**

Contains:
- Task hierarchy
- Implementation status
- Dependencies
- Completion tracking

## 🚀 Quick Start Guides

### Installation

```bash
# Clone repository
git clone https://github.com/your-handle/toxilens.git
cd toxilens

# Install dependencies
conda create -n toxilens python=3.11 -y
conda activate toxilens
conda install -c conda-forge rdkit -y
pip install -r requirements.txt

# Start with Docker (recommended)
docker-compose up --build
```

### First Prediction

```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    json={'smiles': 'CC(=O)Oc1ccccc1C(=O)O'}  # Aspirin
)

data = response.json()
print(f"Risk Level: {data['risk_level']}")
print(f"Composite Risk: {data['composite_risk']:.3f}")
```

## 📖 API Documentation

### Interactive API Docs
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Key Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Single molecule prediction |
| `/predict_batch` | POST | Batch virtual screening |
| `/generate_report` | POST | LLM assessment report |
| `/derisk` | POST | Generate de-risked variants |
| `/what_if` | POST | Compare molecular modifications |
| `/similar` | GET | Find similar molecules |
| `/health` | GET | Service health check |

## 🎓 Learning Resources

### For New Users
1. Read [README.md](../README.md) - Overview and features
2. Try [example_molecules.json](../examples/example_molecules.json) - Test with known compounds
3. Follow [api_usage_examples.ipynb](../examples/api_usage_examples.ipynb) - Interactive tutorial

### For Developers
1. Review [design.md](../.kiro/specs/toxilens-platform/design.md) - System architecture
2. Study [architecture_diagrams.md](architecture_diagrams.md) - Visual architecture
3. Check [requirements.md](../.kiro/specs/toxilens-platform/requirements.md) - Specifications

### For Researchers
1. Read [model_card.md](model_card.md) - Model details and performance
2. Review training scripts in `ml/scripts/` - Implementation details
3. Check citations in [README.md](../README.md) - Research background

## 🐛 Troubleshooting

### Common Issues

**Installation Problems**
- See [README.md - Troubleshooting](../README.md#-troubleshooting)

**Runtime Errors**
- Check [README.md - Troubleshooting](../README.md#-troubleshooting)

**API Issues**
- Verify backend is running: `curl http://localhost:8000/health`
- Check logs: `docker-compose logs backend`

**Performance Issues**
- See [model_card.md - Limitations](model_card.md#limitations)

## 🤝 Contributing

### How to Contribute
1. Read [README.md - Contributing](../README.md#-contributing)
2. Fork the repository
3. Create a feature branch
4. Submit a pull request

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Format code
black backend/
isort backend/
```

## 📊 Performance Benchmarks

### Model Performance
- **Mean AUROC:** 0.847 across 12 Tox21 assays
- **Best Assay:** NR-AR (0.881 AUROC)
- **Inference Latency:** <200ms CPU, <30ms GPU

See [model_card.md - Performance](model_card.md#evaluation) for details.

### System Performance
- **Single prediction:** <200ms (CPU)
- **Batch 100 molecules:** <12s (CPU)
- **SHAP + Captum:** <800ms (CPU)

See [README.md - Performance](../README.md#-model-performance) for details.

## 🔬 Research Background

### Key Papers
- **MoltiTox** (2025) - 4-modal fusion, 0.831 AUROC
- **GPS+ToxKG** (2025) - Knowledge graph GNN, 0.956 AUC on NR-AR
- **JLGCN-MTT** (2025) - Joint correlation loss for multi-task
- **ChemBERTa-2** (2022) - Pretrained on 77M SMILES
- **CLADD** (2025) - RAG-powered multi-agent LLM

See [README.md - Research Background](../README.md#-research-background) for full list.

## 📝 License

ToxiLens is released under the **MIT License**.

See [LICENSE](../LICENSE) for full text.

## 📧 Contact and Support

- **GitHub Issues:** [Report bugs or request features](https://github.com/your-handle/toxilens/issues)
- **Live Demo:** [https://toxilens.hf.space](https://toxilens.hf.space)
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

## 🗺️ Documentation Roadmap

### Completed ✅
- [x] Main README with comprehensive overview
- [x] Model card with performance metrics
- [x] Architecture diagrams
- [x] Example molecules and CSV
- [x] API usage Jupyter notebook
- [x] Troubleshooting guide
- [x] Contributing guidelines
- [x] License information
- [x] Citations and references

### Planned 📋
- [ ] Video tutorials
- [ ] Advanced use case guides
- [ ] Deployment guides (AWS, GCP, Azure)
- [ ] Model retraining guide
- [ ] Custom dataset integration guide
- [ ] API client libraries (Python, R, JavaScript)

## 📚 External Resources

### Datasets
- [Tox21 Challenge](https://tripod.nih.gov/tox21/)
- [ZINC Database](https://zinc.docking.org/)
- [ChEMBL](https://www.ebi.ac.uk/chembl/)

### Tools and Libraries
- [RDKit Documentation](https://www.rdkit.org/docs/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Captum](https://captum.ai/)
- [SHAP](https://shap.readthedocs.io/)

### Related Projects
- [DeepChem](https://deepchem.io/)
- [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)
- [MoleculeNet](http://moleculenet.org/)

---

*Last Updated: January 2026*  
*Documentation Version: 1.0*

**Need help?** Start with the [README.md](../README.md) or open an [issue](https://github.com/your-handle/toxilens/issues)!
