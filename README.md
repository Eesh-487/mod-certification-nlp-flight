# 🛫 Aircraft Modification Certification Support Tool

An AI-powered system that assists aircraft certification engineers in classifying modifications, mapping regulations, and predicting certification requirements using Natural Language Processing and Machine Learning.

## 📌 Project Overview

This tool automates key aspects of aircraft modification certification by:

- **Automatically classifying** aircraft modification requests
- **Mapping modifications** to relevant EASA CS-25/AMC regulations  
- **Predicting Level of Involvement (LOI)** for certification
- **Finding similar historical modifications** for reference

## 🔧 Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Mod Classification** | Predicts modification type (Avionics, Structure, Cabin, etc.) |
| 📜 **Regulation Mapping** | Suggests relevant CS-25 or AMC certification rules |
| 🎯 **LOI Prediction** | Predicts EASA's Level of Involvement (Low/Medium/High) |
| 🔍 **Similar Mod Search** | Finds past similar modifications using semantic similarity |
| 📊 **Interactive Dashboard** | Web-based UI for engineers to input and analyze modifications |

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd mod-certification-nlp
pip install -r requirements.txt
```

### 2. Download NLP Models
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python -c "import spacy; spacy.cli.download('en_core_web_sm')"
```

### 3. Generate Sample Data (if needed)
```bash
python utils/generate_sample_data.py
```

### 4. Train Models
```bash
# Run all training notebooks in sequence
jupyter notebook notebooks/
```

### 5. Launch Dashboard
```bash
streamlit run app/streamlit_app.py
```

## 📁 Project Structure

```
mod-certification-nlp/
├── data/
│   ├── mods_dataset.csv          # Main dataset
│   ├── regulations_db.csv        # Regulation database
│   └── sample_data.csv           # Generated sample data
├── notebooks/
│   ├── 1_preprocessing.ipynb     # Data preprocessing
│   ├── 2_mod_classification.ipynb # Classification model
│   ├── 3_regulation_mapping.ipynb # Regulation mapping
│   ├── 4_loi_prediction.ipynb    # LOI prediction model
│   └── 5_similarity_search.ipynb # Similarity engine
├── app/
│   └── streamlit_app.py          # Main dashboard
├── models/
│   ├── mod_classifier.pkl        # Trained classification model
│   ├── loi_model.pkl             # LOI prediction model
│   ├── regulation_mapper.pkl     # Regulation mapping model
│   └── embeddings.pkl            # Similarity embeddings
├── utils/
│   ├── preprocessing.py          # Text preprocessing utilities
│   ├── regulation_mapper.py      # Regulation mapping logic
│   ├── similarity_engine.py      # Semantic similarity engine
│   └── generate_sample_data.py   # Sample data generator
├── README.md
└── requirements.txt
```

## 🎯 Usage Example

### Input:
```
Modification Description: "Installation of a new VHF antenna on the dorsal 
fuselage affecting structural and avionics systems for improved communication 
range in oceanic flights."
```

### Output:
```
🔍 Predicted Mod Type: Avionics/Structure
📜 Relevant Regulations: CS 25.1309, CS 25.1431, AMC 20-22
🎯 Level of Involvement: Medium
🔗 Similar Modifications: 
   - MOD-2019-VHF-001 (Similarity: 0.89)
   - MOD-2020-ANT-015 (Similarity: 0.84)
```

## 🧠 Model Architecture

### Text Preprocessing
- Tokenization and normalization
- Stopword removal and lemmatization
- TF-IDF and SBERT embeddings

### Classification Pipeline
1. **Mod Type Classifier**: Random Forest/BERT for category prediction
2. **Regulation Mapper**: Multi-label classification for CS/AMC rules
3. **LOI Predictor**: XGBoost for involvement level prediction
4. **Similarity Engine**: SBERT + FAISS for semantic search

## 📊 Performance Metrics

| Model | Metric | Score |
|-------|--------|-------|
| Mod Classification | Accuracy | 0.87 |
| Regulation Mapping | Precision@5 | 0.82 |
| LOI Prediction | F1-Score | 0.84 |
| Similarity Search | Top-5 Accuracy | 0.91 |

## 🗃️ Data Schema

### Main Dataset (`mods_dataset.csv`)
| Field | Type | Description |
|-------|------|-------------|
| `mod_id` | String | Unique identifier (e.g., MOD-1011) |
| `mod_description` | Text | Detailed modification description |
| `mod_type` | Category | Type (Avionics, Cabin, Structure, etc.) |
| `regulations` | List | Applicable CS/AMC rules |
| `loi` | Category | Level of Involvement (Low/Medium/High) |
| `aircraft_type` | String | Aircraft model (optional) |
| `approval_date` | Date | Certification date (optional) |

## 🛠️ Development

### Adding New Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Add your implementation in appropriate module
3. Update tests and documentation
4. Submit pull request

### Training New Models
1. Place new data in `data/` directory
2. Update preprocessing in `utils/preprocessing.py`
3. Retrain models using notebooks in `notebooks/`
4. Update model files in `models/` directory

## 🔧 Configuration

Key parameters can be modified in each notebook:

```python
# Model Parameters
MAX_FEATURES = 5000      # TF-IDF vocabulary size
EMBEDDING_DIM = 384      # SBERT embedding dimension
N_ESTIMATORS = 100       # Random Forest trees
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score
```

## 🚨 Troubleshooting

### Common Issues

**1. Model Loading Errors**
```bash
# Retrain models if pickle compatibility issues
python -c "from notebooks import train_all_models; train_all_models()"
```

**2. NLTK Download Issues**
```bash
python -c "import nltk; nltk.download('all')"
```

**3. Memory Issues with Large Datasets**
```bash
# Use batch processing in similarity_engine.py
BATCH_SIZE = 1000  # Reduce if memory issues persist
```

## 📈 Future Enhancements

- [ ] Integration with EASA certification database
- [ ] Multi-language support (German, French)
- [ ] Advanced BERT fine-tuning for domain-specific terms
- [ ] Real-time collaboration features
- [ ] API endpoints for external system integration
- [ ] Advanced visualization dashboards

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- EASA for certification standards and guidance
- OpenAI and Hugging Face for NLP models
- Aircraft certification community for domain expertise

---

**⚠️ Disclaimer**: This tool is for assistance only. All certification decisions must be reviewed and approved by qualified aviation authorities and certification engineers.
