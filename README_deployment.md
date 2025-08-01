# Aircraft Modification Certification Support Tool

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🚀 Live Demo

**[Launch Application →](https://your-app-url.streamlit.app)**

## 📋 Overview

This Aircraft Modification Certification Support Tool is an AI-powered assistant designed to help aviation engineers and certification specialists with aircraft modification classification, regulatory mapping, and certification planning.

### ✨ Key Features

- **🔍 Intelligent Classification**: Automatic modification type prediction
- **📋 Regulatory Mapping**: CS-25 and AMC regulation identification
- **⚖️ LOI Assessment**: EASA Level of Involvement prediction
- **🔍 Similarity Search**: Historical modification database
- **📊 Analytics Dashboard**: Comprehensive data insights

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly, Seaborn
- **ML/NLP**: Scikit-learn, Sentence Transformers
- **Deployment**: Streamlit Cloud

## 📦 Installation

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mod-certification-nlp.git
cd mod-certification-nlp
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements_streamlit.txt
```

4. **Run the application**
```bash
streamlit run streamlit_cloud_app.py
```

### Streamlit Cloud Deployment

1. **Fork this repository**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub account**
4. **Deploy with these settings**:
   - Repository: `yourusername/mod-certification-nlp`
   - Branch: `main`
   - Main file path: `streamlit_cloud_app.py`
   - Requirements file: `requirements_streamlit.txt`

## 📊 Data Structure

The application expects a CSV file with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `mod_id` | Unique modification ID | MOD-001 |
| `mod_description` | Detailed description | "Installation of VHF antenna..." |
| `mod_type` | Modification category | Avionics, Structure, Cabin, etc. |
| `regulations` | Applicable regulations | CS 25.1309,AMC 20-151 |
| `loi` | Level of Involvement | Low, Medium, High |
| `aircraft_type` | Aircraft model (optional) | A320, B737, etc. |

## 🎯 Usage Examples

### Classification Analysis
```python
# Example modification description
description = "Installation of enhanced weather radar system with storm detection capabilities"

# Expected results
- Type: Avionics
- LOI: Medium
- Regulations: CS 25.1309, CS 25.1431, AMC 20-151
```

### Similarity Search
Find historical modifications similar to your current project for reference and best practices.

### Regulatory Mapping
Automatic identification of applicable CS-25 and AMC regulations based on modification type and description.

## 📁 Project Structure

```
mod-certification-nlp/
├── streamlit_cloud_app.py      # Main cloud-optimized application
├── requirements_streamlit.txt   # Cloud dependencies
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── data/
│   ├── mods_dataset.csv        # Main dataset
│   └── regulations_db.csv      # Regulations database
├── utils/
│   ├── preprocessing.py        # Text preprocessing
│   ├── regulation_mapper.py    # Regulation mapping
│   └── similarity_engine.py    # Similarity calculations
├── app/
│   └── streamlit_app.py        # Local version with LLM
└── README_deployment.md        # This file
```

## 🔧 Configuration

### Environment Variables

For production deployment, consider setting:

```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ENABLECORS=false
STREAMLIT_THEME_PRIMARYCOLOR="#FF6B6B"
```

### Streamlit Secrets

For sensitive configurations, use Streamlit secrets:

```toml
# .streamlit/secrets.toml
[database]
host = "your-db-host"
username = "your-username"
password = "your-password"
```

## 📊 Performance Considerations

### Cloud Optimizations

- **Memory Management**: Efficient data loading and caching
- **Startup Time**: Lazy loading of heavy dependencies
- **Resource Usage**: Optimized for Streamlit Cloud limits
- **Error Handling**: Graceful degradation for missing features

### Scaling Recommendations

- **Data Size**: Optimize for datasets up to 10MB
- **Concurrent Users**: Designed for moderate concurrent usage
- **Response Time**: Target response time under 3 seconds

## 🛡️ Security & Compliance

### Data Privacy
- No sensitive data stored permanently
- Session-based processing only
- GDPR compliant data handling

### Regulatory Compliance
- Tool provides guidance only
- Always consult certified authorities
- Regular updates with regulatory changes

## 🚨 Limitations

### Cloud Deployment Limitations
- ❌ No LLM/Ollama support (memory constraints)
- ❌ Limited to rule-based analysis
- ❌ No persistent data storage
- ❌ 1GB memory limit

### Workarounds
- ✅ Comprehensive rule-based classification
- ✅ Efficient similarity algorithms
- ✅ Cached regulation mappings
- ✅ Optimized data structures

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
pip install -r requirements_streamlit.txt
```

**2. Data Loading Issues**
- Ensure `data/mods_dataset.csv` exists
- Check file permissions
- Verify CSV format

**3. Memory Issues**
```python
# Reduce dataset size if needed
df = df.sample(n=1000)  # Use subset for testing
```

**4. Deployment Failures**
- Check requirements.txt syntax
- Verify file paths are relative
- Ensure no local-only dependencies

### Performance Optimization

```python
# Enable caching for expensive operations
@st.cache_data
def load_data():
    return pd.read_csv("data/mods_dataset.csv")

# Use session state for persistence
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
```

## 📈 Future Enhancements

### Planned Features
- [ ] Enhanced ML models for classification
- [ ] Real-time regulation updates
- [ ] Multi-language support
- [ ] Export functionality
- [ ] API integration

### Technical Improvements
- [ ] Database integration
- [ ] Advanced caching
- [ ] Performance monitoring
- [ ] Automated testing

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

### Getting Help
- **Documentation**: Check this README and inline help
- **Issues**: Open GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Email**: contact@yourcompany.com

### Status Page
Check the application status at: `https://your-app-url.streamlit.app/health`

## 🙏 Acknowledgments

- **EASA**: For regulatory framework and guidance
- **Streamlit**: For the amazing deployment platform
- **Open Source Community**: For the excellent Python libraries

---

**⚠️ Disclaimer**: This tool is for guidance only. Always consult with certified aviation authorities and follow official certification procedures.

**🚀 Ready to deploy?** Follow the [Streamlit Cloud deployment guide](#streamlit-cloud-deployment) above!
