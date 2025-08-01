"""
Streamlit Dashboard for Aircraft Modification Certification Support Tool
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import json

# Ollama integration
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    st.warning("Ollama not installed. Install with: pip install ollama")

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from preprocessing import TextPreprocessor, FeatureExtractor
    from regulation_mapper import RegulationMapper
    from similarity_engine import SimilarityEngine
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all utility modules are properly installed.")

# Page configuration
st.set_page_config(
    page_title="Aircraft Mod Certification Tool",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .similarity-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e6e6e6;
        margin-bottom: 1rem;
    }
    .regulation-tag {
        background-color: #e1f5fe;
        color: #01579b;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin: 0.1rem;
        display: inline-block;
    }
    .chatbot-message {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.2rem 0;
        border-left: 3px solid #1f77b4;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 3px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 3px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class ModCertificationApp:
    """Main application class for the mod certification tool"""
    
    def __init__(self):
        self.data_loaded = False
        self.models_loaded = False
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'mod_data' not in st.session_state:
            st.session_state.mod_data = None
        if 'similarity_engine' not in st.session_state:
            st.session_state.similarity_engine = None
        if 'regulation_mapper' not in st.session_state:
            st.session_state.regulation_mapper = None
        if 'preprocessor' not in st.session_state:
            st.session_state.preprocessor = None
        if 'last_analysis_results' not in st.session_state:
            st.session_state.last_analysis_results = None
    
    def load_data(self):
        """Load modification data and initialize models"""
        try:
            # Load modification data
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mods_dataset.csv')
            if os.path.exists(data_path):
                st.session_state.mod_data = pd.read_csv(data_path)
                self.data_loaded = True
                
                # Initialize models
                self.initialize_models()
                
                return True
            else:
                st.error(f"Data file not found: {data_path}")
                return False
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def initialize_models(self):
        """Initialize ML models and engines"""
        try:
            # Initialize preprocessor
            st.session_state.preprocessor = TextPreprocessor()
            
            # Initialize regulation mapper
            reg_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'regulations_db.csv')
            if os.path.exists(reg_path):
                st.session_state.regulation_mapper = RegulationMapper(reg_path)
            
            # Initialize similarity engine
            st.session_state.similarity_engine = SimilarityEngine()
            if st.session_state.mod_data is not None:
                st.session_state.similarity_engine.load_modifications(st.session_state.mod_data)
            
            self.models_loaded = True
            
        except Exception as e:
            st.error(f"Error initializing models: {e}")
            self.models_loaded = False
    
    def predict_mod_type(self, description: str) -> str:
        """Predict modification type using simple rules"""
        description_lower = description.lower()
        
        # Simple rule-based classification
        if any(word in description_lower for word in ['antenna', 'radio', 'navigation', 'radar', 'avionics']):
            return 'Avionics'
        elif any(word in description_lower for word in ['wing', 'fuselage', 'structure', 'frame', 'reinforcement']):
            return 'Structure'
        elif any(word in description_lower for word in ['cabin', 'seat', 'galley', 'passenger', 'lighting']):
            return 'Cabin'
        elif any(word in description_lower for word in ['hydraulic', 'fuel', 'air conditioning', 'oxygen']):
            return 'Systems'
        elif any(word in description_lower for word in ['emergency', 'safety', 'evacuation', 'fire']):
            return 'Safety'
        elif any(word in description_lower for word in ['engine', 'thrust', 'propulsion', 'reverser']):
            return 'Propulsion'
        else:
            return 'Systems'  # Default
    
    def predict_loi(self, mod_type: str, num_regulations: int) -> str:
        """Predict Level of Involvement"""
        # Simple rule-based LOI prediction
        high_impact_types = ['Structure', 'Safety', 'Propulsion']
        
        if mod_type in high_impact_types or num_regulations >= 4:
            return 'High'
        elif num_regulations >= 3:
            return 'Medium'
        else:
            return 'Low'
    
    def get_regulation_explanation(self, regulation_code: str) -> str:
        """Get explanation for a regulation code"""
        explanations = {
            "CS 25.1309": "Equipment, systems, and installations - General requirements for aircraft systems",
            "CS 25.1431": "Electronic equipment - Requirements for electronic systems and their installation",
            "CS 25.561": "General - Structural requirements and emergency landing conditions",
            "CS 25.629": "Aeroelastic stability requirements - Flutter and divergence prevention",
            "CS 25.321": "General loads - Flight load requirements and structural analysis",
            "CS 25.783": "Doors - Design requirements for passenger and cargo doors",
            "CS 25.807": "Emergency exits - Requirements for emergency evacuation systems",
            "CS 25.809": "Emergency exit arrangement - Layout and accessibility requirements",
            "CS 25.812": "Emergency lighting - Requirements for emergency illumination systems",
            "CS 25.853": "Compartment interiors - Fire safety and material requirements",
            "CS 25.857": "Cargo compartments - Design and safety requirements for cargo areas",
            "CS 25.901": "Installation - General requirements for powerplant installation",
            "CS 25.903": "Engines - Requirements for engine design and certification",
            "CS 25.933": "Reversers - Requirements for thrust reverser systems",
            "CS 25.965": "Fuel system hot weather operation - Fuel system performance requirements",
            "CS 25.1441": "Oxygen equipment and supply - Requirements for oxygen systems",
            "AMC 25-12": "Electrical power systems - Acceptable means of compliance for electrical systems",
            "AMC 25-17": "Emergency equipment - Guidance for emergency safety equipment",
            "AMC 20-115": "Software aspects of certification - Requirements for software in airborne systems",
            "AMC 20-130": "Airworthiness and operational approval - Guidance for system approvals",
            "AMC 20-149": "Engine installation - Guidance for powerplant installations",
            "AMC 20-151": "Avionics equipment - Requirements for electronic flight systems",
            "AMC 20-22": "Structural design - Guidance for structural modifications"
        }
        return explanations.get(regulation_code, f"Regulation {regulation_code} - Specific aviation safety requirement")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for LLM analysis"""
        return """You are an expert aviation certification analyst specializing in aircraft modifications and regulatory compliance. 

Your role is to provide detailed, accurate analysis of aircraft modification classifications, regulatory requirements, and certification pathways.

ANALYSIS REQUIREMENTS:
1. Explain the classification rationale clearly and concisely
2. Identify key regulatory considerations and compliance requirements
3. Highlight potential certification challenges or considerations
4. Provide actionable insights for certification planning
5. Reference specific regulations when relevant
6. Consider both technical and procedural aspects

RESPONSE FORMAT:
- Use clear, professional language appropriate for aviation professionals
- Structure your response with logical sections
- Provide specific, actionable recommendations
- Keep explanations focused and relevant to the modification type
- Include confidence indicators when appropriate

Remember: Your analysis will guide critical certification decisions. Be thorough, accurate, and professional."""
    
    def generate_analysis_explanation(self, results: dict) -> str:
        """Generate comprehensive analysis explanation using Llama3 LLM"""
        if not OLLAMA_AVAILABLE:
            return self.generate_fallback_explanation(results)
        
        # Prepare detailed context for the LLM
        context = self.prepare_analysis_context(results)
        
        # Try different model names in order of preference
        model_names = [
            'llama3:latest',    # Exact match for your available model
            'llama3',           # Base model name
            'llama3:8b',        
            'llama2:7b',        
            'mistral:7b'        
        ]
        
        for model_name in model_names:
            try:
                response = ollama.chat(
                    model=model_name,
                    messages=[
                        {
                            'role': 'system',
                            'content': self.get_system_prompt()
                        },
                        {
                            'role': 'user',
                            'content': context
                        }
                    ],
                    options={
                        'temperature': 0.3,  # Lower temperature for more consistent analysis
                        'top_p': 0.9,
                        'num_predict': 800   # Allow longer response for detailed analysis
                    }
                )
                
                return response['message']['content']
                
            except Exception as e:
                error_msg = str(e).lower()
                if "not found" in error_msg or "404" in error_msg:
                    continue
                elif "memory" in error_msg or "system memory" in error_msg:
                    st.warning(f"⚠️ Insufficient memory for {model_name}. Using rule-based analysis.")
                    return self.generate_fallback_explanation(results)
                else:
                    st.error(f"LLM Error with {model_name}: {e}")
                    break
        
        # If all models fail, use fallback
        return self.generate_fallback_explanation(results)
    
    def prepare_analysis_context(self, results: dict) -> str:
        """Prepare detailed context for LLM analysis"""
        mod_type = results.get('predicted_type', 'Unknown')
        loi = results.get('loi', 'Unknown')
        regulations = results.get('regulations', [])
        similar_mods = results.get('similar_modifications', [])
        description = results.get('description', '')
        confidence = results.get('confidence', 0)
        
        context = f"""
AIRCRAFT MODIFICATION ANALYSIS RESULTS

Modification Description: {description}

CLASSIFICATION RESULTS:
- Predicted Type: {mod_type}
- Level of Involvement (LOI): {loi} 
- Classification Confidence: {confidence:.0%}

REGULATORY MAPPING ({len(regulations)} regulations identified):
"""
        
        # Add regulations with explanations
        for i, reg in enumerate(regulations[:8], 1):  # Top 8 regulations
            explanation = self.get_regulation_explanation(reg)
            context += f"{i}. {reg}: {explanation}\n"
        
        if len(regulations) > 8:
            context += f"...and {len(regulations) - 8} additional regulations\n"
        
        # Add similar modifications analysis
        if similar_mods:
            context += f"\nSIMILAR MODIFICATIONS ANALYSIS ({len(similar_mods)} found):\n"
            for i, mod in enumerate(similar_mods[:3], 1):
                similarity = mod.get('similarity_score', 0)
                mod_desc = mod.get('mod_description', 'No description')[:150]
                context += f"{i}. Similarity: {similarity:.1%} - {mod_desc}...\n"
        
        # Add category scores if available
        if 'category_scores' in results:
            context += "\nREGULATORY CATEGORY RELEVANCE:\n"
            for category, score in sorted(results['category_scores'].items(), 
                                        key=lambda x: x[1], reverse=True)[:5]:
                if score > 0:
                    context += f"- {category}: {score:.2f}\n"
        
        context += """

Please provide a comprehensive analysis explanation covering:
1. Classification rationale and implications
2. Regulatory requirements and compliance strategy  
3. Certification approach and timeline considerations
4. Risk assessment and mitigation strategies
5. Recommended next steps and focus areas

Make this analysis professional, actionable, and suitable for certification engineers."""
        
        return context
    
    def generate_fallback_explanation(self, results: dict) -> str:
        """Generate fallback explanation if LLM is unavailable"""
        mod_type = results.get('predicted_type', 'Unknown')
        loi = results.get('loi', 'Unknown')
        num_regs = len(results.get('regulations', []))
        
        explanation = f"""
## 📋 Analysis Explanation

### Classification Summary
This modification has been classified as **{mod_type}** with a **{loi}** Level of Involvement, based on {num_regs} applicable regulations.

### Key Insights
- **Modification Type**: {mod_type} modifications typically require specific attention to system integration and regulatory compliance
- **Complexity Level**: {loi} LOI indicates {'extensive certification requirements' if loi == 'High' else 'moderate certification requirements' if loi == 'Medium' else 'standard certification procedures'}
- **Regulatory Scope**: {num_regs} regulations identified, indicating {'comprehensive regulatory oversight' if num_regs >= 5 else 'standard regulatory requirements'}

### Recommendations
- Consult with certified aviation engineers for detailed certification planning
- Review all applicable regulations for specific compliance requirements
- Consider similar modification precedents for guidance
- Plan appropriate testing and documentation phases

*Note: LLM analysis unavailable. This is a simplified explanation based on classification rules.*
        """
        
        return explanation
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🛫 Aircraft Modification Certification Support Tool</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p><strong>AI-powered assistant for aircraft modification classification and certification support</strong></p>
            <p>Classify modifications • Map regulations • Predict LOI • Find similar modifications</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with controls and information"""
        st.sidebar.markdown("## 🔧 Control Panel")
        
        # Data status
        st.sidebar.markdown("### 📊 Data Status")
        if self.data_loaded:
            st.sidebar.success(f"✅ Data loaded: {len(st.session_state.mod_data)} modifications")
        else:
            st.sidebar.error("❌ Data not loaded")
        
        if self.models_loaded:
            st.sidebar.success("✅ Models initialized")
        else:
            st.sidebar.error("❌ Models not initialized")
        
        # LLM Status
        if OLLAMA_AVAILABLE:
            try:
                # Test Ollama connection and list available models
                models_response = ollama.list()
                
                # Handle different response formats
                if isinstance(models_response, dict) and 'models' in models_response:
                    available_models = [model['name'] for model in models_response['models']]
                elif isinstance(models_response, list):
                    available_models = [model['name'] if isinstance(model, dict) else str(model) for model in models_response]
                else:
                    available_models = []
                
                # Check for preferred models
                preferred_models = ['llama3', 'llama2', 'mistral', 'codellama']
                found_models = []
                
                for model in available_models:
                    model_base = model.lower().split(':')[0]  # Remove version suffix
                    if model_base in preferred_models:
                        found_models.append(model)
                        break
                
                if found_models:
                    st.sidebar.success(f"✅ LLM available: {found_models[0]}")
                    if len(found_models) > 1:
                        st.sidebar.info(f"Other models: {', '.join(found_models[1:3])}")
                else:
                    st.sidebar.warning("⚠️ No suitable models found")
                    if available_models:
                        st.sidebar.info(f"Available: {', '.join(available_models[:3])}")
                    
            except Exception as e:
                error_msg = str(e).lower()
                if "memory" in error_msg or "system memory" in error_msg:
                    st.sidebar.warning("⚠️ LLM detected but insufficient memory")
                    st.sidebar.info("Using rule-based analysis instead")
                else:
                    st.sidebar.error(f"❌ Ollama connection failed: {str(e)[:50]}...")
        else:
            st.sidebar.error("❌ Ollama not installed")
        
        # Settings
        st.sidebar.markdown("### ⚙️ Settings")
        similarity_method = st.sidebar.selectbox(
            "Similarity Method",
            ["SBERT", "TF-IDF"],
            help="Method for finding similar modifications"
        )
        
        similarity_threshold = st.sidebar.slider(
            "Similarity Threshold",
            0.0, 1.0, 0.3, 0.1,
            help="Minimum similarity score for matches"
        )
        
        max_similar_mods = st.sidebar.slider(
            "Max Similar Modifications",
            1, 20, 5,
            help="Maximum number of similar modifications to show"
        )
        
        return similarity_method, similarity_threshold, max_similar_mods
    
    def render_input_section(self):
        """Render modification input section"""
        st.markdown("## 📝 Modification Description")
        
        # Example descriptions
        examples = {
            "VHF Antenna Installation": "Installation of a new VHF antenna on the dorsal fuselage affecting structural and avionics systems for improved communication range in oceanic flights.",
            "LED Cabin Lighting": "Retrofit of LED cabin lighting system replacing existing fluorescent lights to improve passenger comfort and reduce power consumption.",
            "Cargo Door Reinforcement": "Installation of reinforced cargo door frame to increase structural integrity and support heavier cargo loads.",
            "Weather Radar Upgrade": "Integration of advanced weather radar system with enhanced turbulence detection capabilities in nose radome."
        }
        
        # Example selector
        selected_example = st.selectbox(
            "Choose an example or enter your own:",
            ["Custom Input"] + list(examples.keys())
        )
        
        # Text input
        if selected_example == "Custom Input":
            default_text = ""
        else:
            default_text = examples[selected_example]
        
        mod_description = st.text_area(
            "Modification Description:",
            value=default_text,
            height=100,
            placeholder="Enter detailed description of the aircraft modification..."
        )
        
        return mod_description
    
    def render_results_section(self, mod_description: str, similarity_method: str, 
                              similarity_threshold: float, max_similar_mods: int):
        """Render analysis results"""
        if not mod_description.strip():
            st.warning("Please enter a modification description to see analysis results.")
            return
        
        st.markdown("## 🔍 Analysis Results")
        
        # Perform analysis and store results
        analysis_results = self.perform_full_analysis(mod_description, similarity_method, 
                                                     similarity_threshold, max_similar_mods)
        
        # Store results for chatbot
        st.session_state.last_analysis_results = analysis_results
        
        # Create columns for results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self.render_classification_results(mod_description)
        
        with col2:
            self.render_regulation_results(mod_description)
        
        # Similar modifications section
        self.render_similarity_results(mod_description, similarity_method, 
                                     similarity_threshold, max_similar_mods)
        
        # Add LLM-generated explanation
        st.markdown("## 🧠 AI Analysis Explanation")
        with st.spinner("Generating detailed analysis explanation..."):
            explanation = self.generate_analysis_explanation(analysis_results)
            st.markdown(explanation)
    
    def perform_full_analysis(self, mod_description: str, similarity_method: str, 
                             similarity_threshold: float, max_similar_mods: int) -> dict:
        """Perform complete analysis and return structured results"""
        results = {
            'description': mod_description,
            'predicted_type': self.predict_mod_type(mod_description),
            'regulations': [],
            'loi': 'Unknown',
            'similar_modifications': [],
            'confidence': 0.0
        }
        
        # Get regulation predictions
        if st.session_state.regulation_mapper:
            try:
                reg_results = st.session_state.regulation_mapper.predict_regulations(mod_description)
                results['regulations'] = reg_results['recommended_regulations']
                results['category_scores'] = reg_results.get('category_scores', {})
            except Exception as e:
                st.error(f"Regulation mapping error: {e}")
        
        # Calculate LOI
        num_regs = len(results['regulations'])
        results['loi'] = self.predict_loi(results['predicted_type'], num_regs)
        
        # Find similar modifications
        if st.session_state.similarity_engine:
            try:
                method_key = 'sbert' if similarity_method == 'SBERT' else 'tfidf'
                similar_mods = st.session_state.similarity_engine.find_similar(
                    mod_description,
                    method=method_key,
                    top_k=max_similar_mods,
                    min_similarity=similarity_threshold
                )
                results['similar_modifications'] = similar_mods
            except Exception as e:
                st.error(f"Similarity search error: {e}")
        
        # Calculate confidence
        results['confidence'] = 0.85 if results['predicted_type'] != 'Systems' else 0.65
        
        return results
    
    def render_classification_results(self, mod_description: str):
        """Render modification classification results"""
        st.markdown("### 🏷️ Classification Results")
        
        # Predict modification type
        predicted_type = self.predict_mod_type(mod_description)
        
        # Simple confidence calculation
        confidence = 0.85 if predicted_type != 'Systems' else 0.65
        
        # Display results
        st.markdown(f"""
        <div class="metric-card">
            <h4>Predicted Modification Type</h4>
            <h2 style="color: #1f77b4;">{predicted_type}</h2>
            <p>Confidence: {confidence:.0%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Text analysis
        if st.session_state.preprocessor:
            try:
                processed = st.session_state.preprocessor.preprocess_for_ml(mod_description)
                
                st.markdown("#### 📊 Text Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Word Count", processed['features'][0]['word_count'])
                with col2:
                    st.metric("Character Count", processed['features'][0]['char_count'])
                with col3:
                    st.metric("Avg Word Length", f"{processed['features'][0]['avg_word_length']:.1f}")
                
                # Key tokens
                if processed['tokens'][0]:
                    st.markdown("**Key Terms:**")
                    key_terms = processed['tokens'][0][:10]  # First 10 terms
                    st.markdown(", ".join(key_terms))
                    
            except Exception as e:
                st.error(f"Text analysis error: {e}")
    
    def render_regulation_results(self, mod_description: str):
        """Render regulation mapping results"""
        st.markdown("### 📜 Regulation Mapping")
        
        if st.session_state.regulation_mapper:
            try:
                # Get regulation predictions
                reg_results = st.session_state.regulation_mapper.predict_regulations(mod_description)
                
                # Display recommended regulations
                if reg_results['recommended_regulations']:
                    st.markdown("#### Recommended Regulations:")
                    for reg in reg_results['recommended_regulations'][:5]:  # Top 5
                        st.markdown(f"<span class='regulation-tag'>{reg}</span>", 
                                  unsafe_allow_html=True)
                
                # Predict LOI
                predicted_type = self.predict_mod_type(mod_description)
                num_regs = len(reg_results['recommended_regulations'])
                predicted_loi = self.predict_loi(predicted_type, num_regs)
                
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 1rem;">
                    <h4>Level of Involvement (LOI)</h4>
                    <h2 style="color: #1f77b4;">{predicted_loi}</h2>
                    <p>Based on {num_regs} relevant regulations</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Category scores
                if reg_results['category_scores']:
                    st.markdown("#### Category Relevance:")
                    for category, score in sorted(reg_results['category_scores'].items(), 
                                                key=lambda x: x[1], reverse=True)[:3]:
                        if score > 0:
                            st.progress(score, text=f"{category}: {score:.2f}")
                
            except Exception as e:
                st.error(f"Regulation mapping error: {e}")
        else:
            st.warning("Regulation mapper not available")
    
    def render_similarity_results(self, mod_description: str, method: str, 
                                threshold: float, max_results: int):
        """Render similar modifications results"""
        st.markdown("### 🔗 Similar Modifications")
        
        if st.session_state.similarity_engine:
            try:
                # Find similar modifications
                method_key = 'sbert' if method == 'SBERT' else 'tfidf'
                similar_mods = st.session_state.similarity_engine.find_similar(
                    mod_description,
                    method=method_key,
                    top_k=max_results,
                    min_similarity=threshold
                )
                
                if similar_mods:
                    for i, mod in enumerate(similar_mods):
                        with st.expander(f"#{i+1} - {mod['mod_id']} (Similarity: {mod['similarity_score']:.3f})"):
                            st.markdown(f"**Type:** {mod['mod_type']}")
                            st.markdown(f"**Description:** {mod['mod_description']}")
                            st.markdown(f"**Regulations:** {mod['regulations']}")
                            st.markdown(f"**LOI:** {mod['loi']}")
                            
                            # Similarity visualization
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = mod['similarity_score'],
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Similarity Score"},
                                gauge = {
                                    'axis': {'range': [None, 1]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 0.5], 'color': "lightgray"},
                                        {'range': [0.5, 0.8], 'color': "yellow"},
                                        {'range': [0.8, 1], 'color': "green"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 0.9
                                    }
                                }
                            ))
                            fig.update_layout(height=200)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No similar modifications found with similarity > {threshold}")
                    
            except Exception as e:
                st.error(f"Similarity search error: {e}")
        else:
            st.warning("Similarity engine not available")
    
    def render_statistics_section(self):
        """Render dataset statistics"""
        if not self.data_loaded:
            return
        
        st.markdown("## 📈 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Modifications", len(st.session_state.mod_data))
        
        with col2:
            unique_types = st.session_state.mod_data['mod_type'].nunique()
            st.metric("Modification Types", unique_types)
        
        with col3:
            unique_aircraft = st.session_state.mod_data['aircraft_type'].nunique()
            st.metric("Aircraft Types", unique_aircraft)
        
        with col4:
            avg_regs = st.session_state.mod_data['regulations'].str.split(',').str.len().mean()
            st.metric("Avg Regulations", f"{avg_regs:.1f}")
        
        # Charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Modification type distribution
            type_counts = st.session_state.mod_data['mod_type'].value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index, 
                        title="Modification Types Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # LOI distribution
            loi_counts = st.session_state.mod_data['loi'].value_counts()
            fig = px.bar(x=loi_counts.index, y=loi_counts.values,
                        title="Level of Involvement Distribution",
                        labels={'x': 'LOI Level', 'y': 'Count'})
            fig.update_layout(xaxis_title="LOI Level", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Aircraft type distribution (top 10)
            aircraft_counts = st.session_state.mod_data['aircraft_type'].value_counts().head(10)
            fig = px.bar(x=aircraft_counts.values, y=aircraft_counts.index,
                        title="Top 10 Aircraft Types", orientation='h',
                        labels={'x': 'Count', 'y': 'Aircraft Type'})
            st.plotly_chart(fig, use_container_width=True)

    def render_input_section(self):
        """Render modification input section"""
        st.markdown("## 📝 Modification Description")
        
        # Example descriptions (based on actual dataset)
        examples = {
            "VHF Antenna Installation": "Installation of dual VHF antennas on fuselage with improved signal processing capabilities",
            "LED Cabin Lighting": "Installation of new in-flight entertainment screens with enhanced lighting control systems",
            "Navigation Software Upgrade": "Upgrade of navigation database software for enhanced navigation accuracy",
            "ADS-B System Integration": "Integration of ADS-B Out system including electromagnetic interference shielding",
            "Cargo Door Reinforcement": "Installation of reinforced cargo door frame to increase structural integrity and support heavier cargo loads",
            "Weather Radar Upgrade": "Integration of advanced weather radar system with enhanced turbulence detection capabilities in nose radome"
        }
        
        # Example selector
        selected_example = st.selectbox(
            "Choose an example or enter your own:",
            ["Custom Input"] + list(examples.keys())
        )
        
        # Text input
        if selected_example == "Custom Input":
            default_text = ""
        else:
            default_text = examples[selected_example]
        
        mod_description = st.text_area(
            "Modification Description:",
            value=default_text,
            height=100,
            placeholder="Enter detailed description of the aircraft modification..."
        )
        
        return mod_description
    
    def render_results_section(self, mod_description: str, similarity_method: str, 
                              similarity_threshold: float, max_similar_mods: int):
        """Render analysis results"""
        if not mod_description.strip():
            st.warning("Please enter a modification description to see analysis results.")
            return
        
        st.markdown("## 🔍 Analysis Results")
        
        # Perform analysis and store results
        analysis_results = self.perform_full_analysis(mod_description, similarity_method, 
                                                     similarity_threshold, max_similar_mods)
        
        # Store results for chatbot
        st.session_state.last_analysis_results = analysis_results
        
        # Create columns for results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self.render_classification_results(mod_description)
        
        with col2:
            self.render_regulation_results(mod_description)
        
        # Similar modifications section
        self.render_similarity_results(mod_description, similarity_method, 
                                     similarity_threshold, max_similar_mods)
        
        # Add LLM-generated explanation
        st.markdown("## 🧠 AI Analysis Explanation")
        with st.spinner("Generating detailed analysis explanation..."):
            explanation = self.generate_analysis_explanation(analysis_results)
            st.markdown(explanation)
    
    def perform_full_analysis(self, mod_description: str, similarity_method: str, 
                             similarity_threshold: float, max_similar_mods: int) -> dict:
        """Perform complete analysis and return structured results"""
        results = {
            'description': mod_description,
            'predicted_type': self.predict_mod_type(mod_description),
            'regulations': [],
            'loi': 'Unknown',
            'similar_modifications': [],
            'confidence': 0.0
        }
        
        # Get regulation predictions
        if st.session_state.regulation_mapper:
            try:
                reg_results = st.session_state.regulation_mapper.predict_regulations(mod_description)
                results['regulations'] = reg_results['recommended_regulations']
                results['category_scores'] = reg_results.get('category_scores', {})
            except Exception as e:
                st.error(f"Regulation mapping error: {e}")
        
        # Calculate LOI
        num_regs = len(results['regulations'])
        results['loi'] = self.predict_loi(results['predicted_type'], num_regs)
        
        # Find similar modifications
        if st.session_state.similarity_engine:
            try:
                method_key = 'sbert' if similarity_method == 'SBERT' else 'tfidf'
                similar_mods = st.session_state.similarity_engine.find_similar(
                    mod_description,
                    method=method_key,
                    top_k=max_similar_mods,
                    min_similarity=similarity_threshold
                )
                results['similar_modifications'] = similar_mods
            except Exception as e:
                st.error(f"Similarity search error: {e}")
        
        # Calculate confidence
        results['confidence'] = 0.85 if results['predicted_type'] != 'Systems' else 0.65
        
        return results
    
    def render_classification_results(self, mod_description: str):
        """Render modification classification results"""
        st.markdown("### 🏷️ Classification Results")
        
        # Predict modification type
        predicted_type = self.predict_mod_type(mod_description)
        
        # Simple confidence calculation
        confidence = 0.85 if predicted_type != 'Systems' else 0.65
        
        # Display results
        st.markdown(f"""
        <div class="metric-card">
            <h4>Predicted Modification Type</h4>
            <h2 style="color: #1f77b4;">{predicted_type}</h2>
            <p>Confidence: {confidence:.0%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Text analysis
        if st.session_state.preprocessor:
            try:
                processed = st.session_state.preprocessor.preprocess_for_ml(mod_description)
                
                st.markdown("#### 📊 Text Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Word Count", processed['features'][0]['word_count'])
                with col2:
                    st.metric("Character Count", processed['features'][0]['char_count'])
                with col3:
                    st.metric("Avg Word Length", f"{processed['features'][0]['avg_word_length']:.1f}")
                
                # Key tokens
                if processed['tokens'][0]:
                    st.markdown("**Key Terms:**")
                    key_terms = processed['tokens'][0][:10]  # First 10 terms
                    st.markdown(", ".join(key_terms))
                    
            except Exception as e:
                st.error(f"Text analysis error: {e}")
    
    def render_regulation_results(self, mod_description: str):
        """Render regulation mapping results"""
        st.markdown("### 📜 Regulation Mapping")
        
        if st.session_state.regulation_mapper:
            try:
                # Get regulation predictions
                reg_results = st.session_state.regulation_mapper.predict_regulations(mod_description)
                
                # Display recommended regulations
                if reg_results['recommended_regulations']:
                    st.markdown("#### Recommended Regulations:")
                    for reg in reg_results['recommended_regulations'][:5]:  # Top 5
                        st.markdown(f"<span class='regulation-tag'>{reg}</span>", 
                                  unsafe_allow_html=True)
                
                # Predict LOI
                predicted_type = self.predict_mod_type(mod_description)
                num_regs = len(reg_results['recommended_regulations'])
                predicted_loi = self.predict_loi(predicted_type, num_regs)
                
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 1rem;">
                    <h4>Level of Involvement (LOI)</h4>
                    <h2 style="color: #1f77b4;">{predicted_loi}</h2>
                    <p>Based on {num_regs} relevant regulations</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Category scores
                if reg_results['category_scores']:
                    st.markdown("#### Category Relevance:")
                    for category, score in sorted(reg_results['category_scores'].items(), 
                                                key=lambda x: x[1], reverse=True)[:3]:
                        if score > 0:
                            st.progress(score, text=f"{category}: {score:.2f}")
                
            except Exception as e:
                st.error(f"Regulation mapping error: {e}")
        else:
            st.warning("Regulation mapper not available")
    
    def render_similarity_results(self, mod_description: str, method: str, 
                                threshold: float, max_results: int):
        """Render similar modifications results"""
        st.markdown("### 🔗 Similar Modifications")
        
        if st.session_state.similarity_engine:
            try:
                # Find similar modifications
                method_key = 'sbert' if method == 'SBERT' else 'tfidf'
                similar_mods = st.session_state.similarity_engine.find_similar(
                    mod_description,
                    method=method_key,
                    top_k=max_results,
                    min_similarity=threshold
                )
                
                if similar_mods:
                    for i, mod in enumerate(similar_mods):
                        with st.expander(f"#{i+1} - {mod['mod_id']} (Similarity: {mod['similarity_score']:.3f})"):
                            st.markdown(f"**Type:** {mod['mod_type']}")
                            st.markdown(f"**Description:** {mod['mod_description']}")
                            st.markdown(f"**Regulations:** {mod['regulations']}")
                            st.markdown(f"**LOI:** {mod['loi']}")
                            
                            # Similarity visualization
                            fig = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = mod['similarity_score'],
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Similarity Score"},
                                gauge = {
                                    'axis': {'range': [None, 1]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 0.5], 'color': "lightgray"},
                                        {'range': [0.5, 0.8], 'color': "yellow"},
                                        {'range': [0.8, 1], 'color': "green"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 0.9
                                    }
                                }
                            ))
                            fig.update_layout(height=200)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No similar modifications found with similarity > {threshold}")
                    
            except Exception as e:
                st.error(f"Similarity search error: {e}")
        else:
            st.warning("Similarity engine not available")
    
    def render_statistics_section(self):
        """Render dataset statistics"""
        if not self.data_loaded:
            return
        
        st.markdown("## 📈 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Modifications", len(st.session_state.mod_data))
        
        with col2:
            unique_types = st.session_state.mod_data['mod_type'].nunique()
            st.metric("Modification Types", unique_types)
        
        with col3:
            unique_aircraft = st.session_state.mod_data['aircraft_type'].nunique()
            st.metric("Aircraft Types", unique_aircraft)
        
        with col4:
            avg_regs = st.session_state.mod_data['regulations'].str.split(',').str.len().mean()
            st.metric("Avg Regulations", f"{avg_regs:.1f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Modification type distribution
            type_counts = st.session_state.mod_data['mod_type'].value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index, 
                        title="Modification Types Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # LOI distribution
            loi_counts = st.session_state.mod_data['loi'].value_counts()
            fig = px.bar(x=loi_counts.index, y=loi_counts.values,
                        title="Level of Involvement Distribution",
                        labels={'x': 'LOI Level', 'y': 'Count'})
            fig.update_layout(xaxis_title="LOI Level", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Main application runner"""
        self.render_header()
        
        # Load data if not already loaded
        if not self.data_loaded:
            with st.spinner("Loading data and initializing models..."):
                self.load_data()
        
        # Render sidebar
        similarity_method, similarity_threshold, max_similar_mods = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📈 Statistics", "ℹ️ About"])
        
        with tab1:
            # Input section
            mod_description = self.render_input_section()
            
            # Process button
            if st.button("🚀 Analyze Modification", type="primary"):
                if mod_description.strip():
                    with st.spinner("Analyzing modification..."):
                        self.render_results_section(
                            mod_description, 
                            similarity_method, 
                            similarity_threshold, 
                            max_similar_mods
                        )
                else:
                    st.warning("Please enter a modification description.")
        
        with tab2:
            self.render_statistics_section()
        
        with tab3:
            self.render_about_section()
    
    def render_about_section(self):
        """Render about section"""
        st.markdown("## ℹ️ About This Tool")
        
        st.markdown("""
        ### 🎯 Purpose
        This AI-powered tool assists aircraft certification engineers in:
        - **Classifying** aircraft modifications by type
        - **Mapping** modifications to relevant EASA CS-25/AMC regulations
        - **Predicting** Level of Involvement for certification
        - **Finding** similar historical modifications for reference
        
        ### 🧠 Technology Stack
        - **NLP**: NLTK, spaCy, Sentence Transformers
        - **ML**: Scikit-learn, TF-IDF, SBERT embeddings
        - **UI**: Streamlit with interactive visualizations
        - **Similarity**: Cosine similarity and FAISS indexing
        
        ### 📊 Features
        - Real-time text preprocessing and analysis
        - Multi-method similarity search (SBERT + TF-IDF)
        - Regulation pattern recognition
        - Interactive result visualization
        
        ### ⚠️ Disclaimer
        This tool is for assistance only. All certification decisions must be 
        reviewed and approved by qualified aviation authorities and certification engineers.
        
        ### 📞 Support
        For technical support or feature requests, please contact the development team.
        """)

def main():
    """Main function"""
    app = ModCertificationApp()
    app.run()

if __name__ == "__main__":
    main()
