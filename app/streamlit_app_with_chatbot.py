"""
Enhanced Streamlit Dashboard with Chatbot for Aircraft Modification Certification Support Tool
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

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
        font-size: 0.9rem;
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

class SimplifiedCertificationApp:
    """Simplified application with chatbot functionality"""
    
    def __init__(self):
        self.initialize_session_state()
        self.load_sample_data()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'chatbot_messages' not in st.session_state:
            st.session_state.chatbot_messages = []
        if 'last_analysis_results' not in st.session_state:
            st.session_state.last_analysis_results = None
        if 'mod_data' not in st.session_state:
            st.session_state.mod_data = None
    
    def load_sample_data(self):
        """Load sample modification data"""
        try:
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_mods_extended.csv')
            if os.path.exists(data_path):
                st.session_state.mod_data = pd.read_csv(data_path)
            else:
                # Create sample data if file doesn't exist
                sample_data = {
                    'mod_id': ['MOD-2024-001', 'MOD-2024-002', 'MOD-2024-003'],
                    'mod_description': [
                        'Installation of VHF antenna in cockpit for communication enhancement',
                        'Addition of LED cabin lighting for passenger comfort',
                        'Retrofit of emergency slide in cargo compartment for safety'
                    ],
                    'mod_type': ['Avionics', 'Cabin', 'Safety'],
                    'regulations': ['CS 25.1431,AMC 20-130', 'CS 25.812,CS 25.785', 'CS 25.807,CS 25.809'],
                    'loi': ['Medium', 'Low', 'High']
                }
                st.session_state.mod_data = pd.DataFrame(sample_data)
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    def predict_mod_type(self, description: str) -> str:
        """Simple rule-based modification type prediction"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['antenna', 'radio', 'navigation', 'radar', 'avionics', 'gps', 'communication']):
            return 'Avionics'
        elif any(word in description_lower for word in ['wing', 'fuselage', 'structure', 'frame', 'reinforcement', 'door']):
            return 'Structure'
        elif any(word in description_lower for word in ['cabin', 'seat', 'galley', 'passenger', 'lighting', 'lavatory']):
            return 'Cabin'
        elif any(word in description_lower for word in ['hydraulic', 'fuel', 'air conditioning', 'oxygen', 'pressure']):
            return 'Systems'
        elif any(word in description_lower for word in ['emergency', 'safety', 'evacuation', 'fire', 'smoke', 'exit']):
            return 'Safety'
        elif any(word in description_lower for word in ['engine', 'thrust', 'propulsion', 'reverser', 'compressor']):
            return 'Propulsion'
        else:
            return 'Systems'
    
    def predict_loi(self, mod_type: str, description: str) -> str:
        """Predict Level of Involvement"""
        description_lower = description.lower()
        high_impact_types = ['Structure', 'Safety', 'Propulsion']
        
        # Check for high-impact keywords
        high_impact_words = ['structural integrity', 'safety critical', 'flight critical', 'major modification']
        
        if (mod_type in high_impact_types or 
            any(word in description_lower for word in high_impact_words) or
            len(description.split()) > 20):
            return 'High'
        elif len(description.split()) > 10:
            return 'Medium'
        else:
            return 'Low'
    
    def get_relevant_regulations(self, mod_type: str) -> list:
        """Get relevant regulations based on modification type"""
        regulations_map = {
            'Avionics': ['CS 25.1309', 'CS 25.1431', 'AMC 20-115', 'AMC 20-130', 'AMC 20-151'],
            'Structure': ['CS 25.561', 'CS 25.629', 'CS 25.321', 'CS 25.783', 'CS 25.143'],
            'Cabin': ['CS 25.812', 'CS 25.785', 'CS 25.853', 'CS 25.857', 'CS 25.773'],
            'Systems': ['CS 25.831', 'CS 25.965', 'CS 25.1441', 'CS 25.729', 'CS 25.841'],
            'Safety': ['CS 25.807', 'CS 25.809', 'CS 25.562', 'CS 25.1457', 'AMC 25-17'],
            'Propulsion': ['CS 25.901', 'CS 25.903', 'CS 25.933', 'CS 25.934', 'AMC 20-149']
        }
        return regulations_map.get(mod_type, ['CS 25.1309'])
    
    def find_similar_modifications(self, description: str, mod_type: str) -> list:
        """Find similar modifications using simple text matching"""
        if st.session_state.mod_data is None:
            return []
        
        similar_mods = []
        description_words = set(description.lower().split())
        
        for _, row in st.session_state.mod_data.iterrows():
            mod_words = set(row['mod_description'].lower().split())
            similarity = len(description_words.intersection(mod_words)) / len(description_words.union(mod_words))
            
            if similarity > 0.1 and row['mod_type'] == mod_type:
                similar_mods.append({
                    'mod_id': row['mod_id'],
                    'mod_description': row['mod_description'],
                    'mod_type': row['mod_type'],
                    'regulations': row['regulations'],
                    'loi': row['loi'],
                    'similarity_score': similarity
                })
        
        return sorted(similar_mods, key=lambda x: x['similarity_score'], reverse=True)[:5]
    
    def perform_analysis(self, description: str) -> dict:
        """Perform complete analysis"""
        mod_type = self.predict_mod_type(description)
        loi = self.predict_loi(mod_type, description)
        regulations = self.get_relevant_regulations(mod_type)
        similar_mods = self.find_similar_modifications(description, mod_type)
        
        return {
            'description': description,
            'predicted_type': mod_type,
            'loi': loi,
            'regulations': regulations,
            'similar_modifications': similar_mods,
            'confidence': 0.85 if mod_type != 'Systems' else 0.65
        }
    
    def get_regulation_explanation(self, regulation_code: str) -> str:
        """Get explanation for regulation codes"""
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
            "AMC 20-151": "Avionics equipment - Requirements for electronic flight systems"
        }
        return explanations.get(regulation_code, f"Aviation regulation {regulation_code}")
    
    def generate_chatbot_response(self, user_question: str) -> str:
        """Generate chatbot response"""
        if not st.session_state.last_analysis_results:
            return "I don't have any analysis results to discuss yet. Please run an analysis first!"
        
        question_lower = user_question.lower()
        results = st.session_state.last_analysis_results
        
        if any(word in question_lower for word in ['regulation', 'standard', 'compliance']):
            regulations = results.get('regulations', [])
            response = f"This {results['predicted_type']} modification requires compliance with {len(regulations)} key regulations:\\n\\n"
            for reg in regulations[:3]:
                response += f"• **{reg}**: {self.get_regulation_explanation(reg)}\\n\\n"
            return response
            
        elif any(word in question_lower for word in ['loi', 'level', 'involvement', 'complexity']):
            loi = results['loi']
            explanations = {
                'High': "High complexity requiring extensive analysis, testing, and documentation. Expect 12-24 months for certification.",
                'Medium': "Moderate complexity with standard certification procedures. Typically 6-12 months.",
                'Low': "Low complexity with minimal additional requirements. Usually 3-6 months."
            }
            return f"**Level of Involvement: {loi}**\\n\\n{explanations.get(loi, 'Assessment unavailable')}"
            
        elif any(word in question_lower for word in ['similar', 'comparison', 'match']):
            similar_mods = results.get('similar_modifications', [])
            if not similar_mods:
                return "No similar modifications found in the database."
            response = f"Found {len(similar_mods)} similar modifications:\\n\\n"
            for i, mod in enumerate(similar_mods[:3], 1):
                response += f"{i}. **{mod['mod_id']}** (Similarity: {mod['similarity_score']:.1%})\\n"
                response += f"   {mod['mod_description'][:100]}...\\n\\n"
            return response
            
        elif any(word in question_lower for word in ['safety', 'risk']):
            loi = results['loi']
            if loi == 'High':
                return "⚠️ **High-risk modification** requiring comprehensive safety analysis, extensive testing, and multiple safety assessments."
            elif loi == 'Medium':
                return "⚠️ **Moderate-risk modification** requiring standard safety analysis and functional testing."
            else:
                return "✅ **Lower-risk modification** with standard safety procedures and routine compliance checks."
                
        elif any(word in question_lower for word in ['cost', 'budget', 'expense']):
            loi = results['loi']
            costs = {
                'High': "Higher costs expected due to extensive engineering, multiple test phases, and extended certification.",
                'Medium': "Moderate costs with standard engineering effort and required testing.",
                'Low': "Lower costs with minimal engineering effort and basic testing requirements."
            }
            return f"**Cost Considerations ({loi} LOI):**\\n\\n{costs.get(loi, 'Cost assessment unavailable')}"
            
        else:
            mod_type = results['predicted_type']
            loi = results['loi']
            num_regs = len(results.get('regulations', []))
            return f"**Analysis Summary:**\\n\\n• Type: {mod_type}\\n• Complexity: {loi}\\n• Regulations: {num_regs}\\n\\nAsk me about regulations, safety, costs, or similar modifications for more details!"
    
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
        """Render sidebar with chatbot"""
        st.sidebar.markdown("## 🔧 Control Panel")
        
        # Data status
        if st.session_state.mod_data is not None:
            st.sidebar.success(f"✅ Data loaded: {len(st.session_state.mod_data)} modifications")
        else:
            st.sidebar.error("❌ Data not loaded")
        
        # Chatbot section
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🤖 Analysis Assistant")
        st.sidebar.markdown("*Ask me about your analysis results!*")
        
        # Quick questions
        st.sidebar.markdown("**Quick Questions:**")
        quick_questions = [
            "Explain the regulations",
            "What's the LOI assessment?", 
            "Show similar modifications",
            "What are safety considerations?",
            "Estimate costs and timeline"
        ]
        
        for question in quick_questions:
            if st.sidebar.button(question, key=f"quick_{question.replace(' ', '_')}"):
                self.handle_chatbot_question(question)
        
        # Custom question
        user_question = st.sidebar.text_input(
            "Ask a custom question:",
            placeholder="e.g., Why is this high complexity?",
            key="chatbot_input"
        )
        
        if st.sidebar.button("Ask", key="ask_button") and user_question:
            self.handle_chatbot_question(user_question)
        
        # Show recent conversation
        if st.session_state.chatbot_messages:
            st.sidebar.markdown("**Recent Conversation:**")
            # Show last 3 messages
            for message in st.session_state.chatbot_messages[-3:]:
                if message['role'] == 'user':
                    st.sidebar.markdown(f"<div class='chatbot-message user-message'><strong>You:</strong> {message['content']}</div>", 
                                      unsafe_allow_html=True)
                else:
                    st.sidebar.markdown(f"<div class='chatbot-message assistant-message'><strong>Assistant:</strong> {message['content'][:200]}...</div>", 
                                      unsafe_allow_html=True)
        
        # Clear chat
        if st.sidebar.button("Clear Chat", key="clear_chat"):
            st.session_state.chatbot_messages = []
            st.rerun()
    
    def handle_chatbot_question(self, question: str):
        """Handle chatbot question"""
        # Add user message
        st.session_state.chatbot_messages.append({
            'role': 'user',
            'content': question,
            'timestamp': datetime.now()
        })
        
        # Generate response
        response = self.generate_chatbot_response(question)
        
        # Add assistant response
        st.session_state.chatbot_messages.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now()
        })
        
        st.rerun()
    
    def render_input_section(self):
        """Render input section"""
        st.markdown("## 📝 Modification Description")
        
        examples = {
            "VHF Antenna Installation": "Installation of a new VHF antenna on the dorsal fuselage for improved communication range in oceanic flights.",
            "LED Cabin Lighting": "Retrofit of LED cabin lighting system replacing existing fluorescent lights to improve passenger comfort and reduce power consumption.",
            "Cargo Door Reinforcement": "Installation of reinforced cargo door frame to increase structural integrity and support heavier cargo loads.",
            "Weather Radar Upgrade": "Integration of advanced weather radar system with enhanced turbulence detection capabilities in nose radome."
        }
        
        selected_example = st.selectbox(
            "Choose an example or enter your own:",
            ["Custom Input"] + list(examples.keys())
        )
        
        if selected_example == "Custom Input":
            default_text = ""
        else:
            default_text = examples[selected_example]
        
        description = st.text_area(
            "Modification Description:",
            value=default_text,
            height=100,
            placeholder="Enter detailed description of the aircraft modification..."
        )
        
        return description
    
    def render_results(self, description: str):
        """Render analysis results"""
        if not description.strip():
            st.warning("Please enter a modification description to see analysis results.")
            return
        
        # Perform analysis
        results = self.perform_analysis(description)
        st.session_state.last_analysis_results = results
        
        st.markdown("## 🔍 Analysis Results")
        
        # Create columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Predicted Type</h4>
                <h2 style="color: #1f77b4;">{results['predicted_type']}</h2>
                <p>Confidence: {results['confidence']:.0%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Level of Involvement</h4>
                <h2 style="color: #1f77b4;">{results['loi']}</h2>
                <p>Based on analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Regulations</h4>
                <h2 style="color: #1f77b4;">{len(results['regulations'])}</h2>
                <p>Applicable standards</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Regulations
        st.markdown("### 📜 Applicable Regulations")
        for reg in results['regulations']:
            st.markdown(f"<span class='regulation-tag'>{reg}</span>", unsafe_allow_html=True)
        
        # Similar modifications
        if results['similar_modifications']:
            st.markdown("### 🔗 Similar Modifications")
            for mod in results['similar_modifications'][:3]:
                with st.expander(f"{mod['mod_id']} - Similarity: {mod['similarity_score']:.1%}"):
                    st.markdown(f"**Type:** {mod['mod_type']}")
                    st.markdown(f"**Description:** {mod['mod_description']}")
                    st.markdown(f"**LOI:** {mod['loi']}")
    
    def run(self):
        """Run the application"""
        self.render_header()
        
        # Create main layout
        main_col, sidebar_col = st.columns([3, 1])
        
        with main_col:
            # Tabs
            tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📊 Statistics", "ℹ️ About"])
            
            with tab1:
                description = self.render_input_section()
                
                if st.button("🔍 Analyze Modification", type="primary"):
                    if description.strip():
                        self.render_results(description)
                    else:
                        st.warning("Please enter a modification description.")
                
                # Show results if available
                if st.session_state.last_analysis_results:
                    self.render_results(st.session_state.last_analysis_results['description'])
            
            with tab2:
                st.markdown("## 📊 Database Statistics")
                if st.session_state.mod_data is not None:
                    df = st.session_state.mod_data
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        type_counts = df['mod_type'].value_counts()
                        st.bar_chart(type_counts)
                        st.caption("Modification Types Distribution")
                    
                    with col2:
                        if 'loi' in df.columns:
                            loi_counts = df['loi'].value_counts()
                            st.bar_chart(loi_counts)
                            st.caption("Level of Involvement Distribution")
            
            with tab3:
                st.markdown("""
                ## About This Tool
                
                This Aircraft Modification Certification Support Tool helps aviation professionals:
                
                - **Classify** aircraft modifications by type
                - **Map** relevant regulatory requirements
                - **Predict** Level of Involvement (LOI)
                - **Find** similar historical modifications
                - **Get AI assistance** through the chatbot
                
                ### How to Use
                1. Enter your modification description
                2. Click "Analyze Modification"
                3. Review the results
                4. Ask the chatbot questions for detailed explanations
                
                ### Chatbot Features
                - Explains regulatory requirements
                - Clarifies LOI assessments
                - Provides safety considerations
                - Estimates timelines and costs
                - Shows similar modifications
                """)
        
        # Render sidebar
        self.render_sidebar()

# Run the app
if __name__ == "__main__":
    app = SimplifiedCertificationApp()
    app.run()
