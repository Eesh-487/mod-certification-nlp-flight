# -*- coding: utf-8 -*-
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
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point"""
    st.title("Aircraft Modification Certification Support Tool")
    st.markdown("AI-powered assistant for aircraft modification classification and certification support")
    
    # Check if data exists
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mods_dataset.csv')
    
    if not os.path.exists(data_path):
        st.error("Dataset not found. Please run setup.py first.")
        st.stop()
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        st.success(f"Loaded {len(df)} modifications")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Main interface
    st.header("Modification Analysis")
    
    # Input section
    description = st.text_area(
        "Enter modification description:",
        height=100,
        placeholder="Enter detailed description of the aircraft modification..."
    )
    
    if st.button("Analyze Modification", type="primary"):
        if description.strip():
            # Simple analysis
            analyze_modification(description, df)
        else:
            st.warning("Please enter a modification description.")
    
    # Show dataset statistics
    show_statistics(df)

def analyze_modification(description, df):
    """Analyze the modification description"""
    st.subheader("Analysis Results")
    
    # Simple rule-based classification
    mod_type = predict_mod_type(description)
    st.write(f"**Predicted Type:** {mod_type}")
    
    # Simple text statistics
    word_count = len(description.split())
    char_count = len(description)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Word Count", word_count)
    with col2:
        st.metric("Character Count", char_count)
    with col3:
        st.metric("Estimated LOI", predict_loi(mod_type, word_count))

def predict_mod_type(description):
    """Simple rule-based modification type prediction"""
    description_lower = description.lower()
    
    if any(word in description_lower for word in ['antenna', 'radio', 'navigation', 'radar', 'avionics']):
        return 'Avionics'
    elif any(word in description_lower for word in ['wing', 'fuselage', 'structure', 'frame']):
        return 'Structure'
    elif any(word in description_lower for word in ['cabin', 'seat', 'galley', 'passenger']):
        return 'Cabin'
    elif any(word in description_lower for word in ['hydraulic', 'fuel', 'air conditioning']):
        return 'Systems'
    elif any(word in description_lower for word in ['emergency', 'safety', 'evacuation']):
        return 'Safety'
    else:
        return 'Systems'

def predict_loi(mod_type, word_count):
    """Simple LOI prediction"""
    if mod_type in ['Structure', 'Safety'] or word_count > 100:
        return 'High'
    elif word_count > 50:
        return 'Medium'
    else:
        return 'Low'

def show_statistics(df):
    """Show dataset statistics"""
    st.header("Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Modifications", len(df))
    with col2:
        st.metric("Modification Types", df['mod_type'].nunique())
    with col3:
        st.metric("Aircraft Types", df['aircraft_type'].nunique())
    with col4:
        avg_regs = df['regulations'].str.split(',').str.len().mean()
        st.metric("Avg Regulations", f"{avg_regs:.1f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Modification type distribution
        type_counts = df['mod_type'].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, 
                    title="Modification Types")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # LOI distribution
        loi_counts = df['loi'].value_counts()
        fig = px.bar(x=loi_counts.index, y=loi_counts.values,
                    title="Level of Involvement",
                    labels={'x': 'LOI Level', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
