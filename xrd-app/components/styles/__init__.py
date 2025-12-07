"""
Custom Styles for XRD Analyzer
"""

import streamlit as st
import os

def inject_custom_css():
    """Inject custom CSS into Streamlit app"""
    css_path = os.path.join(os.path.dirname(__file__), 'custom.css')

    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            css = f.read()
    else:
        css = get_default_css()

    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


def get_default_css():
    """Return default CSS styles"""
    return """
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    /* Sidebar */
    .css-1d391kg {
        padding-top: 1rem;
    }

    /* Headers */
    h1, h2, h3 {
        color: #1f77b4;
    }

    /* Metric cards */
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
    }

    /* File uploader */
    .stFileUploader {
        border: 2px dashed #1f77b4;
        border-radius: 0.5rem;
        padding: 1rem;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 0.5rem;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1f77b4;
    }

    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }

    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    """
