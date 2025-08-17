import streamlit as st
import pickle
import pandas as pd
import json
import plotly.graph_objects as go
import numpy as np

# min and max values for the sliders
with open("models/feature_stats.json", "r") as f:
    FEATURE_STATS = json.load(f)

def get_min_max_mean(key):
    return FEATURE_STATS[key]

# Define the slider labels and keys
SLIDER_LABELS = [
    ("Radius (mean)", "radius_mean"),
    ("Texture (mean)", "texture_mean"),
    ("Perimeter (mean)", "perimeter_mean"),
    ("Area (mean)", "area_mean"),
    ("Smoothness (mean)", "smoothness_mean"),
    ("Compactness (mean)", "compactness_mean"),
    ("Concavity (mean)", "concavity_mean"),
    ("Concave points (mean)", "concave points_mean"),
    ("Symmetry (mean)", "symmetry_mean"),
    ("Fractal dimension (mean)", "fractal_dimension_mean"),
    ("Radius (se)", "radius_se"),
    ("Texture (se)", "texture_se"),
    ("Perimeter (se)", "perimeter_se"),
    ("Area (se)", "area_se"),
    ("Smoothness (se)", "smoothness_se"),
    ("Compactness (se)", "compactness_se"),
    ("Concavity (se)", "concavity_se"),
    ("Concave points (se)", "concave points_se"),
    ("Symmetry (se)", "symmetry_se"),
    ("Fractal dimension (se)", "fractal_dimension_se"),
    ("Radius (worst)", "radius_worst"),
    ("Texture (worst)", "texture_worst"),
    ("Perimeter (worst)", "perimeter_worst"),
    ("Area (worst)", "area_worst"),
    ("Smoothness (worst)", "smoothness_worst"),
    ("Compactness (worst)", "compactness_worst"),
    ("Concavity (worst)", "concavity_worst"),
    ("Concave points (worst)", "concave points_worst"),
    ("Symmetry (worst)", "symmetry_worst"),
    ("Fractal dimension (worst)", "fractal_dimension_worst"),
]

#-- add the slidebar

def add_sidebar():

    st.sidebar.header("Cell Nuclei Measurements")

    input_dict = {}

    # Add the sliders
    for label, key in SLIDER_LABELS:
        min_value, max_value = get_min_max_mean(key)
        input_dict[key] = st.sidebar.slider(
            label,
            min_value= min_value,
            max_value= max_value,
            value= (min_value + max_value) / 2,  # Default value is the mean
            format="%.2f")



#---- main function
def main():
    st.set_page_config(page_title="Breast Cancer Diagnosis",
        page_icon="👩‍⚕️", 
        layout="wide", 
        initial_sidebar_state="expanded")

    #--- add the sidebar
    input_dict = add_sidebar()
    # ----Set up the structure

    with st.container():
        st.title("Breast Cancer Diagnosis")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")
        col1, col2 = st.columns([4,2])
        with col1:
            st.write("Column 1")
        with col2:
            st.write("Column 2")


#----- define the main function


if __name__ == '__main__':
    main()