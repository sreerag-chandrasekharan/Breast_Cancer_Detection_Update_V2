import streamlit as st
import pickle
import pandas as pd
import json
import plotly.graph_objects as go
import numpy as np


#-- add the slidebar

def add_sidebar():

    st.sidebar.header("Cell Nuclei Measurements")

    slider_labels = [
    ("Radius", "radius_mean"),
    ("Texture", "texture_mean"),
    ("Perimeter", "perimeter_mean"),
    ("Area", "area_mean"),
    ("Smoothness", "smoothness_mean"),
    ("Compactness", "compactness_mean"),
    ("Concavity", "concavity_mean"),
    ("Concave points", "concave_points_mean"),
    ("Symmetry", "symmetry_mean"),
    ("Fractal dimension", "fractal_dimension_mean"),
]
    input_dict = {}

    # Add the sliders
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value= None,
            max_value= None,
            value= None,
            step= None,
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