import streamlit as st

# importing sys

#IMPORTANT: set right path:
#Enter the folloing into Terminal
#    export PYTHONPATH='path/to/NLP/project' ####

from Unused.model import MLModel as ml

txt = st.text_area('Text to analyze', '')


if st.button('Init Model'):
    ml.init()

if st.button('Analyse Text Model'):
     ml.analyse_text(txt)

