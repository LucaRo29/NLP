import streamlit as st
import numpy as np
import pathlib as pl
import pandas as pd

# importing sys
import sys

#IMPORTANT: set right path:
#Enter the folloing into Terminal
#    export PYTHONPATH='path/to/NLP/project' ####

from model import MLModel as ml


txt = st.text_area('Text to analyze', '')


if st.button('Init Model'):
    ml.init()

if st.button('Analyse Text Model'):
     ml.analyse_text(txt)

