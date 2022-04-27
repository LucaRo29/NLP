"""
===============================================
Objective: Text preprocessing techniques workflow in python
Author: Sharmila.Polamuri
Blog: https://dataaspirant.com
Date: 2020-09-14
===============================================
"""

import pandas as pd
from preprocessing import Preprocess

#Available preprocessing techniques
"""
    lcc = lower case convertion
rht = Removing HTML tags
rurls = Revoing Urls 
rn = Removing Numbers
    ntw = convert numbers to words
    sc = Spelling Correction
    ata = convert accented to ASCII code
    sto = short_to_original
ec = Expanding Contractions
    ps = Stemming (Porter Stemming)
    l = Lemmatization
re = Removing Emojis
ret = Removing Emoticons
ew = Convert Emojis to words
etw = Convert Emoticons to words
    rp = Removing Punctuations
    rs = Removing Stopwords
rfw = Removing Frequent Words
rrw = Removing Rare Words
rsc = Removing Single characters
    res = Removing Extra Spaces
"""

techniques = ["lcc", "ntw", "rp", "res", "sto", "ata",   "sc", "rs", "ps", "l"]
#techniques = ["lcc", "ntw"]

# Load dataset
data_path = "../data/spam.csv"
data = pd.read_csv(data_path)

documents = data['text'].tolist()

# initiate Preprocess object
preprocessing = Preprocess()

preprocessed_documents = preprocessing.preprocessing(documents, techniques)

#somehow safe this shit into a DF
data['text'] = preprocessed_documents

file_name = "../data/spam_preprocessed.csv"
data.to_csv(file_name, sep=',', encoding='utf-8', index=False)


print("END")