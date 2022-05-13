import pandas as pd
import numpy as np
import re
import collections

#import contractions

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import warnings
warnings.simplefilter(action='ignore', category=Warning)
import keras
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

from sklearn.preprocessing import LabelEncoder

# --------INFO----------------------------------------------------------------------------------------------------------
# Tutorial: https://www.analyticsvidhya.com/blog/2021/05/sms-spam-detection-using-lstm-a-hands-on-guide/
#
# The problem with this tutorial:
#
# The python package contractions is not available as ARM version therefore I am not able to use it on M1 out of the box
#
# There is no split in data / test / validation
# Simply the whole dataset is used to train the model!
#
#TODO: Question: what are these .pkl files / how to use them?
#
# ----------------------------------------------------------------------------------------------------------------------


def set_print_params():
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 10)


def word_count_plot(data):
    # finding words along with count
    word_counter = collections.Counter([word for sentence in data for word in sentence.split()])
    most_count = word_counter.most_common(30)  # 30 most common words
    # sorted data frame
    most_count = pd.DataFrame(most_count, columns=["Word", "Count"]).sort_values(by="Count")
    most_count.plot.barh(x="Word", y="Count", color="green", figsize=(10, 15))


# lem = WordNetLemmatizer()
# def preprocessing(data):
#       sms = contractions.fix(data) # converting shortened words to original (Eg:"I'm" to "I am")
#       sms = sms.lower() # lower casing the sms
#       sms = re.sub(r'https?://S+|www.S+', "", sms).strip() #removing url
#       sms = re.sub("[^a-z ]", "", sms) # removing symbols and numbes
#       sms = sms.split() #splitting
#       # lemmatization and stopword removal
#       sms = [lem.lemmatize(word) for word in sms if not word in set(stopwords.words("english"))]
#       sms = " ".join(sms)
#       return sms


def create_model(tokenizer, max_length_sequence):
    TOT_SIZE = len(tokenizer.word_index) + 1

    lstm_model = Sequential()
    lstm_model.add(Embedding(TOT_SIZE, 32, input_length=max_length_sequence))
    lstm_model.add(LSTM(100))
    lstm_model.add(Dropout(0.4))
    lstm_model.add(Dense(20, activation="relu"))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(Dense(1, activation="sigmoid"))

    lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print("\n")
    lstm_model.summary()

    return lstm_model


def main():
    print("START!")
    df = pd.read_csv("../../../Project/NLP/LSTM/spam_luca.csv", encoding='latin-1')
    print(df.head())
    print("shape: ", df.shape)  # output - (5572, 8674))

    # TODO: following steps should be done in preprocessing
    df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
    df.columns = ["SpamHam", "Tweet"]

    sns.countplot(df["SpamHam"])
    plt.show()

    word_count_plot(df["Tweet"])
    plt.show()

    # preprocessing step which can not be executed since the used library is not available for M1 processor!
    # X = df["v2"].apply(preprocessing)
    # word_count_plot(X)
    X = df["Tweet"]    #without preprocessing!

    lb_enc = LabelEncoder()
    y = lb_enc.fit_transform(df["SpamHam"])

    tokenizer = Tokenizer()  # initializing the tokenizer
    tokenizer.fit_on_texts(X)  # fitting on the sms data
    text_to_sequence = tokenizer.texts_to_sequences(X)  # creating the numerical sequence

    print("\n")
    for i in range(5):
        print("Text               : ", X[i])
        print("Numerical Sequence : ", text_to_sequence[i])

    #print("\n")
    #print(tokenizer.index_word)  # this will output a dictionary of index and words


    #Normalizing sequences
    max_length_sequence = max([len(i) for i in text_to_sequence])
    # finding the length of largest sequence
    padded_sms_sequence = pad_sequences(text_to_sequence, maxlen=max_length_sequence, padding="pre")
    print("\n")
    print(padded_sms_sequence)

    #Creating model
    lstm_model = create_model(tokenizer, max_length_sequence)

    #Train the model
    lstm_model.fit(padded_sms_sequence, y, epochs=5, validation_split=0.2, batch_size=16)

    #Save model and tokenizer as model
    pickle.dump(tokenizer, open("sms_spam_tokenizer.pkl", "wb"))
    pickle.dump(lstm_model, open("lstm_model.pkl", "wb"))


if __name__ == "__main__":
    set_print_params()
    main()
