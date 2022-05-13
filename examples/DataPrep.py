
## Required packages
import random
import spacy
import pandas as pd
import seaborn as sns
from spacy.util import minibatch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

## Paths
data_path = "../Bert/spam.csv"
data_path2 = "../Bert/enron_spam_data.csv"

from string import punctuation
import re

from nltk.stem.snowball import SnowballStemmer
# nltk.download('wordnet') # uncomment to download 'wordnet'
from nltk.corpus import wordnet as wn


def clean_email(email):
    """ Remove all punctuation, urls, numbers, and newlines.
    Convert to lower case.
    Args:
        email (unicode): the email
    Returns:
        email (unicode): only the text of the email
    """
    email = re.sub(r'http\S+', ' ', email)
    email = re.sub("\d+", " ", email)
    email = email.replace('\n', ' ')
    email = email.translate(str.maketrans("", "", punctuation))
    email = email.lower()
    return email





def preproces_text(email):
    """ Split the text string into individual words, stem each word,
    and append the stemmed word to words. Make sure there's a single
    space between each stemmed word.
    Args:
        email (unicode): the email
    Returns:
        words (unicode): the text of the email
    """

    words = ""
    # Create the stemmer.
    stemmer = SnowballStemmer("english")
    # Split text into words.
    email = email.split()
    for word in email:
        # Optional: remove unknown words.
        # if wn.synsets(word):
        words = words + stemmer.stem(word) + " "

    return words



######## Main method ########

def main():


    merged_data = pd.DataFrame(columns=["label", "text"])

    # Load dataset 1
    dataset1 = pd.read_csv(data_path,encoding='latin-1')
    print(dataset1.head())
    print(dataset1.columns.values)
   # dataset1.drop(columns=['Unnamed: 2', 'Unnamed: 3','Unnamed: 4'],inplace=True)
    print(dataset1.columns.values)

    # Load dataset 2
    dataset2 = pd.read_csv(data_path2, encoding='latin-1')
    print(dataset2.head())
    print(dataset2.columns.values)

    #Drop Nans
    dataset2 = dataset2.dropna(subset=['Message'])
    #clean data

    #dataset2['Message'] = dataset2['Message'].apply(clean_email)

    #prep data

   # dataset2['Message'] = dataset2['Message'].apply(preproces_text)

    dataset2.drop(columns=['Message ID', 'Subject','Date'],inplace=True)
    dataset2.rename(columns={"Message": "text", "Spam/Ham": "label"})
    dataset2 = dataset2.reindex(columns=["label", "text"])
    print(dataset2.columns.values)

    frames = [dataset1,dataset2]
    result = pd.concat(frames)
    print(result.columns.values)
    print(result.head())
    result.to_csv("../Bert/data.csv")


if __name__ == "__main__":
    main()