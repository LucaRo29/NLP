## Required packages
import random
import spacy
import pandas as pd
import seaborn as sns
from spacy.tokens import DocBin
from spacy.util import minibatch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import datetime
import timeit
import time

#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#---------------------USE SPACY VERSION: 3.X !------------------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#

## Paths
data_path = "../data/spam.csv"
#data_path = "../data/spam_preprocessed.csv"

## Configurations
sns.set(style="darkgrid")


## UDF's
def train_model(model, train_data, optimizer, batch_size, epochs=10):
    losses = {}
    random.seed(1)

    for epoch in range(epochs):
        random.shuffle(train_data)

        batches = minibatch(train_data, size=batch_size)
        for batch in batches:
            # Split batch into texts and labels
            texts, labels = zip(*batch)

            # Update model with texts and labels
            model.update(texts, labels, sgd=optimizer, losses=losses)
        # print("Loss: {}".format(losses['textcat']))

    return losses['textcat']


def get_predictions(model, texts):
    # Use the model's tokenizer to tokenize each input text
    docs = [model.tokenizer(text) for text in texts]

    # Use textcat to get the scores for each doc
    textcat = model.get_pipe('textcat')
    scores, _ = textcat.predict(docs)

    # From the scores, find the label with the highest score/probability
    predicted_labels = scores.argmax(axis=1)
    predicted_class = [textcat.labels[label] for label in predicted_labels]

    return predicted_class


# import spacy
#
# # tqdm is a great progress bar for python
# # tqdm.auto automatically selects a text based progress for the console
# # and html based output in jupyter notebooks
# from tqdm.auto import tqdm
#
# # DocBin is spacys new way to store Docs in a binary format for training later
# from spacy.tokens import DocBin
#
# # We want to classify movie reviews as positive or negative
# # load movie reviews as a tuple (text, label)
# data = pd.read_csv(data_path, encoding='latin-1')
# train_data, valid_data = train_test_split(
#         data['text'], data['label'], test_size=0.33, random_state=7)
#
#
#
# # load a medium sized english language model in spacy
# nlp = spacy.load("en_core_web_md")
#
# # we are so far only interested in the first 5000 reviews
# # this will keep the training time short.
# # In practice take as much data as you can get.
# num_texts = 5000
#
#
# def make_docs(data):
#     """
#     this will take a list of texts and labels and transform them in spacy documents
#
#     texts: List(str)
#     labels: List(labels)
#
#     returns: List(spacy.Doc.doc)
#     """
#
#     docs = []
#
#     # nlp.pipe([texts]) is way faster than running nlp(text) for each text
#     # as_tuples allows us to pass in a tuple, the first one is treated as text
#     # the second one will get returned as it is.
#
#     for doc, label in tqdm(nlp.pipe(data, as_tuples=True), total=len(data)):
#         # we need to set the (text)cat(egory) for each document
#         doc.cats["positive"] = label
#
#         # put them into a nice list
#         docs.append(doc)
#
#     return docs
#
#
# # we are so far only interested in the first 5000 reviews
# # this will keep the training time short.
# # In practice take as much data as you can get.
# # you can always reduce it to make the script even faster.
# num_texts = 5000
#
# # first we need to transform all the training data
# train_docs = make_docs(train_data[:num_texts])
# # then we save it in a binary file to disc
# doc_bin = DocBin(docs=train_docs)
# doc_bin.to_disk("./data/train.spacy")
#
# # repeat for validation data
# valid_docs = make_docs(valid_data[:num_texts])
# doc_bin = DocBin(docs=valid_docs)
# doc_bin.to_disk("./data/valid.spacy")
#

######## Main method ########
def main():
    # Load dataset
    
    df = pd.read_csv(data_path, encoding='latin-1')
    #df = pd.read_csv(data_path)
    # observations = len(df.index)
    # print("Dataset Size: {}".format(observations))

    # data.text = data.text.fillna('') # needed if preprocessing is used


    # nlp = spacy.blank("en")
    nlp = spacy.load("en_core_web_sm")

    train = df.sample(frac=0.8, random_state=25)
    test = df.drop(train.index)

    print(train.shape, test.shape)

    train['tuples'] = train.apply(lambda row: (row['text'], row['label']), axis=1)
    train = train['tuples'].tolist()
    test['tuples'] = test.apply(lambda row: (row['text'], row['label']), axis=1)
    test = test['tuples'].tolist()

    def document(data):
        text = []
        for doc, label in nlp.pipe(data, as_tuples=True):
            if (label == 'ham'):
                doc.cats['ham'] = 1
                doc.cats['spam'] = 0

            else:
                doc.cats['ham'] = 0
                doc.cats['spam'] = 1
            text.append(doc)

        return text

    start_time = time.time()

    # passing the train dataset into function 'document'
    train_docs = document(train)

    # Creating binary document using DocBin function in spaCy
    doc_bin = DocBin(docs=train_docs)

    # Saving the binary document as train.spacy
    doc_bin.to_disk("train.spacy")
    end_time = time.time()

    # Printing the time duration for train dataset
    print('Duration: {} seconds'.format(end_time - start_time))

    start_time = time.time()

    # passing the test dataset into function 'document'
    test_docs = document(test)
    doc_bin = DocBin(docs=test_docs)
    doc_bin.to_disk("valid.spacy")
    end_time = time.time()

    # Printing the time duration for test dataset
    print('Duration: {} seconds'.format(end_time - start_time))


    # Create an empty spacy model

    # Create the TextCategorizer with exclusive classes and "bow" architecture
    # config = {"model": {"@architectures": "spacy.TextCatBOW.v2", "exclusive_classes": "true", "ngram_size": 1,
    #                     "no_output_layer": "false"}}
    # text_cat = nlp.add_pipe("textcat", config=config)
    # text_cat.add_label('ham')
    # text_cat.add_label('spam')

    # Adding the TextCategorizer to the created empty model
    # nlp.add_pipe('text_cat')

    # Add labels to text classifier
    # nlp.add_label("ham")
    # nlp.add_label("spam")

    # Split data into train and test datasets
    # x_train, x_test, y_train, y_test = train_test_split(
    #     data['text'], data['label'], test_size=0.33, random_state=7)
    #
    # # Create the train and test data for the spacy model
    # train_lables = [{'cats': {'ham': label == 'ham',
    #                           'spam': label == 'spam'}} for label in y_train]
    # test_lables = [{'cats': {'ham': label == 'ham',
    #                          'spam': label == 'spam'}} for label in y_test]

    # Spacy model data
    # train_data = list(zip(x_train, train_lables))
    # test_data = list(zip(x_test, test_lables))

    # Model configurations
    # optimizer = nlp.begin_training()
    batch_size = 5
    epochs = 10

    # Training the model
    # train_model(nlp, train_data, optimizer, batch_size, epochs)

    # Sample predictions
    # print(train_data[0])
    # sample_test = nlp(train_data[0][0])
    # print(sample_test.cats)
    #
    # # Train and test accuracy
    # train_predictions = get_predictions(nlp, x_train)
    # test_predictions = get_predictions(nlp, x_test)
    # train_accuracy = accuracy_score(y_train, train_predictions)
    # test_accuracy = accuracy_score(y_test, test_predictions)
    #
    # print("Train accuracy: {}".format(train_accuracy))
    # print("Test accuracy: {}".format(test_accuracy))
    #
    # # Creating the confusion matrix graphs
    # cf_train_matrix = confusion_matrix(y_train, train_predictions)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cf_train_matrix, annot=True, fmt='d')
    # plt.show()
    #
    # cf_test_matrix = confusion_matrix(y_test, test_predictions)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cf_test_matrix, annot=True, fmt='d')
    # plt.show()


if __name__ == "__main__":
    main()


#RESULT:
'''
spam.csv
    Train accuracy: 0.999196356817573
    Test accuracy: 0.9809679173463839

spam_preprocessed.csv: 
    Train accuracy: 0.998392713635146
    Test accuracy: 0.9885869565217391
    
    --> Train accuracy is lower but test accuracy increased by 0.008 
'''