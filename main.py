import spacy
from spacy.tokens import DocBin
import pandas as pd

from examples.SpacyExample import data_path

data_path = "../data/spam.csv"


###
# This creates the training and test spacy doc from the medium english model
# We can train the model by executing in the shell
# '$python -m spacy train config.cfg --verbose  --output ./output_updated'
###
def main():
    # Load dataset
    df = pd.read_csv(data_path, encoding='latin-1')

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

    # passing the train dataset into function 'document'
    train_docs = document(train)

    # Creating binary document using DocBin function in spaCy
    doc_bin = DocBin(docs=train_docs)

    # Saving the binary document as train.spacy
    doc_bin.to_disk("./data/train.spacy")

    test_docs = document(test)
    doc_bin = DocBin(docs=test_docs)
    doc_bin.to_disk("./data/valid.spacy")


if __name__ == "__main__":
    main()
