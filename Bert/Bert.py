import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns


def main():
    df = pd.read_csv('./spam.csv')
    # check count and unique and top values and their frequency
    df['label'].value_counts()

    # creating 2 new dataframe as df_ham , df_spam

    df_spam = df[df['label'] == 'spam']

    df_ham = df[df['label'] == 'ham']

    print("Ham Dataset Shape:", df_ham.shape)

    print("Spam Dataset Shape:", df_spam.shape)

    # downsampling ham dataset - take only random 747 example
    # will use df_spam.shape[0] - 747
    df_ham_downsampled = df_ham.sample(df_spam.shape[0])
    print(df_ham_downsampled.shape)

    # concating both dataset - df_spam and df_ham_balanced to create df_balanced dataset
    df_balanced = pd.concat([df_spam, df_ham_downsampled])
    print(df_balanced['label'].value_counts())

    print(df_balanced.sample(10))

    # creating numerical representation of category - one hot encoding
    df_balanced['spam'] = df_balanced['label'].apply(lambda x: 1 if x == 'spam' else 0)

    # displaying data - spam -1 , ham-0
    print(df_balanced.sample(4))

    # loading train test split

    X_train, X_test, y_train, y_test = train_test_split(df_balanced['text'], df_balanced['spam'],
                                                        stratify=df_balanced['spam'])

    # downloading preprocessing files and model
    bert_preprocessor = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
    bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='Inputs')
    preprocessed_text = bert_preprocessor(text_input)
    embeed = bert_encoder(preprocessed_text)
    dropout = tf.keras.layers.Dropout(0.1, name='Dropout')(embeed['pooled_output'])
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='Dense')(dropout)

    # creating final model
    model = tf.keras.Model(inputs=[text_input], outputs=[outputs])

    print(model.summary())

    Metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
               tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall')
               ]

    # compiling our model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=Metrics)

    history = model.fit(X_train, y_train, epochs=10)

    model.evaluate(X_test, y_test)

    # getting y_pred by predicting over X_text and flattening it
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()  # require to be in one-dimensional array , for easy manipulation

    y_test = y_test.to_numpy()
    y_pred = np.where(y_pred > 0.5, 1, 0)
    # creating confusion matrix
    print(type(y_test), type(y_pred))
    print(y_test)
    print(y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    # plotting as a graph - importing seaborn

    # creating a graph out of confusion matrix
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # printing classification report
    print(classification_report(y_test, y_pred))

    predict_text = [
        # Spam
        'We’d all like to get a $10,000 deposit on our bank accounts out of the blue, but winning a prize—especially if you’ve never entered a contest',
        'Netflix is sending you a refund of $12.99. Please reply with your bank account and routing number to verify and get your refund',
        'Your account is temporarily frozen. Please log in to to secure your account ',
        # ham
        'The article was published on 18th August itself',
        'Although we are unable to give you an exact time-frame at the moment, I would request you to stay tuned for any updates.',
        'The image you sent is a UI bug, I can check that your article is marked as regular and is not in the monetization program.'
    ]
    test_results = model.predict(predict_text)
    output = np.where(test_results > 0.5, 'spam', 'ham')
    print(output)


if __name__ == "__main__":
    main()
