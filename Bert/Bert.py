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
    # df1 = pd.read_csv('../Data/spam.csv',index_col=False)
    df = pd.read_csv('../Data/preprocessed/enron_prep.csv', index_col=False)
    df2 = pd.read_csv('../Data/raw/data.csv', index_col=False)

    # print(df.head())

    # print(df.shape)
    # df = df.dropna()
    # print(df.shape)
    #
    #
    #
    # df.to_csv("../Data/data.csv", index=False)
    # print(df.head())
    # print(df1.head())
    # return
    # check count and unique and top values and their frequency
    df['label'].value_counts()

    # creating 2 new dataframe as df_ham , df_spam

    df_spam = df[df['label'] == 1]

    df_ham = df[df['label'] == 0]

    df2_spam = df2[df2['label'] == 'spam']

    df2_ham = df2[df2['label'] == 'ham']

    print("Ham Dataset Shape:", df_ham.shape)

    print("Spam Dataset Shape:", df_spam.shape)

    print("Ham Dataset Shape:", df2_ham.shape)

    print("Spam Dataset Shape:", df2_spam.shape)

    print(df.sample(5))

    print(df2.sample(2))

    # downsampling ham dataset - take only random 747 example
    # will use df_spam.shape[0] - 747
    df_ham_downsampled = df_ham.sample(df_spam.shape[0])
    print(df_ham_downsampled.shape)

    # concating both dataset - df_spam and df_ham_balanced to create df_balanced dataset
    df_balanced = pd.concat([df_spam, df_ham_downsampled])
    print(df_balanced['label'].value_counts())

    print()
    print('Balanced:')
    print(df_balanced.sample(10))

    df2_ham_downsampled = df2_ham.sample(df2_spam.shape[0])
    print(df2_ham_downsampled.shape)

    # concating both dataset - df_spam and df_ham_balanced to create df_balanced dataset
    df2_balanced = pd.concat([df2_spam, df2_ham_downsampled])
    print(df2_balanced['label'].value_counts())

    print()
    print('Balanced:')
    print(df2_balanced.sample(10))

    # creating numerical representation of category - one hot encoding
    df2_balanced['spam'] = df2_balanced['label'].apply(lambda x: 1 if x == 'spam' else 0)

    print('Balanced after lambda :')
    print(df2_balanced.sample(10))

    # displaying data - spam -1 , ham-0
    # print(df_balanced.sample(4))

    # loading train test split
    print()
    print()
    print()
    print()
    print()
    print()
    print()


    X_train, X_test, y_train, y_test = train_test_split(df_balanced['transformed_text'], df_balanced['label'])
    print((type(X_train)))
    print(X_train.head(2))
    print(type(y_train))
    print(y_train.head(2))

    X_train, X_test, y_train, y_test = train_test_split(df2_balanced['text'], df2_balanced['spam'],
                                                        stratify=df2_balanced['spam'])
    print("!!!!!!!!!!!!!!!!!!!!!!")
    print((type(X_train)))
    print(X_train.head(2))
    print(type(y_train))
    print(y_train.head(2))

    # X_train = X_train.to_numpy()
    # print((type(X_train)))
    # print(X_train)
    #
    # print(type(y_train))
    # print(y_train)
    #
    # y_train = y_train.to_numpy()
    #
    # print(type(y_train))
    # print(y_train)
    # downloading preprocessing files and model
    bert_preprocessor = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
    bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='Inputs')
    preprocessed_text = bert_preprocessor(text_input)
    embed = bert_encoder(preprocessed_text)
    dropout = tf.keras.layers.Dropout(0.1, name='Dropout')(embed['pooled_output'])
    x = tf.keras.layers.Dense(128, activation='relu')(dropout)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='Dense')(x)

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

    history = model.fit(X_train, y_train, epochs=1)

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
    plt.savefig('../Bert/Data/bertCM.png')
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
