import keras
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns


def main():
    df = pd.read_csv('../Data/preprocessed/enron_prep.csv', index_col=False)
    df = pd.read_csv('../Data/preprocessed/dataset_elias_preprocessed.csv', index_col=False)
    new_model = True
    downsample = False

    # check count and unique and top values and their frequency
    df['label'].value_counts()

    # creating 2 new dataframe as df_ham , df_spam

    df_spam = df[df['label'] == 1]

    df_ham = df[df['label'] == 0]

    # downsampling ham dataset

    df_ham_downsampled = df_ham.sample(df_spam.shape[0])

    # concating both dataset - df_spam and df_ham_balanced to create df_balanced dataset
    df_balanced = pd.concat([df_spam, df_ham_downsampled])

    df_balanced.dropna(inplace=True)

    if (downsample):
        df_balanced = df_balanced.sample(100)
        print('data is downsampled')

    X_train, X_val, y_train, y_val = train_test_split(df_balanced['text'], df_balanced['label'], train_size=0.7)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, train_size=0.5)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    return
    # downloading preprocessing files and model
    model = None
    if (new_model):

        print('Generating new model')
        bert_preprocessor = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
        bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')

        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='Inputs')
        preprocessed_text = bert_preprocessor(text_input)
        embed = bert_encoder(preprocessed_text)
        dropout = tf.keras.layers.Dropout(0.5, name='Dropout')(embed['pooled_output'])
        x = tf.keras.layers.Dense(64, activation='relu')(dropout)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='Dense')(x)

        # creating final model
        model = tf.keras.Model(inputs=[text_input], outputs=[outputs])

        model.save("../Bert/Model")
    else:
        print('Loading old model')
        model = keras.models.load_model("../Bert/Model")

    if (model is None):
        print("Error: No model")
        return

    print(model.summary())

    Metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
               tf.keras.metrics.Precision(name='precision'),
               tf.keras.metrics.Recall(name='recall')
               ]

    # compiling our model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=Metrics)

    es = EarlyStopping(patience=5, verbose=1, min_delta=0.001, monitor='loss', mode='auto',
                       restore_best_weights=True)

    num_epochs = 2
    history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), batch_size=50,
                        callbacks=[es], shuffle=True)

    # getting y_pred by predicting over X_test
    y_pred = model.predict(X_test)

    # require to be in one-dimensional array
    y_pred = y_pred.flatten()

    # convert logits to labels
    y_pred = np.where(y_pred > 0.5, 1, 0)

    # creating confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # print(cm)
    best_epoch = es.best_epoch + 1
    # creating a graph out of confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # creating a graph out of confusion matrix
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # plt.savefig('../Bert/Data/bert_CM.png')
    #plt.savefig('/content/drive/MyDrive/NLP/Bert_' + dataset + '/figures/bert' + dataset + '_CM.png')
    plt.show()

    # epochs = range(1, num_epochs+1)
    # print(epochs)
    plt.plot(history.history['loss'], 'g', label='Training loss')
    plt.plot(history.history['val_loss'], 'b', label='validation loss')
    plt.axvline(x=best_epoch, color='r', label='best_epoch')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.savefig('/content/drive/MyDrive/NLP/Bert_' + dataset + '/figures/bert' + dataset + '_loss.png')
    plt.show()

    # epochs = range(1, num_epochs+1)
    plt.plot(history.history['precision'], 'g', label='Training precision')
    plt.plot(history.history['val_precision'], 'b', label='validation precision')
    plt.axvline(x=best_epoch, color='r', label='best_epoch')
    plt.title('Training and Validation precision')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.savefig('/content/drive/MyDrive/NLP/Bert_' + dataset + '/figures/bert' + dataset + '_precision.png')
    # plt.savefig('../Bert/Data/bert_precision.png')
    plt.show()

    plt.plot(history.history['accuracy'], 'g', label='Training accuracy')
    plt.plot(history.history['val_accuracy'], 'b', label='validation accuracy')
    plt.axvline(x=best_epoch, color='r', label='best_epoch')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.savefig('/content/drive/MyDrive/NLP/Bert_' + dataset + '/figures/bert' + dataset + '_accuracy.png')
    # plt.savefig('../Bert/Data/bert_accuracy.png')
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
