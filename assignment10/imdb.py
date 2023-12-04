import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.datasets import imdb

NUM_WORDS=10000
NUM_EPOCHS=30
BATCH_SIZE=512
VALIDATION_SPLIT=0.2
PATIENCE=3


def vectorize_sequences(sequences):
    results = np.zeros((len(sequences),NUM_WORDS))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results




def build_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation="sigmoid"))  
    return model





def train(modelname):
#complie 
    model = modelname 

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS)
    x_train = vectorize_sequences(train_data) 
    x_test  = vectorize_sequences(test_data) 

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])    

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)
    history = model.fit(x_train, y_train, epochs=NUM_EPOCHS, 
                        batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, 
                        callbacks=[callback])

    history_dict = history.history
    history_dict.keys()
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'r-', label='training loss')
    plt.plot(epochs, val_loss_values, 'b--', label='validation loss')
    plt.title('training vs. validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    plt.show()

    model.save('Omar_Rabbah_imdb')

def evaluate():
    from keras.models import load_model

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS)

    x_test  = vectorize_sequences(test_data) 

    y_test = np.asarray(test_labels).astype('float32')

    modd = load_model('Omar_Rabbah_imdb')


    # Make predictions on the test set
    predictions = modd.predict(x_test)

    # Convert predictions to binary labels (0 or 1)
    predicted_labels = np.round(predictions).flatten()

    # Print the first few predictions and actual labels
    for i in range(20):
        print(f"Sample {i + 1}: Predicted={predicted_labels[i]}, Actual={y_test[i]}")

    loss, accuracy = modd.evaluate(x_test, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Test loss: {loss * 100:.2f}%')
