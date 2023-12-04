import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

NUM_WORDS = 10000
NUM_CLASSES = 46  # Number of classes in the Reuters dataset
NUM_EPOCHS = 30
BATCH_SIZE = 512
VALIDATION_SPLIT = 0.2
PATIENCE = 3

def vectorize_sequences(sequences, dimension=NUM_WORDS):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def to_one_hot(labels, dimension=NUM_CLASSES):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(NUM_WORDS,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    return model

def train(modelname):
    model = modelname

    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=NUM_WORDS)
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    y_train = to_one_hot(train_labels)
    y_test = to_one_hot(test_labels)

    # Pad sequences to have consistent lengths
    x_train = pad_sequences(x_train, maxlen=NUM_WORDS)
    x_test = pad_sequences(x_test, maxlen=NUM_WORDS)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)
    history = model.fit(x_train, y_train,
                        epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_split=VALIDATION_SPLIT,
                        callbacks=[callback])

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'r-', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b--', label='Validation loss')
    plt.title('Training vs. Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    model.save('Omar_Rabbah_Reuters')

def evaluate():
    from keras.models import load_model

    loaded_model = load_model('Omar_Rabbah_Reuters')

    (_, _), (x_test, y_test) = reuters.load_data(num_words=NUM_WORDS)
    x_test = vectorize_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=NUM_WORDS)
    y_test = to_one_hot(y_test)

    # Make predictions on the test set
    predictions = loaded_model.predict(x_test)

    # Convert predictions to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Print the first few predictions and actual labels
    for i in range(20):
        print(f"Sample {i + 1}: Predicted={predicted_labels[i]}, Actual={np.argmax(y_test[i])}")

    # Evaluate the model on the test set
    test_loss, test_accuracy = loaded_model.evaluate(x_test, y_test)

    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    print(f'Test loss: {test_loss * 100:.2f}%')


