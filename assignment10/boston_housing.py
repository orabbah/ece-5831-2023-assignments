from keras import models
from keras import layers
import matplotlib.pyplot as plt

from keras.datasets import boston_housing   

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()


def normalize_data(data):
    mean = data.mean(axis=0)
    data -= mean
    std = data.std(axis=0)
    data /= std
    return data


x_data = normalize_data(train_data)
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(x_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model    


def train(modelname):
    model = modelname
    
    history = model.fit(x_data, train_labels, validation_split=0.2, batch_size=1, epochs=200)



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
    model.save('Omar_Rabbah_Boston')

def evaluate():
    from keras.models import load_model
    from keras.datasets import boston_housing
    import numpy as np

    # Load the saved model
    loaded_model = load_model('Omar_Rabbah_Boston')

    # Load the Boston Housing test data
    (_, _), (x_test, y_test) = boston_housing.load_data()

    # Normalize the test data using the same mean and std as the training data
    mean = x_test.mean(axis=0)
    std = x_test.std(axis=0)
    x_test = (x_test - mean) / std

    # Evaluate the model on the test set
    predictions = loaded_model.predict(x_test).flatten()
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values on Test Set')
    plt.show()

    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions - y_test))
    print(f'Test Mean Absolute Error (MAE): {mae:.2f}')
