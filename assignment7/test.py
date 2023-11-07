from keras.models import load_model
import Mnist_keras


(train_images , train_labels) , (test_images, test_labels) = Mnist_keras.load_data()

model = load_model('model_Omar_Rabbah.h5')



# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_images, test_labels)
print("Test loss:", loss)
print("Test accuracy:", accuracy)