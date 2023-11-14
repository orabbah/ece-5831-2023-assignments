import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os
import shutil
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import image


#model.compile(optimizer = "rmsprop",
#            loss = "sparse_categorical_crossentropy",
#            metrics = ["accuracy"])

#model.fit(train_images , train_labels, epochs =100, batch_size = 64)
#model.save('model_Omar_Rabbah.h5')




def build_model():
    input = keras.Input(shape=(180, 180, 3))
    x = layers.Rescaling(1./255)(input)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=input, outputs=outputs)
    return model

def train(modelname):
    #complie 
    model = build_model() 
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    #preparing dataset
    #data_filename = "dogs-vs-cats.zip"
    data_from_kaggle = "data-from-kaggle/train"
    #data_from_kaggle_test = "data-from-kaggle/test1"
    data_dirname = "dogs-vs-cats"

    def make_dataset(subset_name, start_idx, end_idx):
        for category in { "cat", "dog" }:
            # data_dirname/subset_name/categoroy 
            dir = f"{data_dirname}/{subset_name}/{category}"
            # print(dir)
            os.makedirs(dir)
            fnames = [f"{category}.{i}.jpg" for i in range(start_idx, end_idx)]
            # print(fnames)
            for fname in fnames: 
                    shutil.copyfile(src=f"{data_from_kaggle}/{fname}", dst=f"{dir}/{fname}") 


    # 
    try:
        make_dataset("train", 0, 1000)
        make_dataset("validation", 7501, 8001)
        # 
        make_dataset("test", 10001, 11001)
    except:
         print('Dataset already exists')


    def fit():
        batch_size = 32
        model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

        train_dataset = image_dataset_from_directory(f"{data_dirname}/train", image_size=(180, 180), batch_size=batch_size)
        validation_dataset = image_dataset_from_directory(f"{data_dirname}/validation", image_size=(180, 180), batch_size=batch_size)
        test_dataset = image_dataset_from_directory(f"{data_dirname}/test", image_size=(180, 180), batch_size=batch_size)

        callbacks = [ keras.callbacks.ModelCheckpoint(
            filepath="Omar-Rabbah-from-scrach",
            save_best_only=True,
            monitor="val_loss"
        )]


        model.fit(train_dataset, validation_data=validation_dataset, epochs=3000, callbacks=callbacks)
        model.save(modelname)

    fit() ; 


def predict(modelname, imagename):

    loaded_model = keras.models.load_model(modelname)

    img = image.load_img(imagename, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  


    predictions = loaded_model.predict(img_array)

    print("Prediction:", predictions[0][0])
    if(predictions> 0.5):
        print('Predicted as a Dog')
    else:
        print('Predicted as a Cat') 


