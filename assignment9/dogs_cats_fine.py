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
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt


from keras.applications import VGG16


batch_size = 20

class dogs_cats_fine():

    def build_model():
        conv_base = VGG16(weights='imagenet',
                include_top=False,
                input_shape=(180, 180, 3))
        
        conv_base.trainable = True
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True    
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
        
        model = models.Sequential()
        model.add(conv_base,)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        #model = keras.Model(inputs=input, outputs=outputs)    
        return model

    

    def train(modelname):
    #complie 
        model = modelname 

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
            make_dataset("test", 10001, 11001)
        except:
             print('Dataset already exists')

        conv_base = VGG16(weights='imagenet',
                include_top=False,
                input_shape=(180, 180, 3))

        def extract_features(directory, sample_count):
            features = np.zeros(shape=(sample_count, 5, 5, 512))
            labels = np.zeros(shape=(sample_count))
            generator = datagen.flow_from_directory(
            directory,
            target_size=(180, 180),
            batch_size=batch_size,
            class_mode='binary')
            i=0
            for inputs_batch, labels_batch in generator:
                features_batch = conv_base.predict(inputs_batch)
                features[i * batch_size : (i + 1) * batch_size] = features_batch
                labels[i * batch_size : (i + 1) * batch_size] = labels_batch
                i += 1
                if i * batch_size >= sample_count:
                    break
            return features, labels
        
        base_dir = "dogs-vs-cats"
        train_dir = os.path.join(base_dir, 'train')
        validation_dir = os.path.join(base_dir, 'validation')
        test_dir = os.path.join(base_dir, 'test')
        datagen = ImageDataGenerator(rescale=1./255)
        
        train_features, train_labels = extract_features(train_dir, 2000)
        validation_features, validation_labels = extract_features(validation_dir, 1000)
        test_features, test_labels = extract_features(test_dir, 1000)

        train_features = np.reshape(train_features, (2000, 5*5* 512))
        validation_features = np.reshape(validation_features, (1000, 5*5* 512))
        test_features = np.reshape(test_features, (1000, 5*5* 512))


        def fit():
             train_datagen = ImageDataGenerator(rescale=1./255,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')
             test_datagen = ImageDataGenerator(rescale=1./255)
             
             train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(180, 180),
                    batch_size=20,
                    class_mode='binary')
             validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(180, 180),
                    batch_size=20,
                    class_mode='binary')
             model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(learning_rate=3e-5),
                    metrics=['acc'])
             
             history = model.fit(
                train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(180, 180),
                    batch_size=20,
                    class_mode='binary'),
                steps_per_epoch=100,
                epochs=11,
                validation_data=validation_generator,
                validation_steps=50)
             
             model.save("Omar_Rabbah_fine")
             
             acc = history.history['acc']
             val_acc = history.history['val_acc']    
             loss = history.history['loss']
             val_loss = history.history['val_loss']
             epochs = range(1, len(acc) + 1)
             plt.plot(epochs, acc, 'bo', label='Training acc')
             plt.plot(epochs, val_acc, 'b', label='Validation acc')
             plt.title('Training and validation accuracy')
             plt.legend()
             plt.figure()
             plt.plot(epochs, loss, 'bo', label='Training loss')
             plt.plot(epochs, val_loss, 'b', label='Validation loss')
             plt.title('Training and validation loss')
             plt.legend()
             plt.show()

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
