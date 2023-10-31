import gzip
import numpy as np
import os
import pickle

class MnistData():   
    image_size = 784  # 28x28
    image_dim = (1, 28, 28)
    train_num = 60000
    test_num  = 10000
    key_file = {
        'train_images': 'mnist/train-images-idx3-ubyte.gz',
        'train_labels': 'mnist/train-labels-idx1-ubyte.gz',
        'test_images':  'mnist/t10k-images-idx3-ubyte.gz',
        'test_labels':  'mnist/t10k-labels-idx1-ubyte.gz'
    }

    def __init__(self):
        pass
    
    def _load_images(self, file_name):
        with gzip.open(file_name, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)
        images = images.reshape(-1, self.image_size)

        print('Done with loading images: ', file_name)    
        return images


    def _load_labels(self, file_name):
        with gzip.open(file_name, 'rb') as f: 
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        
        print('Done with loading labels: ', file_name)    
        return labels
 
    
    def _change_one_hot_label(self, x):
        t = np.zeros((x.size, 10))
        for idx, row in enumerate(t):
            row[x[idx]] = 1

        return t
    

    def load(self, normalize=True, flatten=True, one_hot_label=True):
        dataset = {}
        dataset['train_images'] = self._load_images(self.key_file['train_images'])
        dataset['train_labels'] = self._load_labels(self.key_file['train_labels'])
        dataset['test_images']  = self._load_images(self.key_file['test_images'])
        dataset['test_labels']  = self._load_labels(self.key_file['test_labels'])
        
        if normalize:
            for key in ('train_images', 'test_images'):
                dataset[key] = dataset[key].astype(np.float32)
                dataset[key] /= 255.0
    
        if one_hot_label:
            dataset['train_labels'] = self._change_one_hot_label(dataset['train_labels'])
            dataset['test_labels'] = self._change_one_hot_label(dataset['test_labels'])
    
        if not flatten:
             for key in ('train_images', 'test_images'):
                dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    
        return (dataset['train_images'], dataset['train_labels']), \
                (dataset['test_images'], dataset['test_labels'])