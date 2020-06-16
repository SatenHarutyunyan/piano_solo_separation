import os
import glob
import numpy as np


class DataLoader:

    def __init__(self, train_features_dir, val_features_dir,# test_features_dir,
                 train_batch_size, val_batch_size):

        self.train_paths = glob.glob(os.path.join(train_features_dir, "**/*.npy"), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_features_dir, "**/*.npy"), recursive=True)
        # self.test_paths = glob.glob(os.path.join(test_features_dir, "**/*.wav"), recursive=True)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size


    def get_input_and_label(self, path):
        label_path = path.replace("\\inputs\\", "\\labels\\")
        input = np.load(path, allow_pickle=True)
        label = np.load(label_path, allow_pickle=True)
        return input, label**2 / (input*22 + label**2+ 1e-7)

    def batch_data_loader(self, batch_size, file_paths, index):
        batch = file_paths[index * batch_size : (index+1) * batch_size ]
        inputs = []
        labels = []
        for path in batch:
            input, label = self.get_input_and_label(path)
            inputs.append(input)
            labels.append(label)
        return inputs, labels

    def train_data_loader(self, index):
        return self.batch_data_loader(self.train_batch_size, self.train_paths, index)

    def val_data_loader(self, index):
        return self.batch_data_loader(self.val_batch_size, self.val_paths, index)

    def test_data_loader(self, index):
        return self.batch_data_loader(self.test_batch_size, self.test_paths, index)

    # def get_params(self):
    #     pass #d return the sequence length and feature shape for defining shapes of the placeholders in BaseNN.py


