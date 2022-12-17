import os
import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def make_folder(dataset_folder):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    else:
        print("Folder {} is existed !".format(str(dataset_folder)))
    return dataset_folder


def convert_gray(img):
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    return img


def load_datasets(dataset="datasets/", max_sample=100):
    names = []
    images = []

    for folder in os.listdir(dataset):
        files = os.listdir(os.path.join(dataset, folder))[:max_sample]

        for i, file_name in enumerate(files):
            if file_name.find(".jpg") > 0:
                img = cv2.imread(filename=os.path.join(*[dataset, folder, file_name]))
                img = convert_gray(img)
                if img is not None:
                    images.append(img)
                    names.append(folder)
    return names, images


def convert_categorical(names):
    le = LabelEncoder()
    le.fit(names)
    labels = le.classes_

    name_vector = le.transform(names)
    label_name_vector = to_categorical(name_vector)
    return label_name_vector, labels


def split_dataset(images, label_name_vector, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(np.array(images, dtype=np.float32),
                                                        np.array(label_name_vector),
                                                        test_size=test_size,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


def plot_history(history):
    names = [['accuracy', 'val_accuracy'],
             ['loss', 'val_loss']]
    for name in names:
        fig1, ax_acc = plt.subplots()
        plt.plot(history.history[name[0]])
        plt.plot(history.history[name[0]])
        plt.xlabel('Epoch')
        plt.ylabel(name[0])
        plt.title('Model - ' + name[0])
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.grid()
        plt.show()

