import numpy as np

import model
import utils


def training_model():
    datasets = "datasets/"

    names, images = utils.load_datasets(dataset=datasets)

    label_name_vector, labels = utils.convert_categorical(names)
    x_train, x_test, y_train, y_test = utils.split_dataset(images=images, label_name_vector=label_name_vector, test_size=0.2)

    print("Dataset: {} sample / {} labels".format(len(names), len(np.unique(names))))
    print("List labels: {}".format(np.unique(names)))
    print("Split datasets: training set ~ {}, test set ~ {}".format(x_train.shape[0], y_train.shape[0]))


