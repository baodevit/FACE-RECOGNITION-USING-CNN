import numpy as np

import model
import utils


def training_model():
    datasets = "datasets/"
    names, images = utils.load_datasets(dataset=datasets)
    print("Dataset: {} sample / {} labels".format(len(names), len(np.unique(names))))
    print("List labels: {}".format(np.unique(names)))
training_model()
