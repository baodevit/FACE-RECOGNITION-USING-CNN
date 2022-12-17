import numpy as np
from model import model_cnn
import utils


def training_model(num_epoch,num_batch_size):
    datasets = "datasets/"

    names, images = utils.load_datasets(dataset=datasets)

    label_name_vector, labels = utils.convert_categorical(names)
    x_train, x_test, y_train, y_test = utils.split_dataset(images=images, label_name_vector=label_name_vector)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    input_shape = x_test[0].shape

    print("Dataset: {} sample / {} labels".format(len(names), len(np.unique(names))))
    print("List labels: {}".format(np.unique(names)))
    print("Split datasets: training set ~ {}, test set ~ {}".format(x_train.shape[0], y_train.shape[0]))
    print("Input shape: ", input_shape)

    num_label = len(labels)

    model = model_cnn(input_shape=input_shape, number_labels=int(num_label))
    history = model.fit(x=x_train,
                        y=y_train,
                        epochs=num_epoch,
                        batch_size=num_batch_size,
                        shuffle=True,
                        validation_steps=0.15)
    utils.plot_history(history=history)
    model.save("model.h5")








