from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation


def model_cnn(input_shape, number_labels):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="valid", activation="relu", input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="valid", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))

    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=number_labels))
    model.add(Activation("softmax"))

    model.summary()

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    return model
