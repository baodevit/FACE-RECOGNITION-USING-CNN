import data_collection
import train
import predict


def input_name():
    name_person = input("Enter your name: ")
    directory = "datasets/" + name_person

    data_collection.get_data(name_person=name_person, dataset_folder=directory)
    print("Data collection is completed !")


def training_model():
    num_epoch = int(input("Enter number epochs: "))
    num_batch_size = int(input("Enter number batch_size: "))
    train.training_model(num_epoch=num_epoch, num_batch_size=num_batch_size)


def test_model():
    predict.predict()


if __name__ == '__main__':

    # input_name()
    #training_model()q
    test_model()
