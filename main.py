import data_collection


def input_name():
    name_person = input("Enter your name: ")
    directory = "datasets/" + name_person

    data_collection.get_data(name_person=name_person, dataset_folder=directory)
    print("Data collection is completed !")


if __name__ == '__main__':
    input_name()