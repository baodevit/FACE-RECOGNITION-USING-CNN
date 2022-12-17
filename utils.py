import os


def make_folder(dataset_folder):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    else:
        print("Folder {} is existed !".format(str(dataset_folder)))
    return dataset_folder
