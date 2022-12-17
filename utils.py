import os
import cv2
import matplotlib as plt


def make_folder(dataset_folder):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    else:
        print("Folder {} is existed !".format(str(dataset_folder)))
    return dataset_folder


def load_datasets(dataset="datasets/", max_sample=100):
    names = []
    images = []

    for folder in os.listdir(dataset):
        files = os.listdir(os.path.join(dataset, folder))[:max_sample]
        names.append(folder)

        for file in files:
            images.append(file)

    print("Count images: ", len(names))
    print("Count images: ", len(images))
    return names, images


def detector_face():
    img = cv2.imread("datasets/baonh/baonh.1.jpg")
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(image=img, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img=img,
                      pt1=(x, y),
                      pt2=(x + w, y + h),
                      color=(255, 0, 0),
                      thickness=2)
    img = cv2.resize(img, (600, 600))
    cv2.imshow(winname="Camera", mat=img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
