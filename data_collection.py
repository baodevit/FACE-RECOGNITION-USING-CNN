import cv2
import utils


def get_data(name_person, dataset_folder):
    utils.make_folder(str(dataset_folder))

    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    camera = cv2.VideoCapture(0)
    img_size = (100, 100)
    max_sample = 0
    while (True):
        ret, frame = camera.read()
        img_gray = cv2.cvtColor(src=frame, code=cv2.COLOR_RGBA2GRAY)

        faces = face_detector.detectMultiScale(image=img_gray,
                                               scaleFactor=1.1,
                                               minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=frame,
                          pt1=(x, y),
                          pt2=(x + w, y + h),
                          color=(255, 0, 255),
                          thickness=2)
            img_resize = cv2.resize(src=img_gray[y:y + h, x:x + w],
                                    dsize=img_size,
                                    interpolation=cv2.INTER_AREA)
            max_sample += 1
            cv2.imwrite(filename="datasets/" + str(name_person) + "/" + str(name_person) + "." + str(max_sample) + ".jpg", img=img_resize)
        cv2.imshow(winname="Camera", mat=frame)
        cv2.waitKey(10)

        if max_sample >= 140:
            break


