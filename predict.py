import cv2
import os
from keras.models import load_model


def predict():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    dataset = "datasets/"

    labels = []
    for folder in os.listdir(dataset):
        labels.append(folder)
    model = load_model("model.h5")
    camera = cv2.VideoCapture(0)
    while (True):

        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=frame,
                          pt1=(x, y),
                          pt2=(x + w, y + h),
                          color=(255, 0, 255),
                          thickness=2)
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (100, 100))
            face_img = face_img.reshape(1, 100, 100, 1)
            result = model.predict(face_img)
            idx = result.argmax(axis=1)[0]
            confidence = result.max(axis=1) * 100
            if confidence > 50:
                label_text = "%s" % labels[idx]
                cv2.putText(img=frame,
                            text=label_text,
                            org=(x + 10, y + h + 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0, 0, 255),
                            thickness=2)
            else:
                label_text = "Unknown"
                cv2.putText(img=frame,
                            text=label_text,
                            org=(x + 10, y + h + 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0, 0, 255),
                            thickness=2)
            print(label_text, confidence)
        cv2.imshow(winname="Camera", mat=frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

