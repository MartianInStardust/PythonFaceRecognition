from collect_training_data import CollectImageData
from train_model import TrainingModel
from utils import *
import cv2

if(yes_or_no("Do you want to collect training data?")):
    fc = CollectImageData()

    fc.ExtractAndSaveFacesFromCamera('dataset')

if(yes_or_no('Do you want to retrain the model?')):
    tm = TrainingModel()

    tm.TrainingModel('dataset')

if(yes_or_no('Is it ready to recognize faces?')):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('Trained_Models/face_trainer.yml')

    cascadePath = 'DATA/haarcascades/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX


    names = ['unkown','Jin']

    cam = cv2.VideoCapture(0+cv2.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 480)

    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:
        ret, img = cam.read()

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray_img,
                                            scaleFactor=1.2,
                                            minNeighbors=15,
                                            minSize=(int(minW), int(minH)))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray_img[y:y+h, x:x+w])

            if(confidence < 50):
                id = names[id]
                confidence = " {0}%".format(round(100.0-confidence))
            else:
                id = 'unknown'
                confidence = " {0}%".format(round(100.0-confidence))
            cv2.putText(
                img,
                str(id),
                (x+5, y-5),
                font,
                1,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                img,
                str(confidence),
                (x+5, y+h-5),
                font,
                1,
                (255, 255, 0),
                1,
            )
        cv2.imshow('camera', img)
        k = cv2.waitKey(1) & 0Xff
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()







    



