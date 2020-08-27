import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


class TrainingModel:

    def TrainingModel(self, path):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier(
            'data/haarcascades/haarcascade_frontalface_default.xml')        
        print('training faces')
        #get the image label
        faces, ids = self.getImageAndLabels(path)        
        recognizer.train(faces, np.array(ids))
        recognizer.write('Trained_Models/face_trainer.yml')
        print('finishing')

    def getImageAndLabels(self, path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []
            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')
                id = int(os.path.split(imagePath)[-1].split('_')[1])
                faceSamples.append(img_numpy)
                ids.append(id)
            return faceSamples, ids

    
