import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import time
from utils import *
#from keras.preprocessing.image import ImageDataGenerator


class CollectImageData:

    #this function will extract face images from a given folder and store them in a given path       
    def ExtractAndSaveFacesFromImages(self, pathToOpen, pathToWrite):
        face_id = number_input('please input a number for the face ID => ')
        count = 0
        detector = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_default.xml')
        imagePaths = [os.path.join(pathToOpen, f) for f in os.listdir(pathToOpen)]
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath)
            img = self.TransformImage(PIL_img)        
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = detector.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(300, 300))
            for (x, y, w, h) in faces:
                count +=1             
                #resize the image to a descent size and save into the datasets folder
                face_img = img_gray[y:y+h, x:x+w]
                fsize = (156, 156)
                resized_face_img = cv2.resize(face_img, fsize)
                cv2.imwrite(pathToWrite +'/user_' + str(face_id) + '_' + str(count) + '.jpg', resized_face_img)
            
    def TransformImage(self, image):
        img_numpy = np.array(image, 'uint8')
        #adding randomness to the image
        image_gen = ImageDataGenerator(rotation_range=15, 
                               width_shift_range=0.1, 
                               height_shift_range=0.1, 
                               rescale=1.2, 
                               shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              fill_mode='nearest')
        img = image_gen.random_transform(img_numpy)
        return img
    
    #this method will extract face images from camera and store it in a given path
    def ExtractAndSaveFacesFromCamera(self, pathToWrite):
        cam = cv2.VideoCapture(0+cv2.CAP_DSHOW)
        cam.set(3, 640)
        cam.set(4, 480)
        face_detector = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
        face_id = input('/n enter user id end press <return> ==>')
        print('/n [info] Initializing face capture, look the camera and wait ---')

        #Initialize individual smapling face count and start time
        count= 0
        tPrevious = time.time()
        while(True):
            ret, img = cam.read()     
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)    
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('image', img) 
            for (x, y, w, h) in faces: 
                if (time.time() - tPrevious > 0.5):                
                    #save the captured image into the datasets folder
                    cv2.imwrite(pathToWrite + '/user_' + str(face_id) + '_' + str(count) + '.jpg', gray_img[y:y+h, x:x+w]) 
                    tPrevious = time.time()
                    count += 1
                    print('image{0}collected'.format(count))                    
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            if count > 60:
                break

        print('exiting program')
        cam.release()
        cv2.destroyAllWindows()  

    
