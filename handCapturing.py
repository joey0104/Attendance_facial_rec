import cv2
import numpy as np
import os

#getting people
people=[]
folder_path="/Users/joeychiu/Documents/facialRec"
for i in os.listdir(folder_path):
    if(i !=".DS_Store"):
        people.append(i)

#haar cascade setup
haar_cascade=cv2.CascadeClassifier("haarCasCadeFrontalFace.xml")

face_imgs=[]
labels=[]

#train from dataset
for person in people:
    person_path= os.path.join(folder_path, person)
    #get person's images
    for img in os.listdir(person_path):
        img_path = os.path.join(person_path, img)
        if ".DS_Store" not in img_path:
            image = cv2.imread(img_path)
            cv2.imshow("image4", image)
            cv2.waitKey(2000)
            #convert image to gray scale for facial-detection mainly using edges
            gray_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #detect face in the gray img and minimize the sensitiveness of noise
            detected_face=haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=7)
            #only accepting pic that's been detected one face on
            if len(detected_face)==1:
                for (x,y,w,h) in detected_face:
                    face_area=gray_img[y:y+h, x:x+w]
                    face_imgs.append(face_area)
                    labels.append(people.index(person))

#convert to numpy
labels_np=np.array(labels)
face_np=np.array(face_imgs, dtype='object')

#create face recognizer
face_rec=cv2.face.LBPHFaceRecognizer_create()
face_rec.train(face_np, names_np)

#save all the data
face_rec.save('trained_face.yml')
np.save('faces_np', face_np)
np.save('labels_np', labels_np)