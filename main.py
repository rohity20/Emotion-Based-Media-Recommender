from keras.models import load_model
from time import sleep
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import pandas as pd
import numpy as np
import time


face_classifier = cv2.CascadeClassifier(r'G:\Emo-Media\haarcascade_frontalface_default.xml')
classifier = load_model(r'G:\Emo-Media\model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Neutral', 'Sad', 'Surprise']
path = r'G:\Emo-Media\opencv_frame_0.png'
cap = cv2.VideoCapture(0)

Music_Player = pd.read_csv("./data_moods.csv")
Music_Player = Music_Player[['name', 'artist', 'mood', 'popularity']]


# Making Songs Recommendations Based on Predicted Class

def Recommend_Songs(pred_class):

    if(pred_class == 'Disgust'):

        Play = Music_Player[Music_Player['mood'] == 'Sad']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        # display(Play)
        # return Play
        print(Play)

    if(pred_class == 'Happy' or pred_class == 'Sad'):

        Play = Music_Player[Music_Player['mood'] == 'Happy']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        print(Play)

    if(pred_class == 'Fear' or pred_class == 'Angry'):

        Play = Music_Player[Music_Player['mood'] == 'Calm']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        # display(Play)
        print(Play)

    if(pred_class == 'Surprise' or pred_class == 'Neutral'):

        Play = Music_Player[Music_Player['mood'] == 'Energetic']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        # display(Play)
        print(Play)


final_label = ""
TIMER = int(20)
i = 1
while i<=1:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(label)
            Recommend_Songs(label)
            # time.sleep(10)
            i += 1
            # break
            # final_label = label
        else:
            cv2.putText(frame, 'No Faces', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detector', frame)
          
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
