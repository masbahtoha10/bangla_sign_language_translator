from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
# import time
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
# from keras.models import load_model

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
def mediapipe_detection(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    img.flags.writeable = False                  # Imgimg is no longer writeable
    results = model.process(img)                 # Make prediction
    img.flags.writeable = True                   # Imgimg is now writeable 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return img, results
def draw_landmarks(img, results):
    # mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
def draw_styled_landmarks(img, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])
# Path for exported data, numpy arrays
# DATA_PATH = os.path.join('M:\Handsign\ActionDetectionforSignLanguage-main\MP_Data') 

# lIST OF CLASS
# actions = np.array(['nam', 'bolo','flower','kokhono_na','amra','bondhu','tumi','kripon','valo','khub_shundor'])
actions = np.array(['Nygv‡bv','gv','evev','e›`yK','`vI','e¨vqvg','Mvwo','dzj','bv','Avgiv','eÜy','Zzwg','K¨v‡giv','wegvb','hy×','‡Uwj‡dvb','mg_©b','bvgvh','fvj','my›`i','bvg'])
# actions = np.array(['ঘুমানো','মা','বাবা', 'বন্দুক','দাও', 'ব্যায়াম','গাড়ি', 'ফুল','না','আমরা','বন্ধু','তুমি', 'ক্যামেরা', 'বিমান', 'যুদ্ধ','টেলিফোন','সমর্থন','নামায','ভাল','সুন্দর', 'নাম'])
#trained Model Import
model = tf.keras.models.load_model('M:\Handsign\ActionDetectionforSignLanguage-main\V_word\draft2.h5',compile=False)
colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame
# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.9

cap = cv2.VideoCapture(1)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Read feeds
        ret, frame = cap.read()

        # Make detections
        img, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        # draw_styled_landmarks(img, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 7: 
                sentence = sentence[-7:]

            # Viz probabilities
            # img = prob_viz(res, actions, img, colors)
            
        # cv2.rectangle(img, (0,405), (720,520), (0, 2, 2), -1)
        # cv2.putText(img, ' '.join(sentence), (3,440), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(img, (0,1080), (1280,880), (0, 2, 2), -1)
        font = ImageFont.truetype("C:\WINDOWS\FONTS\Siyam Rupali ANSI.ttf", 32)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        # draw.text((100, 880),  ' '.join(sentence), font = font, fill = (20, 255, 255, 0))
        draw.text((10, 410),  ' '.join(sentence), font = font, fill = (20, 255, 255, 0))
        img = np.array(img_pil)
        
        # Show to screen
        # cv2.resizeWindow("OpenCV Feed", 400, 300)
        cv2.imshow('OpenCV Feed', img)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()