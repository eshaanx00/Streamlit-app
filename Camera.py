import cv2
import streamlit as st
import mediapipe as mp

st.title("Webcam Application")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(0)

detection_confidence = st.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.75)
tracking_confidence = st.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.75)
max_num_hands_Mp = st.slider('Min Tracking Confidence', min_value = 1,max_value = 4,value = 2)

mpHands = mp.solutions.hands
Hands = mpHands.Hands(min_detection_confidence=detection_confidence,min_tracking_confidence=tracking_confidence,max_num_hands=max_num_hands_Mp)
mpDraw = mp.solutions.drawing_utils


while run:
    ret, frame = cam.read()
    results = Hands.process(frame)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')
