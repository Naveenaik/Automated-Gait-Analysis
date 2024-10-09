import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, 
                    min_detection_confidence=0.7, 
                    min_tracking_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

model = load_model('gait_model.h5')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

def extract_keypoints(landmarks, image_shape):
    h, w = image_shape[:2]
    keypoints = {}

    head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y * h
    left_foot_y = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * h
    right_foot_y = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * h
    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h
    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h

    left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w
    right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w
    height = abs(max(left_foot_y, right_foot_y) - head_y)
    width = abs(right_shoulder_x - left_shoulder_x)

    left_foot_speed = 0
    right_foot_speed = 0

    keypoints['height'] = height
    keypoints['width'] = width
    keypoints['height_width_ratio'] = height / width if width > 0 else 0
    keypoints['left_knee_angle'] = abs(left_foot_y - left_knee_y)
    keypoints['right_knee_angle'] = abs(right_foot_y - right_knee_y)
    keypoints['left_foot_speed'] = left_foot_speed
    keypoints['right_foot_speed'] = right_foot_speed

    return keypoints

def predict_person_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    window_name= 'Output'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  
    cv2.resizeWindow(window_name, 320, 480)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            keypoints = extract_keypoints(results.pose_landmarks.landmark, frame.shape)
            features = np.array([[keypoints['height'], keypoints['width'], keypoints['height_width_ratio'], 
                                  keypoints['left_foot_speed'], keypoints['right_foot_speed'], 
                                  keypoints['left_knee_angle'], keypoints['right_knee_angle']]])
            
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)
            predicted_class = np.argmax(prediction, axis=1)
            predicted_name = label_encoder.inverse_transform(predicted_class)
            
            predictions.append(predicted_name[0])

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=5),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=5))

            cv2.putText(frame, f'Person: {predicted_name[0]}', (50, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 128), 10, cv2.LINE_AA)
            
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return predictions

video_path = 'data/himmath.mp4'
predicted_person = predict_person_from_video(video_path)
print(f"Predicted Person: {predicted_person}")
