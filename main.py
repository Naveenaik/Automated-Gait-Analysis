import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, 
                    min_detection_confidence=0.7, 
                    min_tracking_confidence=0.7)


def extract_keypoints(landmarks, image_shape):
    h, w = image_shape[:2]
    keypoints = {}

    head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y * h
    left_foot_y = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * h
    right_foot_y = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * h
    height = abs(max(left_foot_y, right_foot_y) - head_y)

    left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w
    right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w
    width = abs(right_shoulder_x - left_shoulder_x)

    keypoints['left_foot_y'] = left_foot_y
    keypoints['right_foot_y'] = right_foot_y
    keypoints['left_foot_speed'] = 0  
    keypoints['right_foot_speed'] = 0  

    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h
    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h
    keypoints['left_knee_angle'] = abs(left_foot_y - left_knee_y)
    keypoints['right_knee_angle'] = abs(right_foot_y - right_knee_y)

    keypoints['height'] = height
    keypoints['width'] = width
    keypoints['height_width_ratio'] = height / width if width > 0 else 0

    return keypoints


def draw_skeleton(image, landmarks):
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=5),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=5)
    )


def process_video(video_path, person_name, output_dir):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    prev_left_foot_y = prev_right_foot_y = None
    frame_num = 0

    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            keypoints = extract_keypoints(results.pose_landmarks.landmark, frame.shape)

            if prev_left_foot_y is not None and prev_right_foot_y is not None:
                keypoints['left_foot_speed'] = abs(keypoints['left_foot_y'] - prev_left_foot_y)
                keypoints['right_foot_speed'] = abs(keypoints['right_foot_y'] - prev_right_foot_y)

            prev_left_foot_y = keypoints['left_foot_y']
            prev_right_foot_y = keypoints['right_foot_y']

            keypoints['frame'] = frame_num
            keypoints['person_name'] = person_name
            keypoints_list.append(keypoints)

            draw_skeleton(frame, results.pose_landmarks)

            frame_filename = os.path.join(output_dir, f"{person_name}_frame_{frame_num}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame_num += 1

    cap.release()
    return keypoints_list

def process_videos_in_directory(directory_path, output_csv, output_frames_dir):
    all_keypoints = []

    for video_file in os.listdir(directory_path):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(directory_path, video_file)
            person_name = os.path.splitext(video_file)[0]
            print(f"Processing {video_file}...")
            person_frames_dir = os.path.join(output_frames_dir, person_name)
            keypoints_list = process_video(video_path, person_name, person_frames_dir)
            if keypoints_list:
                all_keypoints.extend(keypoints_list)

    df = pd.DataFrame(all_keypoints)
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")


video_directory = 'data' 
output_csv = 'output_gait_parameters_per_frame.csv' 
output_frames_dir = 'skeleton_frames' 

process_videos_in_directory(video_directory, output_csv, output_frames_dir)
