import mediapipe as mp

def initialize_mediapipe_pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=static_image_mode,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence)
    return pose
