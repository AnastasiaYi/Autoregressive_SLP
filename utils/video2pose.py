import cv2
import mediapipe as mp
import numpy as np
import os


def extract_keypoints_from_folder(image_files, transform = None, draw=False, out_path="./trail_pose/"):
    '''
    Employ Mediapipe to extract 61 2D keypoints (comprised of 21 for each hand, 9 for the body, and 10 for the face)
    :param folder_path: Path to the folder containing images
    :param draw: Whether to draw the pose landmarks on the images
    :param out_path: Path to save the images with pose landmarks drawn
    :return: List of keypoints for each frame
    '''
    keypoints_list = []

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5)
    hands = mp_hands.Hands(
        static_image_mode=True,
        min_detection_confidence=0.5)

    mp_drawing = mp.solutions.drawing_utils


    for i, file_name in enumerate(image_files):
        file_path = os.path.join(folder_path, file_name)
        # Read the image
        frame = cv2.imread(file_path)
        if frame is None:
            continue

        if transform:
            frame = transform(frame)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        pose_keypoints = []
        if pose_results.pose_landmarks:
            for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]:  # Specific body keypoints
                    pose_keypoints.append((landmark.x, landmark.y))

        # Process hand keypoints
        hand_results = hands.process(rgb_frame)
        left_hand_keypoints = []
        right_hand_keypoints = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_keypoints = []
                for landmark in hand_landmarks.landmark:
                    hand_keypoints.append((landmark.x, landmark.y))
                if len(left_hand_keypoints) == 0:
                    left_hand_keypoints = hand_keypoints
                else:
                    right_hand_keypoints = hand_keypoints

        # Combine all keypoints
        all_keypoints = left_hand_keypoints[:21] + right_hand_keypoints[:21] + pose_keypoints
        # Subsample frames (process every 3rd image)
        if (i + 1) % 3 == 0:
            continue

        keypoints_list.append(all_keypoints)

        if draw:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            # Draw pose landmarks on the image
            draw_pose_landmarks(out_path, mp_drawing, mp_pose, i+1, frame, pose_results, hand_results)

    return keypoints_list

def draw_pose_landmarks(out_path, mp_drawing, mp_pose, i, frame, pose_results, hand_results):
    '''
    Draws pose, hand, and face landmarks on the image.
    :param out_path: Path to save the images with pose landmarks drawn
    :param mp_drawing: MediaPipe drawing utilities
    :param mp_pose: MediaPipe pose utilities
    :param BG_COLOR: Background color for the image
    :param i: Frame number
    :param frame: Image frame
    :param pose_results: Pose landmarks
    :param hand_results: Hand landmarks
    :param face_results: Face landmarks
    '''
    BG_COLOR = (192, 192, 192)  # gray
    annotated_image = frame.copy()
    # Draw segmentation on the image.
    condition = np.stack((pose_results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(frame.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        pose_results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=1))
    
    # Draw hand landmarks on the image.
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=1))
    
    cv2.imwrite(out_path + str(i) + '.png', annotated_image)
    print(f"drawn {i} frames")

def calculate_angles(keypoints):
    '''
    Calculates the angle between three keypoints (shoulder, elbow, wrist) for each frame.
    :param keypoints: List of keypoints for each frame
    :return: List of angles for each frame
    '''
    angles = []
    # Calculate angle between three keypoints (shoulder, elbow, wrist)
    for frame_keypoints in keypoints:
        if len(frame_keypoints) >= 14:  # Ensure we have enough points
            shoulder = np.array(frame_keypoints[11])  # Left shoulder
            elbow = np.array(frame_keypoints[13])  # Left elbow
            wrist = np.array(frame_keypoints[15])  # Left wrist
            
            # Calculate angle using vector math
            vector1 = elbow - shoulder
            vector2 = wrist - elbow
            angle = np.arctan2(vector2[1], vector2[0]) - np.arctan2(vector1[1], vector1[0])
            angle = np.abs(angle * (180.0 / np.pi))  # Convert to degrees
            angles.append(angle)

    return angles


if __name__ == "__main__":
    # Example usage
    folder_path = './trail_vid/'
    # Set draw to True to visualize the pose landmarks, false to only extract keypoints
    keypoints = extract_keypoints_from_folder(folder_path, draw=True)
    angles = calculate_angles(keypoints)

    # # Output the results
    print("Extracted Keypoints:")
    for i, frame_keypoints in enumerate(keypoints):
        print(f"Frame {i + 1}: {len(frame_keypoints)}")

    # print("\nCalculated Angles:")
    # for i, angle in enumerate(angles):
    #     print(f"Frame {i + 1}: {angle:.2f} degrees")