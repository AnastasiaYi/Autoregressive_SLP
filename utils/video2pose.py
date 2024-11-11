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
    keypoints_list = [None] * len(image_files)

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

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
        pose_keypoints = [(None, None)] * 19
        pose_index = 0
        if pose_results.pose_landmarks:
            for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]:  # Specific body keypoints
                    pose_keypoints[pose_index]=(landmark.x, landmark.y)
                    pose_index+=1

        # Process hand keypoints
        hand_results = hands.process(rgb_frame)
        left_hand_keypoints = [(None, None)] * 21
        right_hand_keypoints = [(None, None)] * 21
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                hand_keypoints = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]
                label = handedness.classification[0].label
                if label == "Left":
                    left_hand_keypoints = hand_keypoints
                else:
                    right_hand_keypoints = hand_keypoints

        # Combine all keypoints
        all_keypoints = left_hand_keypoints[:21] + right_hand_keypoints[:21] + pose_keypoints
        keypoints_list.append(all_keypoints)

        if draw:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            # Draw pose landmarks on the image
            draw_pose_landmarks(out_path, mp_drawing, mp_pose, i+1, frame, pose_results, hand_results)

    # # Interpolate missing keypoints across the sequence
    # last_valid_idx = None
    # for i in range(len(keypoints_list)):
    #     if (None, None) in keypoints_list[i]:
    #         if i == 0:
            
    #         elif i == len(keypoints_list) - 1:
    #         # Interpolate if there was a missing segment
    #         if last_valid_idx is not None and last_valid_idx < i - 1:
    #             interpolate_keypoints(last_valid_idx, i, keypoints_list)
    #         last_valid_idx = i

    return keypoints_list

def interpolate_keypoints(start_idx, end_idx, keypoints_list):
    """Interpolate missing frames between two valid frames."""
    start_keypoints = keypoints_list[start_idx]
    end_keypoints = keypoints_list[end_idx]
    for j in range(start_idx + 1, end_idx):
        t = (j - start_idx) / (end_idx - start_idx)
        interpolated_keypoints = [
            ((1 - t) * start + t * end) if start and end else (None, None)
            for start, end in zip(start_keypoints, end_keypoints)
        ]
        keypoints_list[j] = interpolated_keypoints

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

if __name__ == "__main__":
    # Example usage
    folder_path = './trail_vid/'
    image_files = sorted(os.listdir(folder_path))
    keypoints = extract_keypoints_from_folder(image_files)

    # for i, frame_keypoints in enumerate(keypoints):
    #     image = cv2.imread(os.path.join(folder_path, image_files[i]))
    #     for x, y in frame_keypoints:
    #         if x is None or y is None:
    #             continue  # Skip missing points
    #         # Draw each keypoint as a small circle
    #         image = cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), radius=5, color=(0, 255, 0), thickness=-1)
    #         cv2.imwrite("./trail_pose/" + str(i) + '.png', image)
    #         print(f"drawn {i} frames")
    # Output the results
    # print("Extracted Keypoints:")
    # for i, frame_keypoints in enumerate(keypoints):
    #     print(f"Frame {i + 1}: {len(frame_keypoints)}")