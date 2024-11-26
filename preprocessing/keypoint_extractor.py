import os
import argparse
import cv2
from deepface import DeepFace
import face_alignment
import numpy as np
from tqdm import tqdm
import sys


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        sys.exit(1)

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total_frames), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


def detect_faces(frames, detector_backend='ssd'):
    facial_areas = []
    for frame in tqdm(frames, desc="Detecting faces"):
        try:
            faces = DeepFace.extract_faces(
                img_path=None,
                img=frame,
                detector_backend=detector_backend,
                align=False,
                enforce_detection=False
            )
            areas = [face['facial_area'] for face in faces]
            facial_areas.append(areas)
        except Exception:
            facial_areas.append([])
    return facial_areas


def extract_keypoints(frames, facial_areas, detector_backend='sfd', landmarks_type='2D', device='cuda'):
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D if landmarks_type.upper() == "2D" else face_alignment.LandmarksType.THREE_D,
        device=device,
        face_detector=detector_backend
    )
    keypoints = []
    for idx in tqdm(range(len(frames)), desc="Extracting keypoints"):
        frame = frames[idx]
        faces = facial_areas[idx]
        frame_keypoints = []
        if not faces:
            preds = fa.get_landmarks(frame)
            frame_keypoints.append(preds[0] if preds else None)
        else:
            for face_roi in faces:
                x1, y1, w, h = face_roi['x'], face_roi['y'], face_roi['w'], face_roi['h']
                bounding_box = [x1, y1, x1 + w, y1 + h]
                preds = fa.get_landmarks(frame, detected_faces=[bounding_box])
                frame_keypoints.append(preds[0] if preds else None)
        keypoints.append(frame_keypoints)
    return keypoints


def save_keypoints(keypoints, output_path):
    # Convert list of lists to (num_frames, 68, 2), setting None to NaNs
    processed_keypoints = []
    for kp in keypoints:
        if len(kp) > 0 and kp[0] is not None:
            processed_keypoints.append(kp[0])
        else:
            # If no face detected, fill with NaNs
            processed_keypoints.append(np.full((68, 2), np.nan))
    np.save(output_path, np.array(processed_keypoints))
    print(f"Keypoints saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect faces and extract keypoints from a video.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file (.avi).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output.")
    args = parser.parse_args()

    video_path = args.video_path
    output_dir = args.output_dir

    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    frames = extract_frames(video_path)
    print(f"Total frames extracted: {len(frames)}")

    facial_areas = detect_faces(frames)
    keypoints = extract_keypoints(frames, facial_areas)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    keypoints_output_path = os.path.join(output_dir, f"{video_name}.npy")
    save_keypoints(keypoints, keypoints_output_path)

    print("Keypoint extraction completed successfully.")


if __name__ == "__main__":
    main()

"""
Usage:
    python keypoint_extractor.py --video_path /path/to/video.avi --output_dir /path/to/output
"""
