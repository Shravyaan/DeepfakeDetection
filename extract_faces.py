# extract_faces_simple.py
import os
import cv2
from tqdm import tqdm
import numpy as np

print("🎭 Starting Face Extraction with OpenCV")
print("=" * 50)

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_faces_from_video(video_path, output_dir, frames_per_video=5):
    """Extract faces from a video file using OpenCV"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return 0
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print(f"❌ Empty video: {video_path}")
            return 0
        
        # Calculate frame interval
        frame_interval = max(1, total_frames // frames_per_video)
        faces_extracted = 0
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for i, (x, y, w, h) in enumerate(faces):
                # Add padding
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size == 0:
                    continue
                
                # Resize to standard size
                face_img = cv2.resize(face_img, (224, 224))
                
                # Save face image
                output_path = os.path.join(
                    output_dir, 
                    f"{video_name}_frame{frame_idx:04d}_face{i}.jpg"
                )
                success = cv2.imwrite(output_path, face_img)
                if success:
                    faces_extracted += 1
                else:
                    print(f"❌ Failed to save image: {output_path}")
            
            # Stop if we have enough faces
            if faces_extracted >= frames_per_video:
                break
        
        cap.release()
        return faces_extracted
        
    except Exception as e:
        print(f"❌ Error processing {video_path}: {e}")
        return 0

def main():
    # Create output directory
    output_base = "extracted_faces"
    os.makedirs(output_base, exist_ok=True)
    
    # Define dataset structure - process only REAL and one FAKE type for testing
    dataset_dirs = {
        "real": "data/FaceForensics++_C23/original",
        "fake_Deepfakes": "data/FaceForensics++_C23/Deepfakes",
        # Comment out others for faster testing
        # "fake_Face2Face": "data/FaceForensics++_C23/Face2Face",
        # "fake_FaceShifter": "data/FaceForensics++_C23/FaceShifter", 
        # "fake_FaceSwap": "data/FaceForensics++_C23/FaceSwap",
        # "fake_NeuralTextures": "data/FaceForensics++_C23/NeuralTextures",
        # "fake_DeepFakeDetection": "data/FaceForensics++_C23/DeepFakeDetection",
    }
    
    total_faces = 0
    
    for label, video_dir in dataset_dirs.items():
        if not os.path.exists(video_dir):
            print(f"❌ Directory not found: {video_dir}")
            continue
        
        print(f"\n🔍 Processing {label} videos from: {video_dir}")
        
        # Get all video files
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        
        if not video_files:
            print(f"   No video files found in {video_dir}")
            continue
        
        print(f"   Found {len(video_files)} videos")
        
        # Create output subdirectory
        output_dir = os.path.join(output_base, label)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process only first 5 videos for testing
        test_mode = True
        if test_mode:
            video_files = video_files[:5]
            print(f"   TEST MODE: Processing first {len(video_files)} videos")
        
        faces_from_category = 0
        for video_file in tqdm(video_files, desc=f"Extracting {label}"):
            video_path = os.path.join(video_dir, video_file)
            faces_extracted = extract_faces_from_video(video_path, output_dir, frames_per_video=3)
            faces_from_category += faces_extracted
        
        total_faces += faces_from_category
        print(f"   ✅ Extracted {faces_from_category} faces from {label}")
    
    print(f"\n🎉 Face extraction completed!")
    print(f"📊 Total faces extracted: {total_faces}")
    print(f"📁 Output directory: {output_base}")

if __name__ == "__main__":
    main()