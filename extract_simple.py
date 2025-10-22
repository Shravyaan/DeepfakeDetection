# extract_simple.py
import os
import cv2
from tqdm import tqdm

print("🎭 Simple Face Extraction")
print("=" * 40)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_faces_simple(video_path, output_dir, max_faces=3):
    """Extract faces from video - simple version"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return 0
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        faces_found = 0
        
        # Sample frames throughout the video
        for frame_idx in range(0, total_frames, total_frames // max_faces):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face
                face_img = frame[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue
                
                # Resize
                face_img = cv2.resize(face_img, (224, 224))
                
                # Save
                output_path = os.path.join(output_dir, f"{video_name}_face{faces_found}.jpg")
                cv2.imwrite(output_path, face_img)
                faces_found += 1
                
                if faces_found >= max_faces:
                    break
            
            if faces_found >= max_faces:
                break
        
        cap.release()
        return faces_found
        
    except Exception as e:
        print(f"Error with {video_path}: {e}")
        return 0

def main():
    output_base = "extracted_faces"
    
    # Only process real and one fake type for testing
    datasets = {
        "real": "data/FaceForensics++_C23/original",
        "fake": "data/FaceForensics++_C23/Deepfakes"
    }
    
    total_faces = 0
    
    for label, video_dir in datasets.items():
        if not os.path.exists(video_dir):
            print(f"❌ Missing: {video_dir}")
            continue
            
        print(f"\n📁 Processing {label}...")
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')][:3]  # Only 3 videos
        
        output_dir = os.path.join(output_base, label)
        faces_count = 0
        
        for video_file in tqdm(video_files, desc=label):
            video_path = os.path.join(video_dir, video_file)
            faces = extract_faces_simple(video_path, output_dir, 2)  # 2 faces per video
            faces_count += faces
        
        total_faces += faces_count
        print(f"   Extracted {faces_count} faces")
    
    print(f"\n✅ Total faces extracted: {total_faces}")

if __name__ == "__main__":
    main()