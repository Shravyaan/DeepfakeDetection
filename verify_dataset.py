# verify_dataset.py
import os

def verify_dataset():
    print("🔍 Verifying existing dataset structure...")
    
    dataset_path = "data/FaceForensics++_C23"
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset folder not found")
        return False
    
    print(f"✅ Dataset found: {dataset_path}")
    
    # Check the structure
    print("\n📂 Dataset structure:")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        
        # Show first few files in each directory
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:
            print(f'{subindent}{file}')
        if len(files) > 3:
            print(f'{subindent}... and {len(files) - 3} more files')
        
        # Stop at 2 levels deep to avoid too much output
        if level >= 2:
            break
    
    return True

def check_video_files():
    print("\n🎥 Checking for video files...")
    
    # Check common paths for videos
    video_paths = [
        "data/FaceForensics++_C23/original_sequences/raw/videos",
        "data/FaceForensics++_C23/manipulated_sequences/Deepfakes/raw/videos",
        "data/FaceForensics++_C23/manipulated_sequences/Face2Face/raw/videos",
    ]
    
    for path in video_paths:
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.mp4')]
            print(f"✅ {path}: {len(files)} video files")
        else:
            print(f"❌ {path}: Path not found")

if __name__ == "__main__":
    if verify_dataset():
        check_video_files()
        print("\n🎉 Dataset is ready! You can skip the download.")
        print("\nNext: Run create_manifest.py")
    else:
        print("\n❌ Dataset verification failed")