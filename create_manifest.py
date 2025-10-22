# create_manifest.py
import os
import random
from tqdm import tqdm

# -- Configuration for YOUR dataset structure --
DATA_DIR = "data/FaceForensics++_C23"

# Path to REAL videos (based on your actual structure)
ORIGINAL_VID_DIR = os.path.join(DATA_DIR, "original")

# Paths to FAKE videos (based on your actual structure)
FAKE_VIDEO_DIRS = [
    os.path.join(DATA_DIR, "Deepfakes"),
    os.path.join(DATA_DIR, "Face2Face"),
    os.path.join(DATA_DIR, "FaceShifter"),
    os.path.join(DATA_DIR, "FaceSwap"),
    os.path.join(DATA_DIR, "NeuralTextures"),
    os.path.join(DATA_DIR, "DeepFakeDetection"),
]

def create_manifest():
    print("🎬 Creating Dataset Manifest for Your Dataset Structure")
    print("=" * 60)
    
    # Check if directories exist
    print("🔍 Checking dataset directories...")
    all_dirs_exist = True
    for dir_path in [ORIGINAL_VID_DIR] + FAKE_VIDEO_DIRS:
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if f.endswith('.mp4')])
            print(f"✅ {os.path.basename(dir_path)}: {file_count} videos")
        else:
            print(f"❌ {os.path.basename(dir_path)}: NOT FOUND")
            all_dirs_exist = False
    
    if not all_dirs_exist:
        print("\n❌ Some directories are missing. Please check your dataset structure.")
        return
    
    print("\n📊 Collecting video files...")
    
    # Get all original video files (REAL videos)
    original_videos = [f for f in os.listdir(ORIGINAL_VID_DIR) if f.endswith('.mp4')]
    original_ids = [f.split('.')[0] for f in original_videos]
    
    print(f"📹 Found {len(original_videos)} original (REAL) videos")
    
    # Shuffle for train/test split (80% train, 20% test)
    random.shuffle(original_ids)
    split_idx = int(0.8 * len(original_ids))
    train_ids = set(original_ids[:split_idx])
    test_ids = set(original_ids[split_idx:])
    
    print(f"🎯 Split: {len(train_ids)} train, {len(test_ids)} test")
    
    manifests = {'train': [], 'test': []}
    
    # Process original videos (REAL = label 0)
    print("\n🔵 Processing REAL videos...")
    for video_file in tqdm(original_videos):
        video_id = video_file.split('.')[0]
        video_path = os.path.join(ORIGINAL_VID_DIR, video_file)
        
        split = 'train' if video_id in train_ids else 'test'
        manifests[split].append((video_path, 0))  # 0 = real
    
    # Process fake videos (FAKE = label 1)
    fake_count = 0
    for fake_dir in FAKE_VIDEO_DIRS:
        fake_type = os.path.basename(fake_dir)
        print(f"\n🔴 Processing {fake_type} fake videos...")
        
        if os.path.exists(fake_dir):
            fake_videos = [f for f in os.listdir(fake_dir) if f.endswith('.mp4')]
            for video_file in tqdm(fake_videos):
                video_id = video_file.split('.')[0]
                video_path = os.path.join(fake_dir, video_file)
                
                # For fake videos, use the same split as their corresponding original
                split = 'train' if video_id in train_ids else 'test'
                manifests[split].append((video_path, 1))  # 1 = fake
                fake_count += 1
    
    print(f"\n📈 Final Dataset Statistics:")
    print(f"   Real videos: {len(original_videos)}")
    print(f"   Fake videos: {fake_count}")
    print(f"   Total videos: {len(original_videos) + fake_count}")
    print(f"   Train set: {len(manifests['train'])} videos")
    print(f"   Test set: {len(manifests['test'])} videos")
    
    # Calculate balance
    train_real = sum(1 for _, label in manifests['train'] if label == 0)
    train_fake = sum(1 for _, label in manifests['train'] if label == 1)
    test_real = sum(1 for _, label in manifests['test'] if label == 0)
    test_fake = sum(1 for _, label in manifests['test'] if label == 1)
    
    print(f"\n⚖️  Dataset Balance:")
    print(f"   Train - Real: {train_real}, Fake: {train_fake}")
    print(f"   Test  - Real: {test_real}, Fake: {test_fake}")
    
    # Save manifests
    os.makedirs('manifests', exist_ok=True)
    
    for split in ['train', 'test']:
        manifest_file = f'manifests/{split}_manifest.csv'
        with open(manifest_file, 'w') as f:
            for path, label in manifests[split]:
                f.write(f'{path},{label}\n')
        
        print(f"💾 Saved {len(manifests[split])} entries to {manifest_file}")
    
    print("\n✅ Manifest creation completed!")
    print("\n🎯 Next step: Run python extract_faces.py")

if __name__ == "__main__":
    create_manifest()
    