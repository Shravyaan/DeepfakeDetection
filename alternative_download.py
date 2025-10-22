# alternative_download.py
import requests
import os
import time

def download_file_chunked(url, filepath):
    """Download large file in chunks with progress"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading... {percent:.1f}%", end='', flush=True)
        
        print(f"\n✓ Download completed: {filepath}")
        return True
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False

def main():
    print("Alternative Dataset Setup")
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    
    # Since direct download might not work due to authentication,
    # let's create a mock dataset structure for testing
    print("Since Kaggle requires login, let's create a test structure...")
    
    # Create sample directory structure
    test_dirs = [
        'data/FaceForensics++_C23/original',
        'data/FaceForensics++_C23/Deepfakes', 
        'data/FaceForensics++_C23/Face2Face',
        'data/FaceForensics++_C23/FaceShifter',
        'data/FaceForensics++_C23/FaceSwap',
        'data/FaceForensics++_C23/NeuralTextures',
    ]
    
    for dir_path in test_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create a README explaining next steps
    with open('data/README.txt', 'w') as f:
        f.write("""Dataset Download Instructions:

1. MANUAL DOWNLOAD (Recommended):
   - Visit: https://www.kaggle.com/datasets/xdxd003/ff-c23
   - Click 'Download' button
   - Save as 'ff-c23.zip' in this folder
   - Extract the zip file here

2. The extracted folder should contain:
   - FaceForensics++_C23/
     - original/ (real videos)
     - Deepfakes/ (fake videos)
     - Face2Face/ (fake videos)
     - FaceShifter/ (fake videos) 
     - FaceSwap/ (fake videos)
     - NeuralTextures/ (fake videos)

3. After downloading, run:
   - python create_manifest.py
   - python extract_faces.py
   - python test_loader.py
""")
    
    print("✓ Created project structure")
    print("✓ Created download instructions in data/README.txt")
    print("\n📥 Please download the dataset manually from Kaggle")
    print("   URL: https://www.kaggle.com/datasets/xdxd003/ff-c23")

if __name__ == "__main__":
    main()