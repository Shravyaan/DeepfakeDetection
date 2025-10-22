# check_extraction.py
import os
import subprocess

def check_extract_faces_script():
    print("🔍 Checking extract_faces.py compatibility...")
    
    if not os.path.exists("extract_faces.py"):
        print("❌ extract_faces.py not found!")
        return False
    
    # Read the script to check its structure
    with open("extract_faces.py", 'r') as f:
        content = f.read()
    
    # Check if it uses the correct paths
    if "original_sequences" in content and "manipulated_sequences" in content:
        print("⚠️  extract_faces.py uses old path structure")
        print("   It expects: original_sequences/ and manipulated_sequences/")
        print("   Your dataset has: original/ and Deepfakes/, Face2Face/, etc.")
        return False
    elif "original" in content and "Deepfakes" in content:
        print("✅ extract_faces.py should work with your dataset structure")
        return True
    else:
        print("❓ Cannot determine extract_faces.py compatibility")
        return None

def suggest_fix():
    print("\n💡 If extract_faces.py fails, here's the fix:")
    print("""
The main changes needed in extract_faces.py:

1. Update video paths from:
   OLD: data/FaceForensics++_C23/original_sequences/raw/videos
   NEW: data/FaceForensics++_C23/original

2. Update fake video paths from:
   OLD: data/FaceForensics++_C23/manipulated_sequences/Deepfakes/raw/videos
   NEW: data/FaceForensics++_C23/Deepfakes
   
3. Same for other fake types: Face2Face, FaceShifter, etc.
""")

if __name__ == "__main__":
    result = check_extract_faces_script()
    if result is False:
        suggest_fix()