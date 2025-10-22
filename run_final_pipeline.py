# run_final_pipeline.py
import os
import time

def run_command(description, command):
    print(f"\n{description}")
    print("─" * 40)
    start_time = time.time()
    
    try:
        result = os.system(command)
        if result == 0:
            print(f"✅ Success! ({time.time() - start_time:.1f}s)")
            return True
        else:
            print(f"❌ Failed with code: {result}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🚀 FINAL PIPELINE - ERROR FREE")
    print("=" * 50)
    
    steps = [
        ("1. Testing OpenCV", "python test_opencv_fixed.py"),
        ("2. Creating manifest", "python create_manifest_fixed.py"),
        ("3. Extracting faces", "python extract_simple.py"),
    ]
    
    all_success = True
    for description, command in steps:
        success = run_command(description, command)
        if not success:
            print(f"\n💡 Stopping pipeline. Fix the issue and continue.")
            all_success = False
            break
    
    if all_success:
        print("\n🎉 PIPELINE COMPLETED!")
        print("\n📁 Check these folders:")
        print("   - manifests/ (contains train/test lists)")
        print("   - extracted_faces/ (contains face images)")
        print("\n🎯 Next: Start model training")

if __name__ == "__main__":
    main()