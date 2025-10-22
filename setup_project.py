# setup_project.py
import os
import zipfile
import shutil

def setup_project():
    print("=== Deepfake Detection Project Setup ===")
    
    # Create directory structure
    directories = [
        'data',
        'manifests', 
        'model_development',
        'model_development/checkpoints',
        'model_development/logs',
        'extracted_faces'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    # Check for dataset zip file
    zip_path = "data/ff-c23.zip"
    
    if os.path.exists(zip_path):
        print(f"\n✓ Dataset found: {zip_path}")
        print("File size: {:.2f} GB".format(os.path.getsize(zip_path) / (1024**3)))
        
        # Extract dataset
        print("Extracting dataset... This may take 5-10 minutes...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("data/")
            print("✓ Extraction completed!")
            
            # List extracted contents
            print("\nExtracted contents:")
            data_items = os.listdir("data/")
            for item in data_items:
                item_path = os.path.join("data", item)
                if os.path.isdir(item_path):
                    print(f"📁 {item}/")
                    try:
                        sub_items = os.listdir(item_path)[:3]  # First 3 items
                        for sub_item in sub_items:
                            print(f"   └── {sub_item}")
                        if len(os.listdir(item_path)) > 3:
                            print(f"   └── ... and {len(os.listdir(item_path)) - 3} more")
                    except:
                        pass
                else:
                    print(f"📄 {item}")
                    
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            return False
    else:
        print(f"\n✗ Dataset not found: {zip_path}")
        print("\n📥 INSTRUCTIONS TO DOWNLOAD:")
        print("1. Go to: https://www.kaggle.com/datasets/xdxd003/ff-c23")
        print("2. Click the 'Download' button (↓ icon)")
        print("3. Save the file as 'ff-c23.zip'")
        print("4. Place it in: data/ff-c23.zip")
        print("5. Run this script again")
        return False
    
    print("\n🎉 Project setup completed!")
    print("\nNext steps:")
    print("1. Run: python create_manifest.py")
    print("2. Run: python extract_faces.py") 
    print("3. Run: python test_loader.py")
    print("4. Start model training in model_development/")
    
    return True

if __name__ == "__main__":
    setup_project()