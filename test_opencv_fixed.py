# test_opencv_fixed.py
import cv2
import numpy as np
import os

print("🧪 Testing OpenCV installation...")

# Test OpenCV version
print(f"✅ OpenCV version: {cv2.__version__}")

# Test face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("❌ Failed to load face detector")
else:
    print("✅ Face detector loaded successfully")

# Test if we can create a simple image
test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
success = cv2.imwrite('test_image.jpg', test_image)
if success:
    print("✅ Can write images")
    # Remove test file
    if os.path.exists('test_image.jpg'):
        os.remove('test_image.jpg')
else:
    print("❌ Cannot write images")

print("🎉 OpenCV test completed!")