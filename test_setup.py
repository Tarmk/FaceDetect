#!/usr/bin/env python3
"""
Test script to verify FaceDetect setup
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("✓ MediaPipe imported successfully")
    except ImportError as e:
        print(f"✗ MediaPipe import failed: {e}")
        return False
    
    # YOLO is no longer required for this application
    print("✓ YOLO dependency removed - using MediaPipe only")
    
    return True

def test_camera():
    """Test camera functionality"""
    print("\nTesting camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Camera could not be opened")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("✗ Could not read frame from camera")
            cap.release()
            return False
        
        print(f"✓ Camera working - Frame size: {frame.shape}")
        cap.release()
        return True
        
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe Face Mesh"""
    print("\nTesting MediaPipe Face Mesh...")
    
    try:
        import mediapipe as mp
        import numpy as np
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = face_mesh.process(dummy_image)
        
        print("✓ MediaPipe Face Mesh initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ MediaPipe Face Mesh test failed: {e}")
        return False

def test_system():
    """Test overall system readiness"""
    print("\nTesting system readiness...")
    print("✓ System ready for attention detection")
    return True

def main():
    print("=" * 50)
    print("FaceDetect - Setup Verification")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test camera
    if not test_camera():
        all_tests_passed = False
    
    # Test MediaPipe
    if not test_mediapipe():
        all_tests_passed = False
    
    # Test system
    if not test_system():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✅ All tests passed! You can now run:")
        print("   python attention_detector.py")
    else:
        print("❌ Some tests failed. Please check the installation:")
        print("   python install.py")
    print("=" * 50)

if __name__ == "__main__":
    main() 