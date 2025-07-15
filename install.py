#!/usr/bin/env python3
"""
Installation script for FaceDetect - Student Attention Detection System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install Python requirements"""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return False

def test_mediapipe():
    """Test MediaPipe functionality"""
    print("Testing MediaPipe...")
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        print("✓ MediaPipe Face Mesh initialized successfully!")
        return True
    except Exception as e:
        print(f"✗ MediaPipe test failed: {e}")
        return False

def check_camera():
    """Check if camera is available"""
    print("Checking camera availability...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            print("✓ Camera is available!")
            return True
        else:
            print("✗ Camera not available")
            return False
    except Exception as e:
        print(f"✗ Error checking camera: {e}")
        return False

def main():
    print("=" * 50)
    print("FaceDetect - Student Attention Detection System")
    print("Installation Script")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Test MediaPipe
    if not test_mediapipe():
        print("⚠ MediaPipe test failed, but you can still try running the application")
    
    # Check camera
    if not check_camera():
        print("⚠ Camera check failed, but you can still try running the application")
    
    print("\n" + "=" * 50)
    print("Installation completed!")
    print("To run the attention detector:")
    print("  python attention_detector.py")
    print("=" * 50)

if __name__ == "__main__":
    main() 