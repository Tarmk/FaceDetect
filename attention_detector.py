import cv2
import numpy as np
import mediapipe as mp
import math
import time

class AttentionDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks indices for MediaPipe
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Eye aspect ratio threshold
        self.EAR_THRESHOLD = 0.15
        self.ATTENTION_THRESHOLD = 0.3
        
        # Colors
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.BLUE = (255, 0, 0)
        self.YELLOW = (0, 255, 255)
        
        # Attention tracking
        self.attention_score = 0.0
        self.attention_history = []
        self.max_history = 10  # 10 frames for faster response
        
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        # Vertical eye landmarks
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal eye landmark
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_eye_landmarks(self, landmarks, eye_indices):
        """Extract eye landmarks from face mesh"""
        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * self.frame_width)
                y = int(landmarks[idx].y * self.frame_height)
                eye_points.append([x, y])
        return np.array(eye_points)
    
    def calculate_head_pose(self, landmarks):
        """Calculate head pose angles"""
        # Get key facial landmarks
        nose_tip = [landmarks[1].x, landmarks[1].y]
        chin = [landmarks[152].x, landmarks[152].y]
        left_eye_corner = [landmarks[33].x, landmarks[33].y]
        right_eye_corner = [landmarks[362].x, landmarks[362].y]
        left_mouth_corner = [landmarks[61].x, landmarks[61].y]
        right_mouth_corner = [landmarks[291].x, landmarks[291].y]
        
        # Calculate face width for normalization
        face_width = abs(right_eye_corner[0] - left_eye_corner[0])
        
        # Calculate head rotation (yaw) - normalized by face width
        eye_center_x = (left_eye_corner[0] + right_eye_corner[0]) / 2
        mouth_center_x = (left_mouth_corner[0] + right_mouth_corner[0]) / 2
        
        # Use both nose and mouth center for better yaw detection
        face_center_x = (nose_tip[0] + mouth_center_x) / 2
        head_yaw = (face_center_x - eye_center_x) / face_width if face_width > 0 else 0
        
        # Calculate head tilt (pitch) - normalized by face height
        face_height = abs(chin[1] - min(left_eye_corner[1], right_eye_corner[1]))
        head_pitch = (chin[1] - nose_tip[1]) / face_height if face_height > 0 else 0
        
        return head_yaw, head_pitch
    
    def is_looking_at_camera(self, left_ear, right_ear, head_yaw, head_pitch):
        """Determine if student is paying attention based on eye and head pose"""
        # Check if eyes are open
        eyes_open = left_ear > self.EAR_THRESHOLD and right_ear > self.EAR_THRESHOLD
        
        # Check if head is facing forward (within reasonable range)
        # Normalized values: smaller thresholds for better detection
        head_forward = abs(head_yaw) < 0.15 and abs(head_pitch) < 0.2
        
        # Combine factors for attention score
        if eyes_open and head_forward:
            return True, 1.0
        elif eyes_open and abs(head_yaw) < 0.25:
            return True, 0.7
        elif eyes_open and abs(head_yaw) < 0.4:
            return True, 0.5
        elif eyes_open and abs(head_yaw) < 0.6:
            return True, 0.3
        elif eyes_open and abs(head_yaw) < 0.8:
            return True, 0.1
        else:
            return False, 0.0
    
    def draw_eye_lines(self, frame, eye_landmarks, color):
        """Draw lines connecting eye landmarks"""
        if len(eye_landmarks) >= 6:
            # Draw eye contour
            cv2.polylines(frame, [eye_landmarks], True, color, 2)
            
            # Draw eye center and direction lines
            eye_center = np.mean(eye_landmarks, axis=0).astype(int)
            cv2.circle(frame, tuple(eye_center), 3, color, -1)
            
            # Draw direction line from eye center towards camera
            direction_end = (eye_center[0], eye_center[1] - 30)
            cv2.arrowedLine(frame, tuple(eye_center), direction_end, color, 2)
    
    def draw_face_box(self, frame, landmarks, attention_status):
        """Draw bounding box around face with attention status"""
        # Get face bounding box
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        
        x_min = int(min(x_coords) * self.frame_width)
        x_max = int(max(x_coords) * self.frame_width)
        y_min = int(min(y_coords) * self.frame_height)
        y_max = int(max(y_coords) * self.frame_height)
        
        # Choose color based on attention status
        box_color = self.GREEN if attention_status else self.RED
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
        
        # Draw attention status text
        status_text = "PAYING ATTENTION" if attention_status else "NOT PAYING ATTENTION"
        cv2.putText(frame, status_text, (x_min, y_min - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    
    def process_frame(self, frame):
        """Process a single frame for attention detection"""
        self.frame_height, self.frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Get eye landmarks
                left_eye_points = self.get_eye_landmarks(landmarks, self.LEFT_EYE_LANDMARKS[:6])
                right_eye_points = self.get_eye_landmarks(landmarks, self.RIGHT_EYE_LANDMARKS[:6])
                
                if len(left_eye_points) >= 6 and len(right_eye_points) >= 6:
                    # Calculate eye aspect ratios
                    left_ear = self.calculate_ear(left_eye_points)
                    right_ear = self.calculate_ear(right_eye_points)
                    
                    # Calculate head pose
                    head_yaw, head_pitch = self.calculate_head_pose(landmarks)
                    
                    # Determine attention status
                    attention_status, attention_score = self.is_looking_at_camera(
                        left_ear, right_ear, head_yaw, head_pitch
                    )
                    
                    # Update attention history for smoothing
                    self.attention_history.append(attention_score)
                    if len(self.attention_history) > self.max_history:
                        self.attention_history.pop(0)
                    
                    # Calculate smoothed attention score
                    smoothed_score = np.mean(self.attention_history)
                    final_attention_status = smoothed_score > self.ATTENTION_THRESHOLD
                    
                    # Draw eye lines
                    self.draw_eye_lines(frame, left_eye_points, self.BLUE)
                    self.draw_eye_lines(frame, right_eye_points, self.BLUE)
                    
                    # Draw face bounding box
                    self.draw_face_box(frame, landmarks, final_attention_status)
                    
                    # Draw attention score
                    score_text = f"Attention Score: {smoothed_score:.2f}"
                    cv2.putText(frame, score_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.YELLOW, 2)
                    
                    # Draw EAR values
                    ear_text = f"Left EAR: {left_ear:.2f} | Right EAR: {right_ear:.2f}"
                    cv2.putText(frame, ear_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.YELLOW, 1)
                    
                    # Draw head pose with more detail
                    pose_text = f"Head Yaw: {head_yaw:.3f} | Pitch: {head_pitch:.3f}"
                    cv2.putText(frame, pose_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.YELLOW, 1)
                    
                    # Draw head pose status
                    yaw_status = "FORWARD" if abs(head_yaw) < 0.15 else "TURNED"
                    status_color = self.GREEN if abs(head_yaw) < 0.15 else self.RED
                    cv2.putText(frame, f"Head: {yaw_status}", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        return frame
    
    def run(self):
        """Main loop for attention detection"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting attention detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display instructions
            cv2.putText(processed_frame, "Press 'q' to quit", (10, processed_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Student Attention Detector', processed_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Attention detection stopped")

def main():
    detector = AttentionDetector()
    detector.run()

if __name__ == "__main__":
    main() 