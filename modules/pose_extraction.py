"""
Pose and Expression Extraction Module

Extracts body pose, facial landmarks, expressions, and lip movements
for each tracked person in the video.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import cv2


@dataclass
class FacialLandmarks:
    """Facial landmark data."""
    landmarks_2d: np.ndarray  # (N, 2) facial landmarks
    landmarks_3d: Optional[np.ndarray] = None  # (N, 3) if available
    confidence: float = 1.0
    
    # Derived features
    left_eye: Optional[np.ndarray] = None
    right_eye: Optional[np.ndarray] = None
    nose: Optional[np.ndarray] = None
    mouth: Optional[np.ndarray] = None
    jaw: Optional[np.ndarray] = None


@dataclass
class LipSyncData:
    """Lip synchronization data for a frame."""
    mouth_landmarks: np.ndarray  # Mouth-specific landmarks
    mouth_open_ratio: float  # 0-1, how open the mouth is
    mouth_width_ratio: float  # Relative mouth width
    viseme: Optional[str] = None  # Phoneme visual representation
    

@dataclass
class ExpressionData:
    """Facial expression data."""
    # Action Units (FACS-based)
    action_units: Dict[str, float] = field(default_factory=dict)
    
    # High-level expressions
    expression_vector: Optional[np.ndarray] = None  # Latent expression
    dominant_expression: Optional[str] = None  # happy, sad, angry, etc.
    expression_intensity: float = 0.0


@dataclass
class BodyPose:
    """Body pose keypoints."""
    keypoints_2d: np.ndarray  # (N, 2) body keypoints
    keypoints_3d: Optional[np.ndarray] = None  # (N, 3) if available
    confidence: np.ndarray  # Per-keypoint confidence
    
    # Pose metadata
    facing_direction: str = "front"  # front, back, left, right
    body_orientation: float = 0.0  # Angle in degrees


@dataclass
class PersonPoseData:
    """Complete pose and expression data for one person in one frame."""
    track_id: int
    frame_idx: int
    bbox: np.ndarray
    
    body_pose: Optional[BodyPose] = None
    facial_landmarks: Optional[FacialLandmarks] = None
    expression: Optional[ExpressionData] = None
    lip_sync: Optional[LipSyncData] = None
    
    # Whether face is visible
    face_visible: bool = True
    face_bbox: Optional[np.ndarray] = None


class PoseExpressionExtractor:
    """
    Extracts pose, facial landmarks, expressions, and lip sync data.
    Uses MediaPipe for facial analysis and MMPose for body pose.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cuda')
        self.pose_model_name = config.get('model', 'rtmpose-l')
        self.body_model = config.get('body_model', 'wholebody')
        self.landmark_model = config.get('landmark_model', 'mediapipe')
        self.lip_sync_precision = config.get('lip_sync_precision', 'high')
        
        self.pose_estimator = None
        self.face_mesh = None
        self.expression_model = None
        
    def initialize(self):
        """Initialize all models."""
        self._init_mediapipe()
        self._init_pose_estimator()
        print("Pose and expression models initialized")
        
    def _init_mediapipe(self):
        """Initialize MediaPipe face mesh."""
        import mediapipe as mp
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True,  # Includes iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def _init_pose_estimator(self):
        """Initialize pose estimation model."""
        # Using RTMPose via MMPose
        try:
            from mmpose.apis import MMPoseInferencer
            self.pose_estimator = MMPoseInferencer(
                pose2d=self.pose_model_name,
                device=self.device
            )
        except ImportError:
            print("Warning: MMPose not available, using MediaPipe for pose")
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose_estimator = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5
            )
            
    def extract_person_data(self, frame: np.ndarray, 
                           bbox: np.ndarray,
                           track_id: int,
                           frame_idx: int) -> PersonPoseData:
        """
        Extract all pose and expression data for a single person.
        
        Args:
            frame: Full BGR frame
            bbox: Person bounding box [x1, y1, x2, y2]
            track_id: Person track ID
            frame_idx: Frame index
            
        Returns:
            PersonPoseData with all extracted information
        """
        if self.face_mesh is None:
            self.initialize()
            
        # Crop person region with padding
        x1, y1, x2, y2 = bbox.astype(int)
        pad = int(max(x2 - x1, y2 - y1) * 0.1)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        person_crop = frame[y1:y2, x1:x2]
        
        person_data = PersonPoseData(
            track_id=track_id,
            frame_idx=frame_idx,
            bbox=bbox
        )
        
        # Extract body pose
        person_data.body_pose = self._extract_body_pose(person_crop, bbox)
        
        # Extract facial data
        facial_result = self._extract_facial_data(person_crop, bbox)
        if facial_result:
            person_data.facial_landmarks = facial_result['landmarks']
            person_data.expression = facial_result['expression']
            person_data.lip_sync = facial_result['lip_sync']
            person_data.face_visible = True
            person_data.face_bbox = facial_result.get('face_bbox')
        else:
            person_data.face_visible = False

        return person_data

    def _extract_body_pose(self, person_crop: np.ndarray,
                          bbox: np.ndarray) -> Optional[BodyPose]:
        """Extract body pose keypoints."""
        try:
            if hasattr(self, 'mp_pose'):
                # MediaPipe fallback
                rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                results = self.pose_estimator.process(rgb)

                if results.pose_landmarks:
                    h, w = person_crop.shape[:2]
                    keypoints = np.array([
                        [lm.x * w + bbox[0], lm.y * h + bbox[1]]
                        for lm in results.pose_landmarks.landmark
                    ])
                    confidence = np.array([
                        lm.visibility for lm in results.pose_landmarks.landmark
                    ])

                    # Determine facing direction
                    facing = self._determine_facing_direction(keypoints)

                    return BodyPose(
                        keypoints_2d=keypoints,
                        confidence=confidence,
                        facing_direction=facing
                    )
            else:
                # MMPose
                results = next(self.pose_estimator(person_crop))
                if results['predictions']:
                    pred = results['predictions'][0][0]
                    keypoints = pred['keypoints']
                    keypoints[:, 0] += bbox[0]
                    keypoints[:, 1] += bbox[1]

                    return BodyPose(
                        keypoints_2d=keypoints[:, :2],
                        confidence=keypoints[:, 2],
                        facing_direction=self._determine_facing_direction(keypoints)
                    )
        except Exception as e:
            print(f"Pose extraction error: {e}")
        return None

    def _extract_facial_data(self, person_crop: np.ndarray,
                            bbox: np.ndarray) -> Optional[Dict]:
        """Extract facial landmarks, expression, and lip sync."""
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h, w = person_crop.shape[:2]

        # Convert to numpy
        landmarks_2d = np.array([
            [lm.x * w + bbox[0], lm.y * h + bbox[1]]
            for lm in face_landmarks.landmark
        ])

        # Extract mouth landmarks for lip sync (indices 61-68, 78-95 in MediaPipe)
        mouth_indices = list(range(61, 69)) + list(range(78, 96))
        mouth_landmarks = landmarks_2d[mouth_indices] if len(landmarks_2d) > max(mouth_indices) else landmarks_2d[-20:]

        # Calculate lip sync metrics
        lip_sync = self._calculate_lip_sync(mouth_landmarks)

        # Extract expression
        expression = self._extract_expression(landmarks_2d)

        return {
            'landmarks': FacialLandmarks(landmarks_2d=landmarks_2d),
            'expression': expression,
            'lip_sync': lip_sync,
            'face_bbox': self._get_face_bbox(landmarks_2d)
        }

    def _calculate_lip_sync(self, mouth_landmarks: np.ndarray) -> LipSyncData:
        """Calculate lip sync metrics from mouth landmarks."""
        # Mouth open ratio (vertical distance / horizontal distance)
        if len(mouth_landmarks) >= 8:
            top = mouth_landmarks[2]  # Upper lip
            bottom = mouth_landmarks[6]  # Lower lip
            left = mouth_landmarks[0]
            right = mouth_landmarks[4]

            vertical = np.linalg.norm(top - bottom)
            horizontal = np.linalg.norm(left - right)

            open_ratio = vertical / (horizontal + 1e-6)
            width_ratio = horizontal / 100  # Normalized
        else:
            open_ratio = 0.0
            width_ratio = 0.5

        return LipSyncData(
            mouth_landmarks=mouth_landmarks,
            mouth_open_ratio=float(open_ratio),
            mouth_width_ratio=float(width_ratio)
        )

    def _extract_expression(self, landmarks: np.ndarray) -> ExpressionData:
        """Extract expression from facial landmarks."""
        # Simplified expression extraction based on landmark geometry
        # In production, use a dedicated expression recognition model
        return ExpressionData(
            action_units={},
            dominant_expression="neutral",
            expression_intensity=0.5
        )

    def _determine_facing_direction(self, keypoints: np.ndarray) -> str:
        """Determine if person is facing front, back, left, or right."""
        # Use shoulder and hip positions to determine orientation
        # This is a simplified heuristic
        if len(keypoints) >= 12:
            left_shoulder = keypoints[11]
            right_shoulder = keypoints[12]
            shoulder_diff = right_shoulder[0] - left_shoulder[0]

            if abs(shoulder_diff) < 20:
                return "back" if shoulder_diff < 0 else "front"
            elif shoulder_diff > 0:
                return "front"
            else:
                return "back"
        return "front"

    def _get_face_bbox(self, landmarks: np.ndarray) -> np.ndarray:
        """Get face bounding box from landmarks."""
        x_min, y_min = landmarks.min(axis=0)
        x_max, y_max = landmarks.max(axis=0)
        return np.array([x_min, y_min, x_max, y_max])

