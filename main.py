# ========================= SETUP AND INSTALLATIONS =========================
!pip install ultralytics opencv-python-headless matplotlib numpy torch torchvision scipy scikit-learn
!apt update &> /dev/null
!apt install ffmpeg &> /dev/null

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import torch
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
import os
import json

# ========================= LOAD MODEL =========================
model_path = "/content/best.pt"

# Check if model exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please ensure the model file is uploaded to /content/best.pt")
    exit()

# Load the model
print(f"Loading model from {model_path}...")
model = YOLO(model_path)
print("Model loaded successfully!")

# ========================= UTILITY FUNCTIONS =========================

def extract_features(image, bbox):
    """Extract visual features from player bounding box"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return np.zeros(256)  # Return zero vector for invalid bbox
    
    # Extract player region
    player_region = image[y1:y2, x1:x2]
    
    if player_region.size == 0:
        return np.zeros(256)
    
    # Resize to standard size
    try:
        player_region = cv2.resize(player_region, (64, 128))
    except:
        return np.zeros(256)
    
    # Color histogram features
    hist_b = cv2.calcHist([player_region], [0], None, [16], [0, 256])
    hist_g = cv2.calcHist([player_region], [1], None, [16], [0, 256])
    hist_r = cv2.calcHist([player_region], [2], None, [16], [0, 256])
    
    # Texture features
    gray = cv2.cvtColor(player_region, cv2.COLOR_BGR2GRAY)
    texture = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Position features (normalized)
    center_x = (x1 + x2) / (2 * w)
    center_y = (y1 + y2) / (2 * h)
    width_ratio = (x2 - x1) / w
    height_ratio = (y2 - y1) / h
    
    # Size features
    area = (x2 - x1) * (y2 - y1)
    aspect_ratio = (x2 - x1) / max(1, (y2 - y1))
    
    # Combine all features (total: 16+16+16+6 = 54, pad to 256)
    features = np.concatenate([
        hist_b.flatten(),
        hist_g.flatten(), 
        hist_r.flatten(),
        [texture, center_x, center_y, width_ratio, height_ratio, area, aspect_ratio]
    ])
    
    # Normalize features
    if np.linalg.norm(features) > 0:
        features = features / np.linalg.norm(features)
    
    # Pad to 256 dimensions
    if len(features) < 256:
        features = np.pad(features, (0, 256 - len(features)), 'constant')
    
    return features[:256]

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def feature_similarity(feat1, feat2):
    """Calculate similarity between two feature vectors"""
    if np.linalg.norm(feat1) == 0 or np.linalg.norm(feat2) == 0:
        return 0.0
    return max(0, 1 - cosine(feat1, feat2))

def euclidean_distance(box1, box2):
    """Calculate euclidean distance between box centers"""
    center1 = [(box1[0] + box1[2])/2, (box1[1] + box1[3])/2]
    center2 = [(box2[0] + box2[2])/2, (box2[1] + box2[3])/2]
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

# ========================= PLAYER TRACKER CLASS =========================

class PlayerTracker:
    def __init__(self, similarity_threshold=0.4, iou_threshold=0.2, max_disappeared=20):
        self.players = {}  # player_id -> player_info
        self.next_id = 1
        self.similarity_threshold = similarity_threshold
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.frame_count = 0
        
    def update(self, detections, frame):
        """Update tracker with new detections"""
        self.frame_count += 1
        current_assignments = {}
        
        if len(detections) == 0:
            # Mark all players as disappeared
            for player_id in self.players:
                self.players[player_id]['disappeared'] += 1
            return current_assignments
        
        # Extract features for all detections
        detection_features = []
        for det in detections:
            bbox = det['bbox']
            features = extract_features(frame, bbox)
            detection_features.append(features)
        
        # Create cost matrix for Hungarian algorithm (simplified)
        active_players = [pid for pid, pinfo in self.players.items() 
                         if pinfo['disappeared'] < self.max_disappeared]
        
        if len(active_players) == 0:
            # No active players, create new ones for all detections
            for i, det in enumerate(detections):
                new_player_id = self.next_id
                self.next_id += 1
                
                self.players[new_player_id] = {
                    'features': detection_features[i],
                    'last_bbox': det['bbox'],
                    'last_seen': self.frame_count,
                    'confidence': det['confidence'],
                    'first_seen': self.frame_count,
                    'disappeared': 0
                }
                
                current_assignments[i] = new_player_id
        else:
            # Match detections to existing players
            unmatched_detections = list(range(len(detections)))
            matched_players = set()
            
            # Simple greedy matching
            for player_id in active_players:
                if len(unmatched_detections) == 0:
                    break
                    
                player_info = self.players[player_id]
                best_match_idx = -1
                best_score = 0
                
                for i, det_idx in enumerate(unmatched_detections):
                    det = detections[det_idx]
                    det_features = detection_features[det_idx]
                    
                    # Calculate similarity scores
                    feature_sim = feature_similarity(player_info['features'], det_features)
                    
                    # Position-based similarity
                    if player_info['last_bbox'] is not None:
                        iou_score = calculate_iou(player_info['last_bbox'], det['bbox'])
                        distance = euclidean_distance(player_info['last_bbox'], det['bbox'])
                        distance_score = max(0, 1 - distance / 200)  # Normalize distance
                        
                        # Combined score
                        combined_score = 0.5 * feature_sim + 0.3 * iou_score + 0.2 * distance_score
                    else:
                        combined_score = feature_sim
                    
                    if combined_score > best_score and combined_score > self.similarity_threshold:
                        best_score = combined_score
                        best_match_idx = i
                
                # Assign best match
                if best_match_idx >= 0:
                    det_idx = unmatched_detections[best_match_idx]
                    det = detections[det_idx]
                    
                    # Update player info with exponential moving average
                    alpha = 0.3  # Learning rate
                    self.players[player_id]['features'] = (1-alpha) * self.players[player_id]['features'] + alpha * detection_features[det_idx]
                    self.players[player_id]['last_bbox'] = det['bbox']
                    self.players[player_id]['last_seen'] = self.frame_count
                    self.players[player_id]['confidence'] = det['confidence']
                    self.players[player_id]['disappeared'] = 0
                    
                    current_assignments[det_idx] = player_id
                    matched_players.add(player_id)
                    unmatched_detections.remove(det_idx)
            
            # Mark unmatched players as disappeared
            for player_id in active_players:
                if player_id not in matched_players:
                    self.players[player_id]['disappeared'] += 1
            
            # Create new players for unmatched detections
            for det_idx in unmatched_detections:
                det = detections[det_idx]
                new_player_id = self.next_id
                self.next_id += 1
                
                self.players[new_player_id] = {
                    'features': detection_features[det_idx],
                    'last_bbox': det['bbox'],
                    'last_seen': self.frame_count,
                    'confidence': det['confidence'],
                    'first_seen': self.frame_count,
                    'disappeared': 0
                }
                
                current_assignments[det_idx] = new_player_id
        
        return current_assignments

# ========================= MAIN PROCESSING FUNCTION =========================

def process_video(video_path, output_path=None):
    """Process video and perform player re-identification"""
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize tracker
    tracker = PlayerTracker()
    
    # Results storage
    results = {
        'frame_results': [],
        'player_tracks': defaultdict(list),
        'video_info': {
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': total_frames
        }
    }
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = 'temp_output.avi'
    out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    
    if not out.isOpened():
        print("Warning: Could not create video writer")
        output_path = None
    
    frame_idx = 0
    colors = {}  # player_id -> color for visualization
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results_yolo = model(frame, verbose=False, conf=0.3, iou=0.5)
            
            # Extract detections
            detections = []
            if len(results_yolo) > 0 and results_yolo[0].boxes is not None:
                boxes = results_yolo[0].boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # Filter detections (adjust class if needed)
                    if conf > 0.4:  # Confidence threshold
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': conf,
                            'class': cls
                        })
            
            # Update tracker
            assignments = tracker.update(detections, frame)
            
            # Visualize results
            vis_frame = frame.copy()
            
            for det_idx, detection in enumerate(detections):
                bbox = detection['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                if det_idx in assignments:
                    player_id = assignments[det_idx]
                    
                    # Assign color to player if not exists
                    if player_id not in colors:
                        colors[player_id] = (
                            np.random.randint(50, 255),
                            np.random.randint(50, 255),
                            np.random.randint(50, 255)
                        )
                    
                    color = colors[player_id]
                    
                    # Draw bounding box
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Add player ID label with background
                    label = f"Player {player_id}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Draw label background
                    cv2.rectangle(vis_frame, 
                                (x1, y1 - text_height - 10), 
                                (x1 + text_width, y1), 
                                color, -1)
                    
                    # Draw text
                    cv2.putText(vis_frame, label, (x1, y1 - 5), 
                              font, font_scale, (255, 255, 255), thickness)
                    
                    # Add confidence score
                    conf_label = f"{detection['confidence']:.2f}"
                    cv2.putText(vis_frame, conf_label, (x2 - 50, y2 - 5), 
                              font, 0.5, color, 1)
                else:
                    # Unassigned detection (shouldn't happen with current logic)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
            
            # Add frame info
            frame_info = f"Frame: {frame_idx}, Players: {len(assignments)}"
            cv2.putText(vis_frame, frame_info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Store frame results
            frame_result = {
                'frame': frame_idx,
                'detections': [],
                'assignments': assignments
            }
            
            for det_idx, detection in enumerate(detections):
                if det_idx in assignments:
                    player_id = assignments[det_idx]
                    frame_result['detections'].append({
                        'player_id': player_id,
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence']
                    })
                    
                    # Add to player tracks
                    results['player_tracks'][player_id].append({
                        'frame': frame_idx,
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence']
                    })
            
            results['frame_results'].append(frame_result)
            
            # Write frame to video
            if output_path:
                out.write(vis_frame)
            
            # Progress update
            if frame_idx % 30 == 0 or frame_idx < 10:
                print(f"Processed frame {frame_idx}/{total_frames} - Detected {len(detections)} players")
            
            frame_idx += 1
    
    finally:
        cap.release()
        if output_path:
            out.release()
            
            # Convert to MP4 using ffmpeg
            if output_path:
                print("Converting video to MP4...")
                cmd = f'ffmpeg -i {temp_output} -c:v libx264 -preset medium -crf 23 -c:a aac {output_path} -y -loglevel quiet'
                os.system(cmd)
                
                # Remove temporary file
                if os.path.exists(temp_output):
                    os.remove(temp_output)
    
    print(f"\nProcessing complete!")
    print(f"- Processed {frame_idx} frames")
    print(f"- Tracked {len(results['player_tracks'])} unique players")
    
    return results

# ========================= ANALYSIS FUNCTIONS =========================

def analyze_results(results):
    """Analyze tracking results and generate statistics"""
    
    print("\n=== TRACKING ANALYSIS ===")
    print(f"Total frames processed: {len(results['frame_results'])}")
    print(f"Total unique players tracked: {len(results['player_tracks'])}")
    
    # Player statistics
    for player_id, track in results['player_tracks'].items():
        print(f"\nPlayer {player_id}:")
        print(f"  - Appearances: {len(track)} frames")
        print(f"  - First seen: frame {track[0]['frame']}")
        print(f"  - Last seen: frame {track[-1]['frame']}")
        print(f"  - Average confidence: {np.mean([t['confidence'] for t in track]):.3f}")
    
    # Frame-by-frame detection count
    detections_per_frame = [len(fr['detections']) for fr in results['frame_results']]
    if detections_per_frame:
        print(f"\nDetections per frame:")
        print(f"  - Average: {np.mean(detections_per_frame):.2f}")
        print(f"  - Min: {np.min(detections_per_frame)}")
        print(f"  - Max: {np.max(detections_per_frame)}")

def save_results(results, output_file):
    """Save results to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'frame_results': [],
        'player_tracks': {},
        'video_info': results['video_info']
    }
    
    for frame_result in results['frame_results']:
        json_frame = {
            'frame': frame_result['frame'],
            'detections': [],
            'assignments': frame_result['assignments']
        }
        
        for det in frame_result['detections']:
            json_det = {
                'player_id': det['player_id'],
                'bbox': [float(x) for x in det['bbox']],
                'confidence': float(det['confidence'])
            }
            json_frame['detections'].append(json_det)
        
        json_results['frame_results'].append(json_frame)
    
    for player_id, track in results['player_tracks'].items():
        json_track = []
        for t in track:
            json_track.append({
                'frame': t['frame'],
                'bbox': [float(x) for x in t['bbox']],
                'confidence': float(t['confidence'])
            })
        json_results['player_tracks'][str(player_id)] = json_track
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_file}")

# ========================= MAIN EXECUTION =========================

# Upload your video file
from google.colab import files

print("Please upload your video file (15sec_input_720p.mp4):")
uploaded = files.upload()

# Get the uploaded video file name
video_filename = list(uploaded.keys())[0]
print(f"Processing video: {video_filename}")

# Process the video
try:
    print("\n" + "="*50)
    print("STARTING VIDEO PROCESSING")
    print("="*50)
    
    results = process_video(video_filename, output_path="output_with_tracking.mp4")
    
    # Analyze results
    analyze_results(results)
    
    # Save results
    save_results(results, "tracking_results.json")
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE!")
    print("="*50)
    
    # Check output files
    if os.path.exists("output_with_tracking.mp4"):
        file_size = os.path.getsize("output_with_tracking.mp4")
        print(f"✅ output_with_tracking.mp4 created successfully ({file_size/1024/1024:.1f} MB)")
    else:
        print("❌ Video file was not created")
    
    if os.path.exists("tracking_results.json"):
        print("✅ tracking_results.json created successfully")
    
    # Display sample results
    print(f"\n📊 SAMPLE RESULTS:")
    for i, frame_result in enumerate(results['frame_results'][:5]):
        detections = frame_result['detections']
        if detections:
            print(f"Frame {frame_result['frame']}: {len(detections)} players")
            for det in detections[:3]:  # Show first 3 detections
                print(f"  - Player {det['player_id']}: confidence {det['confidence']:.3f}")
    
    print(f"\n🎯 TRACKING SUMMARY:")
    print(f"- Total frames processed: {len(results['frame_results'])}")
    print(f"- Unique players tracked: {len(results['player_tracks'])}")
    print(f"- Video output: output_with_tracking.mp4")
    print(f"- Data output: tracking_results.json")

except Exception as e:
    print(f"❌ Error during processing: {str(e)}")
    import traceback
    print("\nFull error traceback:")
    traceback.print_exc()

# Download the results
print("\n" + "="*50)
print("DOWNLOADING RESULTS")
print("="*50)

try:
    if os.path.exists("output_with_tracking.mp4"):
        print("📥 Downloading output_with_tracking.mp4...")
        files.download("output_with_tracking.mp4")
        print("✅ Video download complete!")
    
    if os.path.exists("tracking_results.json"):
        print("📥 Downloading tracking_results.json...")
        files.download("tracking_results.json")
        print("✅ JSON download complete!")
        
except Exception as e:
    print(f"❌ Download error: {e}")

print("\n🎉 ALL DONE! Your soccer player re-identification solution is ready!")
print("The output video shows players with consistent IDs and tracking throughout the video.")
