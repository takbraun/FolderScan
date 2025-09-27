import os
import shutil
import json
from datetime import datetime
from pathlib import Path
import threading
from collections import defaultdict
import mimetypes
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging
import tempfile
import psutil 
import glob 

# Suppress verbose FFMPEG logs from OpenCV
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '0'

# Try to import NudeNet
try:
    from nudenet import NudeDetector
    NUDENET_AVAILABLE = True
except ImportError:
    NUDENET_AVAILABLE = False
    print("WARNING: NudeNet not installed. Install with: pip install nudenet")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SuggestiveContentClassifier:
    """
    A simple, heuristic-based classifier to detect suggestive content like lingerie
    that might be missed by the primary nudity detector. This serves as a second pass.
    """
    def __init__(self):
        # These are just example heuristics. A real implementation would use a trained model.
        self.skin_tone_threshold = 0.3  # Min proportion of skin to be considered
        self.dark_color_threshold = 0.1 # Min proportion of black/reds
        self.suggestive_threshold = 0.5 # Default sensitivity

    def set_threshold(self, threshold):
        if threshold is not None:
            self.suggestive_threshold = threshold
        logger.info(f"Suggestive Content Classifier threshold set to: {self.suggestive_threshold}")

    def analyze_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            img_array = np.array(image)

            # 1. Skin tone analysis
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / (img_array.shape[0] * img_array.shape[1])

            if skin_ratio < self.skin_tone_threshold:
                return 0 # Not enough skin to be considered suggestive

            # 2. Color analysis (looking for common lingerie colors like black, red)
            # Define color ranges in HSV
            # Red
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            # Red (wrap-around)
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])
            # Black
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 30])

            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            black_mask = cv2.inRange(hsv, lower_black, upper_black)
            
            dark_color_ratio = (np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0) + np.sum(black_mask > 0)) / (img_array.shape[0] * img_array.shape[1])

            # Combine heuristics into a final score
            score = (skin_ratio * 0.6) + (dark_color_ratio * 0.4)
            
            return score
        except Exception as e:
            logger.error(f"Suggestive classifier error for {image_path}: {e}")
            return 0


class NSFWDetector:
    def __init__(self, model_type='nudenet', fallback_to_simple=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.fallback_to_simple = fallback_to_simple
        self.nude_detector = None
        self.suggestive_classifier = SuggestiveContentClassifier()
        
        self.exposed_threshold = 0.4
        self.covered_threshold = 0.6
        self.final_threshold = 0.4
        self.video_threshold = 0.3
        
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.1f}GB VRAM")
            if "2060" in gpu_name:
                logger.info("RTX 2060 detected - applying optimizations")
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
        else:
            logger.info("Using CPU mode - no GPU detected")
        
        if model_type == 'nudenet' and NUDENET_AVAILABLE:
            try:
                logger.info("Initializing NudeNet model...")
                self.nude_detector = NudeDetector()
                logger.info("✅ NudeNet model loaded successfully!")
                self.model_type = 'nudenet'
            except Exception as e:
                logger.error(f"Failed to load NudeNet: {e}")
                if fallback_to_simple:
                    logger.warning("Falling back to simple detection")
                    self.model_type = 'simple'
                else:
                    raise
        else:
            if model_type == 'nudenet' and not NUDENET_AVAILABLE:
                logger.warning("NudeNet requested but not available")
            if fallback_to_simple:
                logger.info("Using simple skin-tone detection")
                self.model_type = 'simple'
            else:
                raise ValueError("No valid detection method available")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.batch_size = int(os.getenv('IMAGE_BATCH_SIZE', '16' if torch.cuda.is_available() else '8'))
    
    def _nudenet_detection(self, image_path):
        try:
            detections = self.nude_detector.detect(str(image_path))
            nsfw_labels = {
                'FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED',
                'BUTTOCKS_EXPOSED', 'ANUS_EXPOSED', 'FEMALE_BREAST_COVERED',
                'FEMALE_GENITALIA_COVERED', 'MALE_GENITALIA_COVERED'
            }
            max_score = 0
            for detection in detections:
                label = detection.get('class', '')
                score = detection.get('score', 0)
                if label in nsfw_labels:
                    if 'EXPOSED' in label and score > self.exposed_threshold:
                        max_score = max(max_score, score)
                    elif 'COVERED' in label and score > self.covered_threshold:
                        max_score = max(max_score, score * 0.7)
            return max_score
        except Exception as e:
            logger.error(f"NudeNet detection error for {image_path}: {e}")
            if self.fallback_to_simple:
                return self._simple_content_detection_from_path(image_path)
            return 0
    
    def set_thresholds(self, exposed=None, covered=None, final=None, video=None, suggestive=None):
        if exposed is not None: self.exposed_threshold = exposed
        if covered is not None: self.covered_threshold = covered
        if final is not None: self.final_threshold = final
        if video is not None: self.video_threshold = video
        if suggestive is not None: self.suggestive_classifier.set_threshold(suggestive)
        logger.info(f"Updated thresholds - Exposed: {self.exposed_threshold}, Covered: {self.covered_threshold}, Final: {self.final_threshold}, Video: {self.video_threshold}, Suggestive: {self.suggestive_classifier.suggestive_threshold}")

    def _simple_content_detection_from_path(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            img_array = np.array(image)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / (img_array.shape[0] * img_array.shape[1])
            return skin_ratio
        except Exception as e:
            logger.error(f"Simple detection error for {image_path}: {e}")
            return 0
            
    def detect_image(self, image_path, enable_suggestive_scan=False):
        try:
            score = 0
            if self.model_type == 'nudenet' and self.nude_detector:
                score = self._nudenet_detection(image_path)
            else:
                score = self._simple_content_detection_from_path(image_path)
            
            is_nsfw = score > self.final_threshold
            logger.info(f"IMAGE RESULT for {Path(image_path).name}: {'NSFW' if is_nsfw else 'SAFE'} (Score: {score:.2f})")
            
            if not is_nsfw and enable_suggestive_scan:
                suggestive_score = self.suggestive_classifier.analyze_image(image_path)
                is_suggestive = suggestive_score > self.suggestive_classifier.suggestive_threshold
                logger.info(f"SUGGESTIVE SCAN for {Path(image_path).name}: {'DETECTED' if is_suggestive else 'CLEAR'} (Score: {suggestive_score:.2f})")
                return is_suggestive

            return is_nsfw
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return False
    
    def detect_video(self, video_path, sample_frames=20, enable_suggestive_scan=False):
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return False
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 0: return False
            
            frame_indices = np.linspace(0, frame_count - 1, min(sample_frames, frame_count), dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret: continue

                temp_path = None
                try:
                    temp_dir = tempfile.gettempdir()
                    temp_path = os.path.join(temp_dir, f"temp_frame_{os.getpid()}_{frame_idx}.jpg")
                    cv2.imwrite(temp_path, frame)
                    
                    if self.detect_image(temp_path, enable_suggestive_scan):
                         logger.info(f"VIDEO RESULT for {Path(video_path).name}: NSFW (frame {frame_idx})")
                         return True
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
            
            logger.info(f"VIDEO RESULT for {Path(video_path).name}: SAFE")
            return False
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}", exc_info=True)
            return False
        finally:
            if cap: cap.release()

    def get_model_info(self):
        primary_model = {'type': 'NudeNet', 'status': 'Active', 'accuracy': 'High', 'description': 'Primary nudity/explicit content detector.'}
        if self.model_type != 'nudenet':
            primary_model = {'type': 'Simple Skin Detection', 'status': 'Active', 'accuracy': 'Low', 'description': 'Basic heuristic detector.'}
        
        secondary_model = {'type': 'Suggestive Content Classifier', 'status': 'Available', 'accuracy': 'Medium', 'description': 'Secondary check for suggestive clothing/poses.'}
        
        return {'primary': primary_model, 'secondary': secondary_model}

class ContentScanner:
    def __init__(self, nsfw_detector, enable_suggestive_scan=False):
        self.detector = nsfw_detector
        self.enable_suggestive_scan = enable_suggestive_scan
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        self.scan_results = defaultdict(list)
        self.stats = {'images': 0, 'videos': 0, 'nsfw_images': 0, 'nsfw_videos': 0, 'total_processed': 0}
        
    def get_file_type(self, file_path):
        ext = Path(file_path).suffix.lower()
        if ext in self.supported_image_formats: return 'image'
        if ext in self.supported_video_formats: return 'video'
        return None
    
    def scan_directory(self, source_dir, nsfw_dir, progress_callback=None):
        source_path, nsfw_path = Path(source_dir), Path(nsfw_dir)
        nsfw_images_dir = nsfw_path / 'images'
        nsfw_videos_dir = nsfw_path / 'videos'
        nsfw_images_dir.mkdir(parents=True, exist_ok=True)
        nsfw_videos_dir.mkdir(parents=True, exist_ok=True)
        
        all_files = []
        all_supported_formats = self.supported_image_formats | self.supported_video_formats
        logger.info(f"Starting recursive scan of '{source_dir}' using os.walk...")
        for root, _, files in os.walk(source_dir):
            for name in files:
                if Path(name).suffix.lower() in all_supported_formats:
                    all_files.append(Path(root) / name)

        logger.info(f"Found {len(all_files)} total media files to process.")

        total_files = len(all_files)
        for i, file_path in enumerate(all_files):
            try:
                logger.info(f"Processing file {i + 1}/{total_files}: {file_path.name}")
                file_type = self.get_file_type(file_path)
                if not file_type: continue
                
                self.stats['total_processed'] += 1
                if file_type == 'image': self.stats['images'] += 1
                else: self.stats['videos'] += 1

                is_nsfw = self.detector.detect_image(file_path, self.enable_suggestive_scan) if file_type == 'image' else self.detector.detect_video(file_path, enable_suggestive_scan=self.enable_suggestive_scan)

                if is_nsfw:
                    dest_dir = nsfw_images_dir if file_type == 'image' else nsfw_videos_dir
                    if file_type == 'image': self.stats['nsfw_images'] += 1
                    else: self.stats['nsfw_videos'] += 1
                    
                    dest_path = dest_dir / file_path.relative_to(source_path)
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(file_path), str(dest_path))
                    self.scan_results[file_type].append({'moved_to': str(dest_path), 'size_bytes': dest_path.stat().st_size})
            except Exception as e:
                logger.error(f"Could not process file {file_path}: {e}")
            finally:
                if progress_callback:
                    progress_callback(i + 1, total_files, self.stats.copy())
        
        return self.generate_report()

    def generate_report(self):
        total_size_moved = sum(f['size_bytes'] for files in self.scan_results.values() for f in files)
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files_processed': self.stats['total_processed'],
                'images_processed': self.stats['images'],
                'videos_processed': self.stats['videos'],
                'nsfw_images_moved': self.stats['nsfw_images'],
                'nsfw_videos_moved': self.stats['nsfw_videos'],
                'total_nsfw_moved': self.stats['nsfw_images'] + self.stats['nsfw_videos'],
                'total_size_moved_mb': round(total_size_moved / (1024 * 1024), 2),
            },
            'model_used': self.detector.get_model_info()
        }
        return report

app = Flask(__name__)
CORS(app)
detector = NSFWDetector()
scan_status = {'running': False, 'progress': 0, 'total': 0, 'stats': {}}

def update_progress(current, total, stats):
    scan_status.update({'progress': current, 'total': total, 'stats': stats})

@app.route('/browse')
def browse_directory():
    path = request.args.get('path', '/')
    items = []
    if path == '/':
        home = os.path.expanduser('~')
        common_dirs = [('🏠 Home', home), ('🖼️ Pictures', os.path.join(home, 'Pictures')), ('📥 Downloads', os.path.join(home, 'Downloads'))]
        for name, fpath in common_dirs:
            if os.path.exists(fpath): items.append({'name': name, 'path': fpath, 'type': 'folder'})
        try:
            for p in psutil.disk_partitions(all=True):
                if p.mountpoint and ' ' not in p.mountpoint and not p.mountpoint.startswith(('/snap', '/boot')):
                    items.append({'name': f'💾 {p.device} ({p.mountpoint})', 'path': p.mountpoint, 'type': 'mount'})
        except Exception: pass
        try:
            for gvfs_path in glob.glob('/run/user/*/gvfs/*'):
                if os.path.isdir(gvfs_path):
                    share_name = os.path.basename(gvfs_path).replace('smb-share:server=', '').replace(',', ', share=')
                    items.append({'name': f'🌐 {share_name}', 'path': gvfs_path, 'type': 'mount'})
        except Exception: pass
    else:
        try:
            if not os.path.exists(path): return jsonify({'error': 'Path does not exist'}), 404
            if path != '/': items.append({'name': '⬆️ .. (parent)', 'path': os.path.dirname(path), 'type': 'parent'})
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path): items.append({'name': f'📁 {item}', 'path': item_path, 'type': 'directory'})
        except Exception as e: return jsonify({'error': str(e)}), 500
    return jsonify({'current_path': path, 'items': items})

@app.route('/model-info')
def get_model_info_route():
    return jsonify(detector.get_model_info())

@app.route('/get-thresholds')
def get_thresholds_route():
    return jsonify({
        'exposed': detector.exposed_threshold, 'covered': detector.covered_threshold,
        'final': detector.final_threshold, 'video': detector.video_threshold,
        'suggestive': detector.suggestive_classifier.suggestive_threshold
    })

@app.route('/update-thresholds', methods=['POST'])
def update_thresholds_route():
    data = request.json
    detector.set_thresholds(
        exposed=data.get('exposed'), covered=data.get('covered'),
        final=data.get('final'), video=data.get('video'),
        suggestive=data.get('suggestive')
    )
    return jsonify({'status': 'success'})

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>NSFW Content Scanner</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }
            input, button, select { padding: 10px; margin: 5px; border-radius: 4px; border: 1px solid #ccc; }
            button { background: #007bff; color: white; cursor: pointer; }
            button:hover { background: #0056b3; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            button.browse-btn { background: #28a745; }
            button.browse-btn:hover { background: #218838; }
            .progress-bar { width: 100%; height: 20px; background: #e0e0e0; border-radius: 10px; margin: 10px 0; }
            .progress-fill { height: 100%; background: #4caf50; border-radius: 10px; transition: width 0.3s; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 20px 0; }
            .stat-box { background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .results { background: white; padding: 15px; border-radius: 8px; margin: 20px 0; }
            .folder-browser { background: white; border: 1px solid #ddd; border-radius: 8px; max-height: 300px; overflow-y: auto; display: none; }
            .folder-item { padding: 10px; border-bottom: 1px solid #eee; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }
            .folder-item:hover { background: #f8f9fa; }
            .folder-item.parent { background: #e3f2fd; font-weight: bold; }
            .current-path { background: #e9ecef; padding: 10px; border-radius: 4px; margin: 10px 0; font-family: monospace; }
            .input-group { margin: 15px 0; }
            .input-group label { display: block; margin-bottom: 5px; font-weight: bold; }
            .input-row { display: flex; gap: 10px; align-items: center; }
            .path-input { flex: 1; }
            .model-info { background: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 4px; margin: 10px 0; }
            .threshold-control { display: flex; align-items: center; gap: 10px; margin: 10px 0; }
            .threshold-slider { flex: 1; }
            .threshold-value { width: 50px; text-align: center; font-weight: bold; }
            .threshold-label { width: 150px; }
            .presets-container { display: flex; gap: 10px; margin: 15px 0; }
            .preset-btn { padding: 8px 16px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer; }
            .preset-btn:hover { background: #5a6268; }
            .suggestive-scan-options { border-top: 1px solid #ccc; margin-top: 20px; padding-top: 15px; }
        </style>
    </head>
    <body>
        <h1>NSFW Content Scanner & Directory Cleanser</h1>
        <div class="container">
            <h3>Detection Model Status</h3>
            <div id="modelInfo"></div>
        </div>
        <div class="container">
            <h3>Detection Sensitivity Settings</h3>
            <div class="presets-container">
                <button class="preset-btn" onclick="setPreset('strict')">Strict</button>
                <button class="preset-btn" onclick="setPreset('balanced')">Balanced</button>
                <button class="preset-btn" onclick="setPreset('relaxed')">Relaxed</button>
            </div>
            <div class="threshold-control">
                <span class="threshold-label">Exposed Content:</span>
                <input type="range" id="exposedThreshold" class="threshold-slider" min="0.1" max="0.9" step="0.05">
                <span id="exposedValue" class="threshold-value"></span>
            </div>
            <div class="threshold-control">
                <span class="threshold-label">Covered Content:</span>
                <input type="range" id="coveredThreshold" class="threshold-slider" min="0.1" max="0.9" step="0.05">
                <span id="coveredValue" class="threshold-value"></span>
            </div>
            <div class="threshold-control">
                <span class="threshold-label">Overall (Images):</span>
                <input type="range" id="finalThreshold" class="threshold-slider" min="0.1" max="0.9" step="0.05">
                <span id="finalValue" class="threshold-value"></span>
            </div>
            <div class="threshold-control">
                <span class="threshold-label">Video Frames:</span>
                <input type="range" id="videoThreshold" class="threshold-slider" min="0.1" max="0.9" step="0.05">
                <span id="videoValue" class="threshold-value"></span>
            </div>
             <div class="suggestive-scan-options">
                <h4>Second Pass (Suggestive Content)</h4>
                <div class="input-group">
                    <label style="display:inline-block; margin-right: 10px;">
                        <input type="checkbox" id="enableSuggestiveScan">
                        Enable Suggestive Content Scan (Second Pass)
                    </label>
                </div>
                 <div class="threshold-control">
                    <span class="threshold-label">Suggestive Sensitivity:</span>
                    <input type="range" id="suggestiveThreshold" class="threshold-slider" min="0.1" max="0.9" step="0.05">
                    <span id="suggestiveValue" class="threshold-value"></span>
                </div>
            </div>
            <button onclick="updateThresholds()">Apply Settings</button>
            <span id="thresholdStatus" style="margin-left: 10px; color: green; display: none;">✅ Settings Updated</span>
        </div>
        <div class="container">
            <h3>Directory Configuration</h3>
             <div class="input-group">
                <label>Source Directory:</label>
                <div class="input-row">
                    <input type="text" id="sourceDir" class="path-input">
                    <button onclick="browse('sourceDir')" class="browse-btn">Browse</button>
                </div>
            </div>
            <div class="input-group">
                <label>NSFW Quarantine Directory:</label>
                <div class="input-row">
                    <input type="text" id="nsfwDir" class="path-input">
                    <button onclick="browse('nsfwDir')" class="browse-btn">Browse</button>
                </div>
            </div>
            <div id="browserModal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5);">
                <div style="background:white; width:50%; margin: 5% auto; padding:20px;">
                    <div class="current-path" id="browser-path">/</div>
                    <div id="folderBrowser" class="folder-browser" style="display:block;"></div>
                    <button onclick="selectCurrentFolder()">Select Current Folder</button>
                    <button onclick="closeBrowser()">Cancel</button>
                </div>
            </div>
             <div class="container" style="padding: 10px; background: #fffbe6; border: 1px solid #ffe58f;">
                <details>
                    <summary><strong>Troubleshooting Network Drives?</strong></summary>
                    <div style="margin-top: 10px;">
                        <p>If your network drive doesn't appear, find its path manually:</p>
                        <ol>
                            <li>Open your <strong>Terminal</strong>.</li>
                            <li>Type <code>ls -d </code> (with a space) into the terminal.</li>
                            <li>Drag the network folder from your file manager and drop it into the terminal.</li>
                            <li>The full path will appear. Copy and paste it into the "Source Directory" box.</li>
                        </ol>
                    </div>
                </details>
            </div>
        </div>
        <button onclick="startScan()" id="scanBtn">Start Directory Cleansing</button>
        <div id="progressContainer" style="display: none;">
            <h3>Scan Progress</h3>
            <div class="progress-bar"><div id="progressFill" class="progress-fill"></div></div>
            <p id="progressText">0 / 0 files</p>
            <div class="stats">
                <div class="stat-box"><h4>Images</h4><p id="imagesCount">0</p></div>
                <div class="stat-box"><h4>Videos</h4><p id="videosCount">0</p></div>
                <div class="stat-box"><h4>NSFW Images</h4><p id="nsfwImagesCount">0</p></div>
                <div class="stat-box"><h4>NSFW Videos</h4><p id="nsfwVideosCount">0</p></div>
            </div>
        </div>
        <div id="results"></div>
        <script>
            let currentTargetInput = '';
            let progressInterval;

            function browse(targetId) {
                currentTargetInput = targetId;
                document.getElementById('browserModal').style.display = 'block';
                loadPath('/');
            }
            function loadPath(path) {
                document.getElementById('browser-path').textContent = path;
                fetch(`/browse?path=${encodeURIComponent(path)}`).then(r => r.json()).then(data => {
                    const browser = document.getElementById('folderBrowser');
                    browser.innerHTML = '';
                    browser.dataset.currentPath = data.current_path;
                    data.items.forEach(item => {
                        const div = document.createElement('div');
                        div.textContent = item.name;
                        div.className = 'folder-item';
                        div.onclick = () => { if (item.type !== 'file') loadPath(item.path); };
                        browser.appendChild(div);
                    });
                });
            }
            function selectCurrentFolder() {
                document.getElementById(currentTargetInput).value = document.getElementById('browser-path').textContent;
                closeBrowser();
            }
            function closeBrowser() { document.getElementById('browserModal').style.display = 'none'; }
            
            function startScan() {
                const sourceDir = document.getElementById('sourceDir').value;
                const nsfwDir = document.getElementById('nsfwDir').value;
                if (!sourceDir || !nsfwDir) { alert('Please select both directories.'); return; }
                
                const payload = {
                    source_dir: sourceDir,
                    nsfw_dir: nsfwDir,
                    enable_suggestive_scan: document.getElementById('enableSuggestiveScan').checked
                };
                
                document.getElementById('scanBtn').disabled = true;
                document.getElementById('progressContainer').style.display = 'block';
                document.getElementById('results').innerHTML = '';
                
                fetch('/scan', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                }).then(r => r.json()).then(data => {
                    if (data.message === 'Scan initiated') {
                        progressInterval = setInterval(updateProgress, 1000);
                    } else {
                        alert('Error: ' + data.error);
                        document.getElementById('scanBtn').disabled = false;
                    }
                });
            }
            
            function updateProgress() {
                fetch('/status').then(r => r.json()).then(data => {
                    if (!data.running && progressInterval) {
                        clearInterval(progressInterval);
                        document.getElementById('scanBtn').disabled = false;
                        if(data.last_scan) showResults(data.last_scan);
                        return;
                    }
                    const progress = data.total > 0 ? (data.progress / data.total) * 100 : 0;
                    document.getElementById('progressFill').style.width = progress + '%';
                    document.getElementById('progressText').textContent = `${data.progress || 0} / ${data.total || 0} files processed`;
                    if (data.stats) {
                        document.getElementById('imagesCount').textContent = data.stats.images || 0;
                        document.getElementById('videosCount').textContent = data.stats.videos || 0;
                        document.getElementById('nsfwImagesCount').textContent = data.stats.nsfw_images || 0;
                        document.getElementById('nsfwVideosCount').textContent = data.stats.nsfw_videos || 0;
                    }
                });
            }

            function showResults(report) {
                 const summary = report.summary;
                 document.getElementById('results').innerHTML = `<h3>Scan Complete</h3>
                    <p>Total files processed: ${summary.total_files_processed}</p>
                    <p>NSFW images moved: ${summary.nsfw_images_moved}</p>
                    <p>NSFW videos moved: ${summary.nsfw_videos_moved}</p>
                    <p>Total size moved: ${summary.total_size_moved_mb} MB</p>`;
            }
            
            function updateThresholds() {
                const thresholds = {
                    exposed: parseFloat(document.getElementById('exposedThreshold').value),
                    covered: parseFloat(document.getElementById('coveredThreshold').value),
                    final: parseFloat(document.getElementById('finalThreshold').value),
                    video: parseFloat(document.getElementById('videoThreshold').value),
                    suggestive: parseFloat(document.getElementById('suggestiveThreshold').value)
                };
                fetch('/update-thresholds', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(thresholds)
                }).then(response => response.json()).then(data => {
                    if (data.status === 'success') {
                        const statusEl = document.getElementById('thresholdStatus');
                        statusEl.style.display = 'inline';
                        setTimeout(() => { statusEl.style.display = 'none'; }, 3000);
                    }
                });
            }
            
            function setPreset(preset) {
                const presets = {
                    strict: {exposed: 0.2, covered: 0.4, final: 0.2, video: 0.2, suggestive: 0.4},
                    balanced: {exposed: 0.4, covered: 0.6, final: 0.4, video: 0.3, suggestive: 0.5},
                    relaxed: {exposed: 0.6, covered: 0.8, final: 0.6, video: 0.5, suggestive: 0.6}
                };
                const s = presets[preset];
                document.getElementById('exposedThreshold').value = s.exposed;
                document.getElementById('coveredThreshold').value = s.covered;
                document.getElementById('finalThreshold').value = s.final;
                document.getElementById('videoThreshold').value = s.video;
                document.getElementById('suggestiveThreshold').value = s.suggestive;
                updateThresholdValues();
                updateThresholds();
            }

            function updateThresholdValues() {
                document.getElementById('exposedValue').textContent = document.getElementById('exposedThreshold').value;
                document.getElementById('coveredValue').textContent = document.getElementById('coveredThreshold').value;
                document.getElementById('finalValue').textContent = document.getElementById('finalThreshold').value;
                document.getElementById('videoValue').textContent = document.getElementById('videoThreshold').value;
                document.getElementById('suggestiveValue').textContent = document.getElementById('suggestiveThreshold').value;
            }

            window.onload = () => {
                fetch('/model-info').then(r => r.json()).then(data => {
                    document.getElementById('modelInfo').innerHTML = `<strong>Primary Model:</strong> ${data.primary.type} | <strong>Accuracy:</strong> ${data.primary.accuracy} <br>
                                                                     <strong>Secondary Model:</strong> ${data.secondary.type} | <strong>Status:</strong> ${data.secondary.status}`;
                });
                fetch('/get-thresholds').then(r => r.json()).then(data => {
                    document.getElementById('exposedThreshold').value = data.exposed;
                    document.getElementById('coveredThreshold').value = data.covered;
                    document.getElementById('finalThreshold').value = data.final;
                    document.getElementById('videoThreshold').value = data.video;
                    document.getElementById('suggestiveThreshold').value = data.suggestive;
                    updateThresholdValues();
                });
                document.getElementById('exposedThreshold').oninput = updateThresholdValues;
                document.getElementById('coveredThreshold').oninput = updateThresholdValues;
                document.getElementById('finalThreshold').oninput = updateThresholdValues;
                document.getElementById('videoThreshold').oninput = updateThresholdValues;
                document.getElementById('suggestiveThreshold').oninput = updateThresholdValues;
            };
        </script>
    </body>
    </html>
    ''')

@app.route('/scan', methods=['POST'])
def start_scan_route():
    if scan_status.get('running'):
        return jsonify({'error': 'Scan already in progress'}), 400
    
    data = request.json
    source_dir = data.get('source_dir')
    nsfw_dir = data.get('nsfw_dir')
    enable_suggestive_scan = data.get('enable_suggestive_scan', False)

    if not source_dir or not nsfw_dir or not os.path.exists(source_dir):
        return jsonify({'error': 'Invalid directories provided'}), 400

    def run_scan():
        scan_status['running'] = True
        scan_status['progress'] = 0
        scan_status['total'] = 0
        scan_status['stats'] = {}
        scan_status['last_scan'] = None
        
        try:
            scanner = ContentScanner(detector, enable_suggestive_scan)
            report = scanner.scan_directory(source_dir, nsfw_dir, update_progress)
            scan_status['last_scan'] = report
        except Exception as e:
            logger.error(f"FATAL SCAN ERROR: {e}", exc_info=True)
            scan_status['error'] = str(e)
        finally:
            scan_status['running'] = False

    thread = threading.Thread(target=run_scan)
    thread.start()
    return jsonify({'message': 'Scan initiated'})

@app.route('/status')
def status():
    return jsonify(scan_status)

if __name__ == '__main__':
    try:
        import psutil
    except ImportError:
        print("\nERROR: 'psutil' library not found. Please install it by running: pip install psutil\n")
        exit(1)
    app.run(host='0.0.0.0', port=8000, debug=False)


