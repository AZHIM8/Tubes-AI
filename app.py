from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from gtts import gTTS
import pygame
import os
import threading
from queue import Queue
import time
import signal

app = Flask(__name__)

# Download model if not exists
def ensure_model():
    model_path = 'yolov8n.pt'
    if not os.path.exists(model_path):
        print("Downloading YOLOv8n model...")
        model = YOLO('yolov8n')  # This will automatically download the model
        print("Model downloaded successfully!")
    return YOLO(model_path)

# Inisialisasi YOLOv8
model = ensure_model()
pygame.mixer.init()

# Konfigurasi folder audio
AUDIO_FOLDER = 'audio'
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

# Antrian untuk menyimpan objek yang terdeteksi
detection_queue = Queue()
last_announcement_time = 0
ANNOUNCEMENT_DELAY = 5  # delay dalam detik antara pengumuman
last_detected_counts = {}  # Untuk melacak perubahan jumlah objek

# Status deteksi dan thread
is_detecting = False
detection_thread = None
camera_thread = None
stop_threads = False

def get_current_detections(results):
    current_counts = {'orang': 0, 'mobil': 0, 'motor': 0}
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            conf = float(box.conf[0])
            
            if conf > 0.6 and class_name in ['person', 'car', 'motorcycle']:
                id_names = {'person': 'orang', 'car': 'mobil', 'motorcycle': 'motor'}
                id_name = id_names[class_name]
                current_counts[id_name] += 1
    
    return current_counts

def announce_detections():
    global last_announcement_time, last_detected_counts, stop_threads
    while not stop_threads:
        try:
            current_time = time.time()
            if current_time - last_announcement_time > ANNOUNCEMENT_DELAY:
                current_counts = {'orang': 0, 'mobil': 0, 'motor': 0}
                
                while not detection_queue.empty():
                    counts = detection_queue.get()
                    for obj, count in counts.items():
                        current_counts[obj] = max(current_counts[obj], count)
                
                if current_counts != last_detected_counts and any(current_counts.values()):
                    announcement_parts = []
                    for obj, count in current_counts.items():
                        if count > 0:
                            announcement_parts.append(f"{count} {obj}")
                    
                    if announcement_parts:
                        announcement = "Di depan ada " + ", ".join(announcement_parts)
                        try:
                            tts = gTTS(text=announcement, lang='id')
                            audio_file = os.path.join(AUDIO_FOLDER, f'announcement_{int(current_time * 1000)}.mp3')
                            tts.save(audio_file)
                            
                            pygame.mixer.music.load(audio_file)
                            pygame.mixer.music.play()
                            
                            while pygame.mixer.music.get_busy():
                                if stop_threads:
                                    break
                                pygame.time.Clock().tick(10)
                            
                            # Hapus file audio lama
                            for file in os.listdir(AUDIO_FOLDER):
                                if file.startswith('announcement_') and file != os.path.basename(audio_file):
                                    try:
                                        os.remove(os.path.join(AUDIO_FOLDER, file))
                                    except:
                                        pass
                            
                            last_detected_counts = current_counts.copy()
                            last_announcement_time = current_time
                        except Exception as e:
                            print(f"Error in audio processing: {e}")
                
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in announce_detections: {e}")
            if stop_threads:
                break
            time.sleep(1)

def cleanup():
    global stop_threads, is_detecting
    print("Membersihkan resources...")
    stop_threads = True
    is_detecting = False
    
    # Tunggu threads selesai
    if detection_thread and detection_thread.is_alive():
        detection_thread.join(timeout=2)
    
    # Hapus file audio
    for file in os.listdir(AUDIO_FOLDER):
        if file.startswith('announcement_'):
            try:
                os.remove(os.path.join(AUDIO_FOLDER, file))
            except:
                pass

def signal_handler(signum, frame):
    print("Sinyal shutdown diterima, membersihkan...")
    cleanup()
    os._exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global is_detecting, detection_thread, stop_threads
    if not is_detecting:
        stop_threads = False
        is_detecting = True
        detection_thread = threading.Thread(target=announce_detections, daemon=True)
        detection_thread.start()
    return jsonify({'status': 'success'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global is_detecting, stop_threads
    is_detecting = False
    stop_threads = True
    return jsonify({'status': 'success'})

def generate_frames():
    global is_detecting
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            if is_detecting:
                try:
                    results = model(frame)
                    current_counts = get_current_detections(results)
                    
                    if any(current_counts.values()):
                        detection_queue.put(current_counts)
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            class_id = int(box.cls[0])
                            class_name = result.names[class_id]
                            conf = float(box.conf[0])
                            
                            if conf > 0.6 and class_name in ['person', 'car', 'motorcycle']:
                                id_names = {'person': 'orang', 'car': 'mobil', 'motorcycle': 'motor'}
                                id_name = id_names[class_name]
                                
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f'{id_name} {conf:.2f}', (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error in detection: {e}")
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    finally:
        cap.release()

if __name__ == "__main__":
    try:
        app.run(debug=True, use_reloader=False)
    finally:
        cleanup()