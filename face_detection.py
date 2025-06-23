import io
import os
import logging
import socketserver
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from http import server
from picamera2 import Picamera2
import numpy as np
import cv2
import face_recognition

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Face Recognition Stream</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {
      margin: 0;
      padding: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(to right, #141e30, #243b55);
      color: #fff;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
      min-height: 100vh;
    }

    h1 {
      margin-top: 40px;
      text-align: center;
      font-size: 2rem;
      background: linear-gradient(90deg, #00c6ff, #0072ff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .stream-container {
      margin-top: 20px;
      width: 90%;
      max-width: 800px;
      border: 4px solid #00c6ff;
      border-radius: 10px;
      overflow: hidden;
      box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }

    img {
      width: 100%;
      height: auto;
      display: block;
    }

    .btn {
      margin: 30px 0;
      padding: 12px 24px;
      font-size: 1rem;
      font-weight: bold;
      color: #fff;
      background: #00c6ff;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      text-decoration: none;
      transition: background 0.3s ease;
    }

    .btn:hover {
      background: #0072ff;
    }

    @media (max-width: 500px) {
      h1 {
        font-size: 1.5rem;
      }

      .btn {
        font-size: 0.9rem;
        padding: 10px 18px;
      }
    }
  </style>
</head>
<body>
  <h1>Facial Recognition Based Authentication Manager</h1>
  <div class="stream-container">
    <img src="stream.mjpg" alt="Live Stream">
  </div>
  <a href="/recordings" class="btn">📁 View & Download Recordings</a>
</body>
</html>
"""

# Email config
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
TO_ADDRESS = os.getenv('TO_ADDRESS')
SMTP_HOST = os.getenv('SMTP_HOST')
SMTP_PORT = os.getenv('SMTP_PORT')

def send_email_alert(subject, message):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_ADDRESS
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))

        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"Error sending email: {e}")

def send_email_async(subject, message):
    threading.Thread(target=send_email_alert, args=(subject, message), daemon=True).start()

class StreamingOutput:
    def _init_(self):
        self.frame = None
        self.condition = threading.Condition()

    def update(self, frame):
        with self.condition:
            self.frame = frame
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning('Removed streaming client %s: %s', self.client_address, str(e))
        elif self.path.startswith('/recordings'):
            recordings_dir = './recordings'
            sub_path = self.path[len('/recordings'):]  # "" or "/filename"
            full_path = os.path.join(recordings_dir, sub_path.lstrip('/'))

            if os.path.isdir(full_path) or self.path == '/recordings':
                try:
                    files = os.listdir(recordings_dir)
                    links = [f'<li><a href="/recordings/{fname}">{fname}</a></li>' for fname in files]
                    html = f"<html><body><h1>Recordings</h1><ul>{''.join(links)}</ul></body></html>"
                    encoded = html.encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.send_header('Content-Length', len(encoded))
                    self.end_headers()
                    self.wfile.write(encoded)
                except Exception as e:
                    self.send_error(500, f"Error reading recordings: {e}")
            elif os.path.isfile(full_path):
                try:
                    with open(full_path, 'rb') as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Content-Disposition', f'attachment; filename="{os.path.basename(full_path)}"')
                    self.send_header('Content-Length', len(content))
                    self.end_headers()
                    self.wfile.write(content)
                except Exception as e:
                    self.send_error(500, f"Error reading file: {e}")
            else:
                self.send_error(404)
                self.end_headers()
        else:
            filepath = '.' + self.path
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Content-Length', len(content))
                    self.end_headers()
                    self.wfile.write(content)
                except Exception as e:
                    self.send_error(500, f"Error reading file: {e}")
            else:
                self.send_error(404)
                self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

# Load admin faces
admin_encodings = []
admin_names = []

for filename in os.listdir("admins"):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join("admins", filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            admin_encodings.append(encodings[0])
            admin_names.append(os.path.splitext(filename)[0])

print(f"Loaded {len(admin_encodings)} admin face(s).")

def camera_thread(picam2, output):
    recording = False
    out_writer = None

    while True:
        frame = picam2.capture_array()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_present = len(face_encodings) > 0

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(admin_encodings, face_encoding, tolerance=0.6)
            is_admin = any(matches)
            name = "Authorized" if is_admin else "Unauthorized"
            color = (0, 255, 0) if is_admin else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Send email alert
            send_email_async(f"{name} access detected", f"{name} person has accessed the device.")

        if face_present:
            if not recording:
                print("Started recording...")
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                height, width, _ = frame.shape
                out_writer = cv2.VideoWriter(f"recording_{timestamp}.avi",
                                             cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))
                recording = True
            out_writer.write(frame)
        elif recording:
            print("Stopped recording.")
            if out_writer:
                out_writer.release()
            recording = False

        _, jpeg = cv2.imencode('.jpg', frame)
        output.update(jpeg.tobytes())

# Initialize camera and start everything
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(video_config)
picam2.start()

output = StreamingOutput()
threading.Thread(target=camera_thread, args=(picam2, output), daemon=True).start()

try:
    address = ('', 8080)
    server = StreamingServer(address, StreamingHandler)
    print("Server started at http://localhost:8080")
    server.serve_forever()
finally:
    picam2.stop()
