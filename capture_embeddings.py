import os
from tkinter import simpledialog, messagebox

import cv2
import torch
import insightface
import time
import tkinter as tk
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# === GUI for name input ===
root = tk.Tk()
root.withdraw()  # Hide main window
name = simpledialog.askstring("Enter Name", "Enter the person's name:")

if not name:
    messagebox.showwarning("Cancelled", "No name entered. Exiting.")
    exit()
# ==========================

embedding_dir = "./embeddings/"
os.makedirs(embedding_dir, exist_ok=True)

model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

cap = cv2.VideoCapture(0)
print("Déplace ton visage devant la caméra...")

embeddings = []
frame_count = 0
max_frames = 30
start_time = time.time()
timeout_seconds = 20

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    faces = model.get(frame)

    if faces:
        face = faces[0]  # Use the first face only
        emb = torch.tensor(face.embedding)
        embeddings.append(emb)
        frame_count += 1

        # Draw box
        box = face.bbox.astype(int)
        cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({frame_count}/{max_frames})", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Move your face slowly in all directions", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Face not detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Capture", frame)

    # Break if enough frames or timeout
    if cv2.waitKey(1) == ord('q') or frame_count >= max_frames or (time.time() - start_time > timeout_seconds):
        break

cap.release()
cv2.destroyAllWindows()

# Save embedding
if embeddings:
    avg_embedding = torch.stack(embeddings).mean(dim=0)
    torch.save(avg_embedding, os.path.join(embedding_dir, f"{name}.pt"))
    print(f"✅ Saved embedding for {name}.")
else:
    print("❌ No face detected consistently. Nothing saved.")
