# face_recognizer.py
import os
import cv2
import torch
import insightface
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load model
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

# Load embeddings
embeddings_dir = "./embeddings/"
all_people_faces = {}

for file in os.listdir(embeddings_dir):
    if file.endswith(".pt"):
        name = file.replace(".pt", "")
        emb = torch.load(os.path.join(embeddings_dir, file))
        all_people_faces[name] = emb.unsqueeze(0)  # shape [1, 512]

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b)

# Start camera
cap = cv2.VideoCapture(0)
print("Running face recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    faces = model.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        emb = torch.tensor(face.embedding).unsqueeze(0)

        best_match = "Unknown"
        best_score = 0.0
        for name, known_emb in all_people_faces.items():
            sim = cosine_similarity(emb, known_emb)
            if sim > best_score and sim > 0.45:  # threshold
                best_score = sim.item()
                best_match = name

        # Draw result
        cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (255, 0, 0), 2)
        cv2.putText(frame, f"{best_match} ({best_score:.2f})", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Live Recognition", frame)
    if cv2.waitKey(1) == 27:
        break

    # add quit button on the frame to exit the loop
    if cv2.getWindowProperty("Live Recognition", cv2.WND_PROP_VISIBLE) < 1:
        break


cap.release()
cv2.destroyAllWindows()
