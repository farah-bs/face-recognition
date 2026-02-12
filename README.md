# Face Recognition System

A Python-based **face recognition system** with a GUI for adding new people and performing live recognition using your webcam. Built with **Tkinter**, **OpenCV**, **PyTorch**, and **InsightFace**.

The system uses the **Buffalo_L model** from InsightFace for **face detection and embedding extraction**, and matches faces via **cosine similarity**.

## Features

- Add new people by capturing multiple face embeddings.
- Live face recognition from webcam with real-time bounding boxes.
- Cosine similarity-based matching for accurate recognition.
- Uses **InsightFace Buffalo_L** model for detection and embeddings.
- Simple and user-friendly GUI for interaction.
- Visual feedback for face capture and recognition progress.

## Notes

* Ensure your webcam is accessible and not used by other applications.
* For best results, move your face slowly during the capture process.
* The recognition threshold is set to `0.45` by default; you can adjust it in `live-recognition.py`.
* You can quit the live recognition by pressing **ESC** or closing the window.
