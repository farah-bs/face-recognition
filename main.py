import tkinter as tk
from tkinter import messagebox
import threading
import subprocess
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
def start_recognition():
    try:
        subprocess.run([sys.executable, "live-recognition.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start recognition:\n{e}")

def start_embedding():
    try:
        subprocess.run([sys.executable, "capture_embeddings.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start recognition:\n{e}")



# ===============================
# GUI SETUP
# ===============================
root = tk.Tk()
root.title("Live Face Recognition System")
root.geometry("420x200")
root.resizable(False, False)

# Center the window on the screen
window_width = 420
window_height = 200

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate position x and y coordinates
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)

root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

# Title
title_label = tk.Label(root, text="Face Recognition System", font=("Helvetica", 18, "bold"))
title_label.pack(pady=15)

# Buttons
btn_capture = tk.Button(root, text="📷 Add New Person", font=("Helvetica", 14), width=25,
                        command=start_embedding)
btn_capture.pack(pady=10)

btn_recognize = tk.Button(root, text="🧠 Start Live Recognition", font=("Helvetica", 14), width=25,
                          command=start_recognition)
btn_recognize.pack(pady=10)

# Start GUI
root.mainloop()