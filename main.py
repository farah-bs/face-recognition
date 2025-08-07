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

def threaded_capture(name_entry):
    name = name_entry.get().strip()
    if not name:
        messagebox.showerror("Error", "Please enter a name before capturing.")
        return

    threading.Thread(target=capture_embeddings_gui, args=(name,), daemon=True).start()


# ===============================
# GUI SETUP
# ===============================
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("420x280")
root.resizable(False, False)

title_label = tk.Label(root, text="Face Recognition System", font=("Helvetica", 18, "bold"))
title_label.pack(pady=15)

# Buttons
btn_capture = tk.Button(root, text="📷 Add New Person", font=("Helvetica", 14), width=25,
                        command=lambda: threaded_capture(name_entry))
btn_capture.pack(pady=10)

btn_recognize = tk.Button(root, text="🧠 Start Live Recognition", font=("Helvetica", 14), width=25,
                          command=start_recognition)
btn_recognize.pack(pady=10)

root.mainloop()
