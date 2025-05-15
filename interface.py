import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# === Load trained model ===
model_path = 'best_sports_model.keras'
model = load_model(model_path)

# === Class names ===
class_names = sorted(os.listdir(
    r'C:/Users/User/Desktop/Universty Document/term8/Bulut-bilişim-Yapayzeka/Sports-Image-Classification/train'))

# === Image settings ===
IMAGE_SIZE = (350, 350)

# === Background image path ===
BACKGROUND_PATH = r'C:/Users/User/Desktop/Universty Document/term8/Bulut-bilişim-Yapayzeka/Sports-Image-Classification/background.webp'

# === Main GUI Window (Fullscreen) ===
root = tk.Tk()
root.title("Sports Image Classifier")
root.state("zoomed")  # Fullscreen for Windows

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# === Try to load background image ===
try:
    bg_img = Image.open(BACKGROUND_PATH)
    bg_img = bg_img.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_img)

    bg_label = tk.Label(root, image=bg_photo)
    bg_label.image = bg_photo
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
except Exception as e:
    print(f"⚠️ Background image could not be loaded: {e}")
    bg_label = tk.Label(root, bg="lightgray")
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# === Image Display Panel ===
image_panel = tk.Label(root, bg="white", bd=2, relief="ridge")
image_panel.place(x=screen_width//2 - 175, y=120, width=350, height=350)

# === Result Display Frame ===
result_frame = tk.Frame(root, bg="white", bd=2, relief="ridge")
result_frame.place(x=screen_width//2 - 150, y=500, width=300, height=100)

result_title = tk.Label(result_frame, text="Prediction", font=("Arial", 16, "bold"), bg="white")
result_title.pack(pady=5)

result_label = tk.Label(result_frame, text="", font=("Arial", 14), bg="white", wraplength=280)
result_label.pack()

# === Prediction Function ===
def classify_image(file_path):
    try:
        img = image.load_img(file_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        result_label.config(text=f"Sport: {predicted_class}\nConfidence: {confidence:.2%}")
    except Exception as err:
        result_label.config(text=f"Error in prediction: {err}")

# === Upload and Display Image ===
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    if file_path:
        try:
            img = Image.open(file_path)
            img = img.resize((350, 350), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            image_panel.configure(image=photo)
            image_panel.image = photo
            classify_image(file_path)
        except Exception as e:
            result_label.config(text=f"Image error: {e}")

# === Styled Upload Button ===
style_frame = tk.Frame(root, bg="#ffffff")
style_frame.place(x=screen_width//2 - 100, y=650)

upload_btn = tk.Button(style_frame,
                       text="📂 Upload Image",
                       command=upload_image,
                       font=("Arial", 16, "bold"),
                       bg="#28a5f5",
                       fg="white",
                       activebackground="#1f8cd6",
                       activeforeground="white",
                       bd=0,
                       padx=20,
                       pady=10,
                       relief="raised",
                       cursor="hand2")
upload_btn.pack()

# === Run the GUI ===
root.mainloop()