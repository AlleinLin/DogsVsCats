import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms

# predict.py to GUI 2024.6.26

label_map = {
    0: "cat",
    1: "dog",
}

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for better accuracy
])

device = "cpu"

model = torch.load("./model/18 AlexNet_CatvsDog_improved.pth")
model.eval()
model.to(device)


def predict_image(image_path):
    """Predicts the class of the image and displays the result."""
    try:
        img = Image.open(image_path)
        img_data = data_transform(img)
        img_data = torch.unsqueeze(img_data, dim=0)

        with torch.no_grad():  # Disable gradient calculation for efficiency
            predict = model(img_data.to(device))

        predict = torch.argmax(predict).cpu().numpy()
        predicted_class = label_map[int(predict)]

        update_result_label(predicted_class)
        display_image(img)
    except Exception as e:
        print(f"Error: {e}")
        update_result_label("Error: Could not process image.")


def update_result_label(text):
    """Updates the label displaying the prediction result."""
    result_label.config(text=text)


def display_image(img):
    global image_label
    photo = ImageTk.PhotoImage(img.resize((300, 300)))  # Resize for better presentation
    image_label.config(image=photo)
    image_label.image = photo  # Keep reference to avoid garbage collection


def browse_images():
    image_path = filedialog.askopenfilename()
    if image_path:
        predict_image(image_path)


root = tk.Tk()
root.title("Dog or Cat Prediction App")

button_frame = tk.Frame(root)
button_frame.pack(padx=10, pady=10)

browse_button = tk.Button(button_frame, text="Browse Image", command=browse_images)
browse_button.pack(side=tk.LEFT, padx=5, pady=5)

result_label = tk.Label(root, text="No image loaded yet", font=("Arial", 16))
result_label.pack(padx=10, pady=10)

image_label = tk.Label(root)
image_label.pack(padx=10, pady=10)

root.mainloop()
