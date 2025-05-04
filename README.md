# Face Recognition GUI using FaceNet (ResNet Backend) and Flask

This project is a complete facial recognition pipeline that includes training and testing a FaceNet model (based on ResNet) and deploying it as a sleek web application using Flask.

## 📌 Features

- 🔍 Face recognition using pretrained FaceNet (ResNet)
- 🖼️ Upload image to identify known users
- ➕ Add new users by saving their embeddings
- 🧠 Trained with custom or public datasets
- 🌐 Clean UI using Tailwind CSS and Font Awesome

---

## 🗂️ Project Structure

├── face-app-flask/<br>
│ └── bin<br>
│ │ ├── recognition.py              # Main logic is stored here<br>
│ └── images<br>
│ │ ├── test1.jpeg                  # Image to evaluate the model<br>
│ │ ├── test2.jpeg                  # Image to evaluate the model<br>
│ │ ├── test3.jpeg                  # Image to evaluate the model<br>
│ └── static/<br>
│ │ └── uploads/<br>
│ │ │ ├── temp.txt                  # You can delete temp.txt<br>
│ └── templates/<br>
│ │ ├── index.html                  # Code for Front End<br>
│ └── app.py                        # Run this<br>
│ └── requirements.txt              # Modules required<br>
├── README.md<br>
├── facenet-loading-script.ipynb    # Test the model after training<br>
├── facenet-training-script.ipynb   # Use this to train it yourself<br>

---

## 🚀 Getting Started

### 1. Clone or Download this Repo
### 2. Traing a Model for Few Epochs Using the "facenet-training-script.ipynb" File
### 3. Use Any Dataset for Your Training (Recommended: lfw-dataset)
### 4. Once the Model is Trained, Save the Model by the Name of "model.pth"
### 5. Copy and Paste the Same Model into "facenet-app-flask" Folder
### 6. Rename the "model.pth" to "siamese-triplet.pth" inside the "facenet-app-flask" Folder
### 7. Run the "app.py" File

---
