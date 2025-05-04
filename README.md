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

├── face-app-flask/
│ └── bin
│ │ ├── recognition.py              # Main logic is stored here
│ └── images
│ │ ├── test1.jpeg                  # Image to evaluate the model
│ │ ├── test2.jpeg                  # Image to evaluate the model
│ │ ├── test3.jpeg                  # Image to evaluate the model
│ └── static/
│ │ └── uploads/
│ │ │ ├── temp.txt                  # You can delete temp.txt
│ └── templates/
│ │ ├── index.html                  # Code for Front End
│ └── app.py                        # Run this
│ └── requirements.txt              # Modules required
├── README.md
├── facenet-loading-script.ipynb    # Test the model after training
├── facenet-training-script.ipynb   # Use this to train it yourself

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
