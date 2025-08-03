Dataset available on kaggle
Link: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

🧠 Face Mask Detection Using Faster R-CNN
This project implements a deep learning-based object detection model to identify whether individuals in an image are wearing a face mask, not wearing one, or wearing it incorrectly. It was developed as part of the AASD 4014: Deep Learning II final project.

🔍 Problem Statement
Manual enforcement of mask-wearing policies is not scalable in crowded environments. Our goal is to automate this process using computer vision and transfer learning.

📁 Dataset
Source: Kaggle - Face Mask Detection
Annotations: Pascal VOC .xml format

Classes:

With Mask
Without Mask
Mask Worn Incorrectly

⚙️ Model
We use Faster R-CNN with ResNet-50 FPN as the backbone:
Pre-trained on COCO dataset
Fine-tuned on 853 annotated images
Capable of detecting multiple faces and mask status in a single frame

🏋️ Training
Framework: PyTorch + Torchvision

Epochs: 25

Optimizer: SGD (momentum = 0.9)
Data Augmentation: Flips, zoom, brightness shift
Evaluation: mAP, confusion matrix

📈 Results
AP (mAP): 0.728
AP50: 0.901
Accuracy: ~94%
Confusion Matrix: Strong class separation

🎯 Features
Custom MaskDataset with annotation parser
Live detection on uploaded images (via Streamlit app)
Model saved as both .pt (weights) and .pth (full model)

🚀 Demo
A simple Streamlit app is included for real-time testing:

bash
streamlit run app.py

📌 Future Scope
YOLOv5 integration for faster inference
Classification of mask type/color
Real-time webcam deployment with Streamlit or Flask
