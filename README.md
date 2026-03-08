# Crowd Guard – AI-Powered Stampede Detection System

Crowd Guard is a computer vision-based safety system designed to detect potential stampede situations in crowded environments. The system uses deep learning and image processing techniques to classify crowd scenes as **normal** or **high-risk**, helping authorities monitor and prevent dangerous situations.

This project demonstrates the application of **Convolutional Neural Networks (CNNs)** for crowd analysis and integrates a **Streamlit interface** for easy interaction and visualization.

---

## Project Overview

Large gatherings such as festivals, concerts, and public events can sometimes lead to dangerous crowd situations. Detecting early signs of overcrowding or panic can help prevent stampedes.

Crowd Guard analyzes crowd images using a trained deep learning model and predicts whether the crowd condition is **safe or risky**.

---

## Features

- Crowd image classification using CNN
- Detection of **normal vs stampede-risk crowd scenes**
- Image preprocessing and data augmentation
- Interactive **Streamlit web interface**
- Real-time prediction on uploaded images
- Automated crowd safety alerts

---

## Tech Stack

**Programming Language**
- Python

**Machine Learning / Deep Learning**
- TensorFlow
- Keras
- Scikit-learn

**Computer Vision**
- OpenCV

**Data Processing**
- NumPy
- Pandas

**Visualization**
- Matplotlib

**Web Interface**
- Streamlit

---

## Project Architecture

1. Data Collection  
   Crowd images representing safe and risky situations.

2. Data Preprocessing  
   - Image resizing  
   - Normalization  
   - Data augmentation  

3. Model Development  
   - CNN model trained to classify crowd images.

4. Prediction System  
   - Model predicts crowd safety level.

5. User Interface  
   - Streamlit app allows users to upload crowd images and view predictions.


## Example Workflow

1. Upload a crowd image.
2. The model processes the image.
3. The system predicts:
   - **Normal Crowd**
   - **Stampede Risk**
4. The result is displayed in the Streamlit interface.

---

## Results

The CNN model achieved approximately **90% classification accuracy** on the validation dataset.

---

## Future Improvements

- Real-time video surveillance integration
- Cloud deployment
- Multi-class crowd behavior detection
- Integration with public safety alert systems
- Larger training datasets

---

## Author

**Chandana Irakam**

- GitHub: https://github.com/Chandana-Irakam
- LinkedIn: https://linkedin.com/in/chandana-irakam
