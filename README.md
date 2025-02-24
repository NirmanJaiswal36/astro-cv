# 🚀 Automatic Detection of Craters and Boulders  

## 📌 Project Overview  
This project focuses on developing an AI-powered computer vision model to detect craters, boulders, and track astronomical objects like asteroids and comets in high-resolution orbiter images. The model is built using deep learning and traditional image processing techniques for real-time analysis.  

## 🎯 Objectives  
- Automatically detect craters and boulders from Orbiter High-Resolution Camera (OHRC) images.  
- Identify and track moving astronomical objects, including asteroids and comets.  
- Build an interactive application for image processing and trajectory visualization.  

## 🛠️ Tools and Technologies  
- **Programming:** Python  
- **Object Detection:** YOLO (You Only Look Once), Ultralytics  
- **Computer Vision:** OpenCV  
- **GUI & Web Interface:** Flask, PyQt5  
- **Annotation:** Manual labeling using Python-based tools  

## 🔬 Methodology  
1. **Data Preprocessing**  
   - Dark reduction for systematic error removal  
   - Bilateral filtering for noise reduction  
2. **Moving Target Detection**  
   - Image merging to highlight motion  
   - Line Segment Detector (LSD) for trajectory detection  
   - Morphological operations for false positive reduction  
3. **Trajectory Analysis**  
   - Connecting sequential line segments to track asteroid/comet movement  

## 📸 Application Features  
✅ Image Upload and Processing  
✅ Real-time Crater and Boulder Detection  
✅ Trajectory Visualization of Moving Objects  
✅ Interactive and User-Friendly Interface  

## 📂 Installation  
```bash
git clone https://github.com/yourusername/your-repository-name.git  
cd your-repository-name  
pip install -r requirements.txt  
python main.py  
```
## 📊 Results and Capabilities
Detection of dim objects with brightness as low as 2% of the background.
Identification of both linear and non-linear trajectories.
Application on NEOSSat data with a user-friendly interface.
## 🔗 References
Yekkehkhany et al.: Vision-based framework for asteroid detection using NEOSSat data.
## 👨‍💻 Contributors
Nirman Jaiswal
Arpan Jain
