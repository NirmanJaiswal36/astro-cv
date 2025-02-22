import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QWidget
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from scipy.ndimage import median_filter

# --- Backend Functions ---
def ensure_output_directory():
    """Ensure the 'output' directory exists."""
    output_dir = r"D:\CS\Computer Vision Project\output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def save_image(output_dir, filename, image):
    """Save an image to the output directory."""
    file_path = os.path.join(output_dir, filename)
    cv2.imwrite(file_path, image)
    print(f"Saved image: {file_path}")  # Debugging
    return file_path

def load_images(track_dir):
    """Loads all images from a directory."""
    import os
    image_files = sorted([os.path.join(track_dir, f) for f in os.listdir(track_dir) if f.endswith('.png') or f.endswith('.jpg')])
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in image_files]
    if not images:
        print(f"Warning: No images found in directory {track_dir}")
    return images

def preprocess_image(image):
    """Applies noise reduction and background subtraction."""
    # Apply bilateral filter for noise reduction
    noise_reduced = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    # Apply median filter for background smoothing
    background = median_filter(noise_reduced, size=21)
    # Subtract background
    preprocessed = cv2.subtract(noise_reduced, background)
    return preprocessed

def merge_images(images):
    """Creates a merged image using maximum pixel intensity."""
    return np.max(np.array(images), axis=0)

def detect_lines(merged_image):
    """Detects lines in the merged image using the LSD algorithm."""
    # Convert merged image to edges using Canny edge detector
    edges = cv2.Canny(merged_image, threshold1=100, threshold2=200)
    # Detect lines using LSD on the edge image
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(edges)[0]  # `detect` returns a tuple, take the first element (lines)
    return lines

def remove_false_positives(lines, merged_image):
    """Filters out false positives based on line intensity and length."""
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        intensity = np.mean([merged_image[int(y1), int(x1)], merged_image[int(y2), int(x2)]])
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if intensity > 50 and length > 10:  # Adjust thresholds as needed
            filtered_lines.append(line)
    return filtered_lines

def draw_lines(image, lines, color=(0, 255, 0)):
    """Draws detected lines on an image."""
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return image

# --- GUI Application ---
class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Asteroid/Comet Detection")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout
        self.main_layout = QHBoxLayout()
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.main_layout)

        # Left panel for buttons
        self.left_panel = QVBoxLayout()
        self.load_button = QPushButton("Load Images")
        self.process_button = QPushButton("Process Images")
        self.left_panel.addWidget(self.load_button)
        self.left_panel.addWidget(self.process_button)
        self.left_panel.addStretch()  # Push buttons to the top
        
        # Right panel for images
        self.right_panel = QVBoxLayout()
        self.image_label = QLabel("Merged Image will be displayed here", alignment=Qt.AlignCenter)
        self.result_label = QLabel("Result Image will be displayed here", alignment=Qt.AlignCenter)
        self.right_panel.addWidget(self.image_label)
        self.right_panel.addWidget(self.result_label)
        
        # Add panels to the main layout
        self.main_layout.addLayout(self.left_panel, 1)  # Left panel takes 1 part
        self.main_layout.addLayout(self.right_panel, 3)  # Right panel takes 3 parts
        
        # Signals
        self.load_button.clicked.connect(self.load_images)
        self.process_button.clicked.connect(self.process_images)
        
        # Attributes
        self.image_directory = None
        self.merged_image = None
        self.result_image = None
    
    def load_images(self):
        """Load images from a directory."""
        self.image_directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if self.image_directory:
            self.image_label.setText(f"Loaded images from: {self.image_directory}")
    
    def process_images(self):
        """Process the loaded images."""
        if not self.image_directory:
            self.image_label.setText("Please load an image directory first!")
            return

        # Load and preprocess images
        images = load_images(self.image_directory)
        if not images:
            self.image_label.setText("No images found!")
            return

        preprocessed_images = [preprocess_image(img) for img in images]

        # Merge images to highlight trajectories
        self.merged_image = merge_images(preprocessed_images)

        # Detect moving objects (lines)
        lines = detect_lines(self.merged_image)
        
        # Check if no lines are detected
        if len(lines) == 0:
            self.image_label.setText("No lines detected!")
            return

        # Remove false positives
        filtered_lines = remove_false_positives(lines, self.merged_image)

        # Draw detected lines on the merged image
        self.result_image = cv2.cvtColor(self.merged_image, cv2.COLOR_GRAY2BGR)
        self.result_image = draw_lines(self.result_image, filtered_lines)

        # Create 'output' directory
        output_dir = ensure_output_directory()

        # Save images in the 'output' directory
        merged_image_path = save_image(output_dir, "merged_image.png", self.merged_image)
        result_image_path = save_image(output_dir, "result_image.png", self.result_image)

        # Display images in the GUI
        self.display_image(merged_image_path, self.image_label)
        self.display_image(result_image_path, self.result_label)

        
    def display_image(self, image_path, label):
        """Display an image on the given label."""
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))
        label.setAlignment(Qt.AlignCenter)

# --- Main Function ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec_())
