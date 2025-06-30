Face Recognition using PCA, LDA, and MLP Classifier

This project implements a face recognition pipeline that:
Extracts features from grayscale face images using Principal Component Analysis (PCA) (Eigenfaces)
Applies Linear Discriminant Analysis (LDA) for dimensionality reduction
Classifies faces using a Multi-Layer Perceptron (MLP) classifier

📂 Project Structure

├── dataset/
│   └── faces/
│       ├── person1/
│       │   ├── image1.jpg
│       │   └── ...
│       ├── person2/
│       └── ...
├── main.py
├── README.md
└── results.pdf

⚙️ Requirements
Python 3.x
NumPy
OpenCV (cv2)
scikit-learn
Matplotlib

Install dependencies:
pip install numpy opencv-python scikit-learn matplotlib

🚀 How to Run
Prepare the dataset

Place your images in dataset/faces/, organized per class:
dataset/faces/
    person1/
        img1.jpg
        img2.jpg
        ...
    person2/
        img1.jpg
        ...
Run the script
python main.py
Outputs

Console logs of shapes, progress, and accuracy
Plots of eigenfaces and test predictions
Final test accuracy

📊 Results
Example output summary:
Number of Samples: 450
Features per Sample: 90000
PCA Components: 150
MLP Classifier Hidden Layers: (10,10)
Accuracy: ~72.57%

🧠 How It Works
Preprocessing:
Convert images to grayscale
Resize to 300x300
Flatten into vectors
PCA (Eigenfaces):
Reduce dimensionality while capturing variance
Visualize top eigenfaces
LDA:
Further reduce dimensionality and maximize class separability
MLP Classifier:
Trains on LDA features
Predicts face classes with probabilities
Visualization:
Gallery of eigenfaces
Gallery of predictions with probabilities and true labels

📝 Notes
The model stops training if loss doesn’t improve for 10 consecutive epochs.
Adjust n_components in PCA or hidden layer sizes in MLP for experimentation.
The script prints training progress (verbose=True).

📄 License
This project is for educational purposes. Feel free to adapt and reuse.
