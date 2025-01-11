# Face Recognition with MTCNN, FaceNet, and Keras

This project implements a facial recognition system using Python 3.7 (via Anaconda), leveraging MTCNN for face detection, FaceNet for generating facial embeddings, and a custom-trained Keras model for face classification. The system can recognize predefined classes (e.g., individuals) in real-time via a webcam.

## Features
- **Real-time Face Detection**: Uses MTCNN to detect faces in video frames.
- **Embedding Generation**: Leverages the pre-trained FaceNet model to generate embeddings for detected faces.
- **Face Classification**: A custom Keras model predicts the identity of detected faces.

## Requirements
- Python 3.7 (via Anaconda)
- OpenCV
- TensorFlow
- Keras
- MTCNN
- scikit-learn
- NumPy
- Pillow

## Installation
1. Clone this repository:
    ```bash
    git clone https: https://github.com/WandersonABR/face_recognition.git
    ```

2. Create a new virtual environment with Anaconda:
    ```bash
    conda create -n facenet-env python=3.7
    conda activate facenet-env
    ```

3. Install the required packages:
    ```bash
    pip install tensorflow==2.4 mtcnn opencv-python-headless scikit-learn numpy pillow
    ```

4. Ensure the following model files are in the project directory:
    - `facenet_keras.h5`: Pre-trained FaceNet model.
    - `faces_desc.h5`: Custom Keras model trained to classify faces.

## Usage
1. Run the script:
    ```bash
    python face_recognition.py
    ```

2. The webcam will activate, and the system will start detecting and classifying faces.

3. Press `ESC` to exit the program.

## File Descriptions
- **`face_recognition.py`**: Main script for real-time face detection and recognition.
- **`facenet_keras.h5`**: Pre-trained FaceNet model for embedding generation.
- **`faces_desc.h5`**: Custom-trained Keras model for face classification.

## How It Works
1. **Face Detection**:
    - The `MTCNN` library detects faces in video frames.
    - Detected faces are extracted and resized to 160x160 pixels.

2. **Embedding Generation**:
    - The pre-trained `FaceNet` model generates a 128-dimensional embedding for each face.

3. **Face Classification**:
    - The embedding is passed to a custom Keras model that classifies the face into one of the predefined categories.

4. **Real-time Output**:
    - Detected faces are highlighted with bounding boxes, and the predicted identity is displayed above each face.

## Example Output
- Recognizes faces with a confidence threshold of 98% or higher.
- Displays bounding boxes and labels in real-time via the webcam.

## Dependencies
The following Python libraries are required:
- `tensorflow==2.4`
- `mtcnn`
- `opencv-python-headless`
- `scikit-learn`
- `numpy`
- `pillow`

## Customization
### Add New Classes
1. Collect a dataset of images for each new class.
2. Generate embeddings for these images using the FaceNet model.
3. Train a new classification model (`faces_desc.h5`) using these embeddings.

## Troubleshooting
### Common Issues
- **Error: `No training configuration found in the save file`**:
  This is a warning indicating that the loaded model is not compiled. This is expected as the system only uses the model for inference.
- **Error: `tf.function retracing`**:
  Ensure that the embedding generation function is not repeatedly defined inside a loop.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- **FaceNet**: [David Sandberg's FaceNet implementation](https://github.com/davidsandberg/facenet)
- **MTCNN**: [MTCNN Face Detection](https://github.com/ipazc/mtcnn)

---

Feel free to contribute to this project by submitting issues or pull requests!
