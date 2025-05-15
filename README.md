
# Sports Image Classifier


This project focuses on building a deep learning-based image classification system using a **Convolutional Neural Network (CNN)** to recognize and predict **100 different sport categories**. The model is trained on a labeled dataset of sports images and optimized using techniques such as data augmentation, dropout, and batch normalization to improve generalization. Once trained, the model is deployed in two interactive environments: a Streamlit-based web application and a Tkinter-based desktop GUI. Users can upload an image through either interface and receive the predicted sport label along with the confidence score. The system is evaluated using standard metrics including Accuracy, F1-Score, and Confusion Matrix visualization.


- A **training script** using TensorFlow/Keras and data generators
- A **Streamlit web interface**
- A **Tkinter desktop GUI interface**
- Visualization of model performance through **accuracy, F1-score**, and **confusion matrix**

---

## 📁 Project Structure

```bash
├── best_sports_model.keras             # Saved trained model (Keras format)
├── interface-streamlit.py             # Streamlit web interface
├── interface-tkinter.py               # Tkinter desktop GUI
├── train_model_and_evaluate.py       # Model training & evaluation script
├── background.webp                    # Background image used in both interfaces
├── train/                             # Training image folders (per class)
├── validation/                        # Validation image folders (per class)
├── test/                              # Test image folders (per class)
```

##  Dependencies

Install required packages: using
requirements.txt

Ensure your Python version is 3.8 or above.


## Model Training & Evaluation

Run the following script to train your CNN model:

```bash
python train_model_and_evaluate.py
```

### Features:
- Preprocessing and augmentation using `ImageDataGenerator`
- CNN with 4 convolutional blocks and global average pooling
- Uses callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Saves best model to `best_sports_model.keras`
- Evaluates on test data and displays:
  - **Accuracy**
  - **F1-score**
  - **Classification Report**
  - **Confusion Matrix**
  - **Loss and Accuracy Curves**

> Training images placed inside the `train/`, `validation/`, and `test/` folders, each containing one subfolder per class (e.g., `train/football`, `train/basketball`, etc.).

---

## Web Interface (Streamlit)

To run the web app, use:

```bash
streamlit run interface-streamlit.py
```

### Highlights:
- Modern fullscreen layout with background image
- Upload a sports image and click "Start Prediction"
- Displays:
  - Predicted class
  - Confidence score
  - Visual confidence bar
- Responsive and styled using embedded CSS

>  Make sure `best_sports_model.keras` and `train/` directory are present in the same folder as the script.

---

## Desktop GUI (Tkinter)

To launch the desktop app, run:

```bash
python interface-tkinter.py
```

### Features:
- Fullscreen interface with a background image
- File upload dialog to select image
- Displays uploaded image, predicted class, and confidence
- Minimalistic and fast for offline use

> Requires `best_sports_model.keras` and `background.webp` in the same directory.

---

## Evaluation Example Output

After training, the evaluation output may look like:

```
Test Accuracy: 0.77
Test F1 Score: 0.76


And training history is plotted (loss and accuracy curves).

---

## Model Details

- Input Shape: `224 x 224 x 3`
- Output Layer: `Dense(num_classes, activation='softmax')`
- Loss Function: `categorical_crossentropy`
- Optimizer: `Adam`
- Regularization: `Dropout + BatchNormalization`
- Model is saved in `.keras` format and reused across both interfaces

---

## Tips

- If the `.keras` file is not found: ensure the **absolute path** is correctly set or the file is in the same directory.

---

##  Author

Developed by [MohammedHajjey]   
Cloud Computing & Artificial Intelligence  
2025

---

##s License

This project is for academic purposes. For any commercial use, please contact the author.
