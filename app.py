import os
import warnings

# ------------------------------
# Clean TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress info/warnings, only errors
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Suppress oneDNN info
warnings.filterwarnings('ignore')
# ------------------------------

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Load the trained model ---
MODEL_PATH = r"C:\Users\ayush\Downloads\fhb-Disease-Detection-Model-main (1)\Plant-Disease-Detection-Model-main\final_wheat_multi_class_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# --- Auto-detect class names from dataset folder ---
# Use test folder (must have same subfolders as training)
TEST_DIR = r"C:\Users\ayush\Downloads\fhb-Disease-Detection-Model-main (1)\Plant-Disease-Detection-Model-main\archive\data\test"
datagen = ImageDataGenerator(rescale=1./255)
gen = datagen.flow_from_directory(TEST_DIR, target_size=(224,224), batch_size=1)
class_names = list(gen.class_indices.keys())
NUM_CLASSES = len(class_names)

print("[INFO] Using class names:", class_names)

# --- Preprocess uploaded image ---
def preprocess_image(image):
    """Resize, normalize, and expand dims for prediction"""
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0  # ✅ normalize just like training
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Predict function for Gradio ---
def predict(image):
    img_array = preprocess_image(image)
    preds = model.predict(img_array, verbose=0)[0]  # shape=(num_classes,)
    
    # Convert predictions to dictionary {class_name: confidence}
    confidences = {class_names[i]: float(preds[i]) for i in range(NUM_CLASSES)}
    
    # Sort dictionary by confidence descending
    confidences = dict(sorted(confidences.items(), key=lambda item: item[1], reverse=True))
    return confidences

# --- Build Gradio Interface ---
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=NUM_CLASSES),
    title="🌾 Wheat FHB Disease Detection",
    description="Upload a wheat image to detect Fusarium Head Blight (FHB) disease severity.",
    allow_flagging="never"
)

# --- Launch the app ---
if __name__ == "__main__":
    interface.launch(share=False)  # share=True creates a public link automatically