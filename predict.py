import tensorflow as tf
import numpy as np
import cv2
import sys

def predict_image(model_path, image_path):
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img = cv2.resize(img, (64, 64))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)  # shape (1,64,64,1)
        
        # Predict
        pred = model.predict(img, verbose=0)[0][0]
        
        # Output only the number (no extra text)
        print(pred)
        return pred
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(0.0)  # Return 0 on error
        return 0.0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(0.0)
        sys.exit(1)
    predict_image("saved_model/mesonet_model.h5", sys.argv[1])
