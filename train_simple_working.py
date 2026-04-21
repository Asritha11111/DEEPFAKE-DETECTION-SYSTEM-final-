import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path

print("=" * 60)
print("SIMPLE WORKING DEEPFAKE DETECTOR")
print("=" * 60)

# Load images
def load_images(folder, label):
    images = []
    labels = []
    folder_path = Path(folder)
    for img_file in list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")):
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = img / 255.0
            images.append(img)
            labels.append(label)
            print(f"Loaded: {img_file.name}")
    return np.array(images), np.array(labels)

print("\nLoading images...")
real_imgs, real_labels = load_images("dataset/real", 1)
fake_imgs, fake_labels = load_images("dataset/fake", 0)

print(f"\nReal: {len(real_imgs)} images")
print(f"Fake: {len(fake_imgs)} images")

if len(real_imgs) == 0 or len(fake_imgs) == 0:
    print("ERROR: No images found in dataset folders!")
    exit()

# Combine and shuffle
X = np.concatenate([real_imgs, fake_imgs])
y = np.concatenate([real_labels, fake_labels])

# Add channel dimension
X = X.reshape(-1, 64, 64, 1)

# Shuffle
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# Split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\nTraining: {len(X_train)} images")
print(f"Testing: {len(X_test)} images")

# Build VERY simple model
print("\nBuilding simple model...")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

# Train
print("\nTraining...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

# Save model
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/mesonet_model.h5")
print("\n✅ Model saved to saved_model/mesonet_model.h5")

# Evaluate
print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.2%}")

# Test each test image individually
print("\nTesting on ALL test images:")
correct = 0
for i in range(len(X_test)):
    pred = model.predict(X_test[i:i+1], verbose=0)[0][0]
    predicted = "REAL" if pred > 0.5 else "FAKE"
    actual = "REAL" if y_test[i] == 1 else "FAKE"
    status = "✓" if predicted == actual else "✗"
    print(f"  {status} {predicted} ({pred:.1%}) - Actual: {actual}")
    if predicted == actual:
        correct += 1

print(f"\nCorrect: {correct}/{len(X_test)} ({correct/len(X_test):.1%})")

print("\n" + "=" * 60)
print("Done! Run 'node server.js' to test the web app")
print("=" * 60)
