import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm

# Configuration
SAMPLE_RATE = 16000
DURATION = 3.0  # seconds
N_MELS = 128
TIME_STEPS = 128
MAX_LEN = int(SAMPLE_RATE * DURATION)

def load_audio(file_path, sr=SAMPLE_RATE):
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=DURATION)
        if len(audio) < MAX_LEN:
            audio = np.pad(audio, (0, MAX_LEN - len(audio)))
        else:
            audio = audio[:MAX_LEN]
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_mel_spectrogram(audio, n_mels=N_MELS, time_steps=TIME_STEPS):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Resize to fixed time steps
    if mel_spec_db.shape[1] < time_steps:
        pad_width = time_steps - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :time_steps]
    # Normalize to [0,1]
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_spec_db

# Build model
def build_audio_model(input_shape=(N_MELS, TIME_STEPS, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Example dataset loading (adjust paths to your ASVspoof folders)
def load_dataset(real_dir, fake_dir):
    X, y = [], []
    # Real files
    for file in tqdm(glob(os.path.join(real_dir, '*.flac')) + glob(os.path.join(real_dir, '*.wav'))):
        audio = load_audio(file)
        if audio is not None:
            spec = extract_mel_spectrogram(audio)
            X.append(np.expand_dims(spec, axis=-1))
            y.append(1)
    # Fake files
    for file in tqdm(glob(os.path.join(fake_dir, '*.flac')) + glob(os.path.join(fake_dir, '*.wav'))):
        audio = load_audio(file)
        if audio is not None:
            spec = extract_mel_spectrogram(audio)
            X.append(np.expand_dims(spec, axis=-1))
            y.append(0)
    return np.array(X), np.array(y)

if __name__ == '__main__':
    # Replace with your dataset paths
    REAL_DIR = 'dataset/audio/real'
    FAKE_DIR = 'dataset/audio/fake'
    X, y = load_dataset(REAL_DIR, FAKE_DIR)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_audio_model()
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    model.save('saved_model/audio_model.h5')
    print("Audio model saved to saved_model/audio_model.h5")