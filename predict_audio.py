import tensorflow as tf
import numpy as np
import librosa
import sys
import os

MODEL_PATH = 'saved_model/audio_model.h5'
SAMPLE_RATE = 16000
DURATION = 3.0
N_MELS = 128
TIME_STEPS = 128
MAX_LEN = int(SAMPLE_RATE * DURATION)

def load_audio(file_path, sr=SAMPLE_RATE):
    audio, _ = librosa.load(file_path, sr=sr, duration=DURATION)
    if len(audio) < MAX_LEN:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)))
    else:
        audio = audio[:MAX_LEN]
    return audio

def extract_mel_spectrogram(audio):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.shape[1] < TIME_STEPS:
        pad_width = TIME_STEPS - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :TIME_STEPS]
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return np.expand_dims(mel_spec_db, axis=-1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(0.0)
        sys.exit(1)
    audio_path = sys.argv[1]
    if not os.path.exists(MODEL_PATH):
        print(0.0)
        sys.exit(1)
    model = tf.keras.models.load_model(MODEL_PATH)
    audio = load_audio(audio_path)
    spec = extract_mel_spectrogram(audio)
    spec = np.expand_dims(spec, axis=0)
    pred = model.predict(spec, verbose=0)[0][0]
    print(pred)