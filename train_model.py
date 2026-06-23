import os
import numpy as np
from sklearn.utils import shuffle

# --- CRITICAL ENVIRONMENT FIXES ---
# Disable oneDNN to prevent the CancelledError/Graph Execution crash
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Reduce logging noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

# Force TensorFlow to handle the math more carefully on Windows
tf.config.run_functions_eagerly(True)

DATASET = "dataset_npy"
FRAME_COUNT = 40
FEATURES = 126  # 2 hands * 21 points * 3 coordinates

# 1. Load and Prepare Data
sequences = []
labels = []

if not os.path.exists(DATASET):
    print(f"Error: Folder '{DATASET}' not found!")
    exit()

actions = sorted(os.listdir(DATASET))
label_map = {label: num for num, label in enumerate(actions)}

print(f"Loading data for actions: {actions}")

for action in actions:
    path = os.path.join(DATASET, action)
    if not os.path.isdir(path):
        continue

    for file in os.listdir(path):
        try:
            data = np.load(os.path.join(path, file))
            # Ensure the data matches the expected shape (40 frames, 126 features)
            if data.shape == (FRAME_COUNT, FEATURES):
                sequences.append(data)
                labels.append(label_map[action])
        except Exception as e:
            print(f"Skipping file {file}: {e}")

X = np.array(sequences)
y = np.array(labels)

# Shuffle the dataset
X, y = shuffle(X, y)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# 2. Build the Model
tf.keras.backend.clear_session() # Clear any old models from memory

model = Sequential()
# Input layer replaces the old input_shape parameter for better compatibility
model.add(Input(shape=(FRAME_COUNT, FEATURES)))

model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.3))

model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.3))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(len(actions), activation="softmax"))

# 3. Compile and Train
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Starting training... This may take a moment to initialize.")

# Using a smaller batch size (4) to prevent memory-related 'CancelledError'
model.fit(X, y, epochs=120, batch_size=4)

# 4. Save Results
model.save("model.h5")
np.save("labels.npy", actions)

print("Training complete. 'model.h5' and 'labels.npy' have been saved.")