import json
import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import  models,layers

from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
#from tensorflow.keras.models import Model, load_model,models
#from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Concatenate, Dropout,GlobalMaxPooling1D

'''
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Concatenate, Dropout,GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
'''
from tqdm import tqdm
MAX_LEN = 512
EMBED_DIM = 70

CHAR_VOCAB = list(
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " ,.;:!?\"'/\\|_@#$%^&*~`+=<>[](){}-"
)

char2idx = {c: i + 1 for i, c in enumerate(CHAR_VOCAB)}
VOCAB_SIZE = len(char2idx) + 1  # +1 for padding
def encode_text(text, max_len=512):
    text = text.lower()
    encoded = [char2idx.get(c, 0) for c in text[:max_len]]
    return encoded + [0] * (max_len - len(encoded))


def load_hc3_dataset(json_path):
    X, y = [], []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in tqdm(data, desc="Loading HC3"):
        for h in item["human_answers"]:
            X.append(encode_text(h))
            y.append(0)  # Human

        for a in item["chatgpt_answers"]:
            X.append(encode_text(a))
            y.append(1)  # AI

    return np.array(X), np.array(y)
class HighwayLayer(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.H = layers.Dense(units, activation="relu")
        self.T = layers.Dense(units, activation="sigmoid")

    def call(self, x):
        h = self.H(x)
        t = self.T(x)
        return h * t + x * (1 - t)
'''
def build_rdnn():
    inputs = layers.Input(shape=(MAX_LEN,), name="input_chars")

    # Character Embedding
    x = layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBED_DIM,
        embeddings_initializer="glorot_uniform",
        name="char_embedding"
    )(inputs)

    # CNN Character-Level Layer
    conv_outputs = []
    for k, f in zip([3, 4, 5, 7], [64, 128, 256, 512]):
        c = layers.Conv1D(
            filters=f,
            kernel_size=k,
            activation="relu"
        )(x)
        c = layers.GlobalMaxPooling1D()(c)
        conv_outputs.append(c)

    cnn_features = layers.Concatenate(name="cnn_concat")(conv_outputs)

    # Distributed Highway Network (2 layers)
    hwy = HighwayLayer(cnn_features.shape[-1])(cnn_features)
    hwy = HighwayLayer(cnn_features.shape[-1])(hwy)

    # Expand dims for BLSTM
    #hwy_seq = tf.expand_dims(hwy, axis=1)
    total_features = cnn_features.shape[-1] 
    hwy_seq = layers.Reshape((1, total_features))(hwy)
    #hwy_seq = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(hwy)
    # BiLSTM Encoder
    encoder = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True),
        name="blstm_encoder"
    )(hwy_seq)

    # BiLSTM Decoder
    decoder = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False),
        name="blstm_decoder"
    )(encoder)

    decoder = layers.Dropout(0.5)(decoder)

    # Classification Layer
    outputs = layers.Dense(2, activation="softmax", name="softmax")(decoder)

    model = models.Model(inputs, outputs, name="RDNN")
    return model
'''
def build_rdnn():
    inputs = layers.Input(shape=(MAX_LEN,), name="input_chars")

    # Character Embedding
    x = layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBED_DIM,
        embeddings_initializer="glorot_uniform",
        name="char_embedding"
    )(inputs)

    # CNN Character-Level Layer
    conv_outputs = []
    for k, f in zip([3, 4, 5, 7], [64, 128, 256, 512]):
        c = layers.Conv1D(
            filters=f,
            kernel_size=k,
            activation="relu"
        )(x)
        c = layers.GlobalMaxPooling1D()(c)
        conv_outputs.append(c)

    cnn_features = layers.Concatenate(name="cnn_concat")(conv_outputs)

    # Distributed Highway Network (2 layers)
    hwy = HighwayLayer(cnn_features.shape[-1])(cnn_features)
    hwy = HighwayLayer(cnn_features.shape[-1])(hwy)

    # FIX: Use Reshape instead of tf.expand_dims
    hwy_seq = layers.Reshape((1, cnn_features.shape[-1]))(hwy)

    # BiLSTM Encoder
    encoder = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True),
        name="blstm_encoder"
    )(hwy_seq)

    # BiLSTM Decoder
    decoder = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False),
        name="blstm_decoder"
    )(encoder)

    decoder = layers.Dropout(0.5)(decoder)

    # Classification Layer
    outputs = layers.Dense(2, activation="softmax", name="softmax")(decoder)

    model = models.Model(inputs, outputs, name="RDNN")
    return model
# Load HC3 dataset
#X, y = load_hc3_dataset("all-Test.json")
#X, y = load_hc3_dataset("medicine.json")
X, y = load_hc3_dataset("Dataset/all-Test.json")

y = to_categorical(y, num_classes=2)

# Train / Validation / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

# Build model
model = build_rdnn()

model.compile(
    optimizer=AdamW(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    #epochs=10,
    epochs=5,

    #epochs=60,

    validation_split=0.1
)
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {acc:.4f}")
# 1. Get predictions (probabilities)
y_pred_probs = model.predict(X_test)

# 2. Convert probabilities to class labels (0 or 1)
# np.argmax handles the [prob_class_0, prob_class_1] output
y_pred = np.argmax(y_pred_probs, axis=1)

# 3. Convert true labels (one-hot) back to class labels (0 or 1)
y_true = np.argmax(y_test, axis=1)

# 4. Calculate Standard Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# 5. Calculate FPR (False Positive Rate)
# FPR = FP / (FP + TN)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
fpr = fp / (fp + tn)

# Print results
print(f"--- Evaluation Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"FPR:       {fpr:.4f}")
