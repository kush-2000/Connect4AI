#!/usr/bin/env python
# coding: utf-8

# In[27]:


print("Starting connect4_server.py...")

import anvil.server
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import io
import os
from tensorflow.keras.activations import softmax

# === 🔹 Custom PatchEmbedding Layer ===
@tf.keras.utils.register_keras_serializable(package="Custom")
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patches, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.proj = tf.keras.layers.Dense(hidden_dim, kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.pos_embedding = None

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            shape=(1, self.num_patches, self.hidden_dim),
            initializer="random_normal",
            trainable=True,
            name="pos_embedding"
        )

    def call(self, x):
        x = self.proj(x)
        return x + self.pos_embedding  

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "hidden_dim": self.hidden_dim})
        return config

# === 🔹 Custom ClassTokenIndex Layer ===
@tf.keras.utils.register_keras_serializable(package="Custom")
class ClassTokenIndex(tf.keras.layers.Layer):
    def call(self, x):
        bs = tf.shape(x)[0]
        indices = tf.range(1)
        indices = tf.expand_dims(indices, 0)
        return tf.tile(indices, [bs, 1])

# === 🔹 Initialize Global Models ===
cnn_model = None
transformer_model = None

@anvil.server.callable
def initialize_models():
    """
    Load CNN and Transformer models into memory.
    """
    global cnn_model, transformer_model

    model_path_cnn = "connect4_cnn.keras"
    model_path_transformer = "connect4_transformer.keras"  

    try:
        # ✅ Load CNN Model with Custom Objects
        if cnn_model is None:
            if os.path.exists(model_path_cnn):
                cnn_model = tf.keras.models.load_model(
                    model_path_cnn, custom_objects={"softmax_v2": softmax}
                )
                print("✅ CNN Model Loaded Successfully!")
            else:
                print(f"⚠️ CNN model file '{model_path_cnn}' not found.")

        # ✅ Load Transformer Model with ALL Custom Layers
        if transformer_model is None:
            if os.path.exists(model_path_transformer):
                transformer_model = tf.keras.models.load_model(
                    model_path_transformer, 
                    custom_objects={
                        "PatchEmbedding": PatchEmbedding,
                        "ClassTokenIndex": ClassTokenIndex 
                    }
                )
                print("✅ Transformer Model Loaded Successfully!")
            else:
                print(f"⚠️ Transformer model file '{model_path_transformer}' not found.")

    except Exception as e:
        print(f"❌ Error loading models: {e}")

# === 🔹 Ensure Models are Loaded at Server Start ===
initialize_models()

# === 🔹 Connect to Anvil Uplink ===
print("Connecting to Anvil Uplink...")
try:
    anvil.server.connect("server_HPZD34YUMH7BDVS2VSWYRWO3-GB4NLLJQNY6VL2WO")  # Replace with your Uplink key
    print("✅ Anvil Uplink connected successfully!")
except Exception as e:
    print(f"❌ Error connecting to Anvil Uplink: {e}")
    raise RuntimeError("Failed to connect to Anvil Uplink.")

# === 🔹 AI MOVE PREDICTION FUNCTION ===
@anvil.server.callable
def get_best_move_cnn(board_state):
    """
    Predict the best move using the CNN model.
    """
    global cnn_model
    print(f"🛠️ Debug: Received board_state for CNN = {board_state}")

    if not isinstance(board_state, list):
        raise TypeError(f"❌ board_state must be a list, but got {type(board_state)}")

    # Convert to numpy array
    board_array = np.array(board_state)
    print(f"✅ Converted board_array shape for CNN: {board_array.shape}")

    # ✅ Preprocess Input
    board_tensor = np.stack([(board_array == 1).astype(np.float32), (board_array == -1).astype(np.float32)], axis=-1)
    board_tensor = np.expand_dims(board_tensor, axis=0)  # Ensure batch shape (1, 6, 7, 2)

    # ✅ Make Prediction
    predictions = cnn_model.predict(board_tensor)
    best_move = int(np.argmax(predictions[0]))

    print(f"✅ CNN Predicted Move: {best_move}")
    return best_move


@anvil.server.callable
def get_best_move_transformer(board_state):
    """
    Predict the best move using the Transformer model.
    """
    global transformer_model
    print(f"🛠️ Debug: Received board_state for Transformer = {board_state}")

    if not isinstance(board_state, list):
        raise TypeError(f"❌ board_state must be a list, but got {type(board_state)}")

    # Convert to numpy array
    board_array = np.array(board_state)
    print(f"✅ Converted board_array shape for Transformer: {board_array.shape}")

    # ✅ Preprocess Input
    board_tensor = np.stack([(board_array == 1).astype(np.float32), (board_array == -1).astype(np.float32)], axis=-1)
    board_tensor = board_tensor.reshape(1, 6 * 7, 2)  # Ensure batch shape for Transformer

    # ✅ Make Prediction
    predictions = transformer_model.predict(board_tensor)
    best_move = int(np.argmax(predictions[0]))

    print(f"✅ Transformer Predicted Move: {best_move}")
    return best_move


# === 🔹 DRAW CONNECT 4 BOARD FUNCTION ===
@anvil.server.callable
def draw_board(board):
    """
    Draw the Connect 4 board with circles representing player and AI moves.
    """
    board_array = np.array(board, dtype=int)

    print("🔹 Board state before drawing:")
    print(board_array)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.set_facecolor("white")
    ax.add_patch(plt.Rectangle((0, 0), 7, 6, color="#8B4513", zorder=1))

    for r in range(board_array.shape[0]):
        for c in range(board_array.shape[1]):
            cutout = Circle((c + 0.5, r + 0.5), 0.4, facecolor="white", edgecolor="#444", zorder=1)
            ax.add_patch(cutout)

            if board_array[r, c] == 1:  # Player move (Red)
                outer_ring = Circle((c + 0.5, r + 0.5), 0.38, facecolor="#A23C2D", edgecolor="black", lw=1.5, zorder=2)
                inner_circle = Circle((c + 0.5, r + 0.5), 0.32, facecolor="#FF4500", edgecolor="none", zorder=3)
                ax.add_patch(outer_ring)
                ax.add_patch(inner_circle)

            elif board_array[r, c] == -1:  # AI move (Yellow)
                outer_ring = Circle((c + 0.5, r + 0.5), 0.38, facecolor="#C8A500", edgecolor="black", lw=1.5, zorder=2)
                inner_circle = Circle((c + 0.5, r + 0.5), 0.32, facecolor="#FFD700", edgecolor="none", zorder=3)
                ax.add_patch(outer_ring)
                ax.add_patch(inner_circle)

    ax.set_xlim(0, 7)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    plt.tight_layout(pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return anvil.BlobMedia("image/png", buf.read(), name="board.png")

# Keep the server running
print("✅ Server is ready and running...")
anvil.server.wait_forever()


# In[ ]:




