"""
src/models/lstm.py
──────────────────
Optional LSTM model definition.
Per specifications:
  TensorFlow/Keras is optional (LSTM only, install only if user approves in Phase 6).
  If TensorFlow is not installed, provide a clear, informative message explaining
  how to enable it once approved.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    HAS_TF = True
except ImportError:
    HAS_TF = False


def is_lstm_available() -> bool:
    """Check if TensorFlow/Keras is installed and available for LSTM modeling."""
    return HAS_TF


def build_lstm_model(
    input_shape: tuple[int, int],
    units: int = 64,
    learning_rate: float = 0.001,
) -> Any:
    """Build an LSTM sequential neural network for time series forecasting.

    Parameters
    ----------
    input_shape: (timesteps, num_features)
    units: Hidden layer unit count
    learning_rate: Optimizer learning rate

    Raises
    ------
    ImportError: If TensorFlow is not installed.
    """
    if not HAS_TF:
        raise ImportError(
            "TensorFlow is not installed. Per project instructions, TensorFlow/Keras "
            "is optional for LSTM only and requires explicit user approval. "
            "To enable LSTM, install tensorflow via 'pip install tensorflow'."
        )

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
