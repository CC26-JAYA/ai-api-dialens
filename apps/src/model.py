import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DenseBlock(keras.layers.Layer):
    """
    Custom Layer: Dense + BatchNormalization + Dropout.
    Dengan optional Residual Connection untuk membantu gradient flow.
    """

    def __init__(self, units, dropout_rate=0.3, use_residual=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual

        self.dense = layers.Dense(units, activation="relu")
        self.bn = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout_rate)

        self.proj = layers.Dense(units, activation="relu") if use_residual else None

    def call(self, inputs, training=None, mask=None):  # type: ignore
        out = self.dense(inputs)
        out = self.bn(out, training=training or False)
        out = self.dropout(out, training=training or False)

        if self.use_residual and self.proj is not None:
            x = self.proj(inputs)
            out = out + x
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "dropout_rate": self.dropout_rate,
                "use_residual": self.use_residual,
            }
        )
        return config


class HealthClassifier(keras.Model):
    """
    Model Subclassing untuk binary health classification.
    Arsitektur: 256 -> 128 (residual) -> 64 (residual) -> 32 -> 1
    """

    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.block1 = DenseBlock(
            256, dropout_rate=0.4, use_residual=False, name="block1"
        )
        self.block2 = DenseBlock(
            128, dropout_rate=0.3, use_residual=True, name="block2"
        )
        self.block3 = DenseBlock(64, dropout_rate=0.2, use_residual=True, name="block3")
        self.block4 = DenseBlock(
            32, dropout_rate=0.1, use_residual=False, name="block4"
        )
        self.out = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):  # type: ignore
        x = self.block1(inputs, training=training or False)
        x = self.block2(x, training=training or False)
        x = self.block3(x, training=training or False)
        x = self.block4(x, training=training or False)
        return self.out(x)

    def get_config(self):
        return super().get_config()


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss — lebih baik dari WeightedBCE untuk imbalanced data.
    """

    def __init__(self, alpha=0.4, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)

        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1.0 - pt, self.gamma)
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        focal_loss = alpha_weight * focal_weight * bce
        return tf.reduce_mean(focal_loss)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config


class BinaryMAE(keras.metrics.Metric):
    """
    Custom Metric: MAE dari binary prediction (threshold 0.5).
    """

    def __init__(self, name="binary_mae", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total = self.add_weight(name="total", initializer="zeros")  # type: ignore
        self.count = self.add_weight(name="count", initializer="zeros")  # type: ignore

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.float32)
        y_pred_binary = tf.cast(y_pred >= self.threshold, tf.float32)  # type: ignore
        y_pred_binary = tf.reshape(y_pred_binary, (-1,))

        mae = tf.abs(y_true - y_pred_binary)
        self.total.assign_add(tf.reduce_sum(mae))  # type: ignore
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))  # type: ignore

    def result(self):
        return self.total / self.count  # type: ignore

    def reset_state(self):
        self.total.assign(0.0)  # type: ignore
        self.count.assign(0.0)  # type: ignore
