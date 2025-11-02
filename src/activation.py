"""
activation.py
Llama Architecture Implementation
"""

# SwiGLU activation
'''
1. SwiGLU is a gated activation function — short for Swish-Gated Linear Unit.
2. Gating helps model learn which features to keep/ignore.
3. Swish is smoother and empirically better than ReLU or GELU

Formula:
x1 , x2 are two linear projections of same input :
        Swish(x)  = x.sigmoid(x)
        SwiGLU(x) = Swish(x1).x2
'''
class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, factor=4):
        super().__init__()

        inner_dim = hidden_dim * factor

        # GLU requires projection to 2 * inner_dim
        self.w1 = tf.keras.layers.Dense(2 * inner_dim, use_bias=False)
        self.w2 = tf.keras.layers.Dense(hidden_dim, use_bias=False)

    def call(self, x):
        x_proj = self.w1(x)
        x1, x2 = tf.split(x_proj, num_or_size_splits=2, axis=-1)  # (a, b)

        return self.w2(x1 * tf.nn.silu(x2))  # SwiGLU

