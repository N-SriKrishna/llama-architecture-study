"""
normalization.py
Llama Architecture Implementation
"""

# RMSNorm = Root Mean Square Normalization
'''
1) normalizes only by the vector magnitude (RMS) 
2) does not subtract the mean
3) uses a learnable scale parameter γ (but no bias)

        ┌───────────────┐
x ───►  │   RMSNorm     │
        └──────┬────────┘
               ▼
        ┌───────────────┐
        │ Multi-Head Att│   (causal masked)
        └──────┬────────┘
               ▼
        ┌───────────────┐
        │ Residual Add  │  x + attention(x)
        └───────────────┘
               ▼
           output

'''

class RMSNorm(tf.keras.layers.Layer):
    def __init__(self,hidden_size,epsilon=1e-8,**kwargs):
        super(RMSNorm,self).__init__(**kwargs)
        self.hidden_size= hidden_size
        self.epsilon = epsilon 

        # Learnable scale parameter γ (same shape as last dim of input)
        self.scale = self.add_weight(
        name="scale",
        shape=(self.hidden_size,),
        initializer=tf.ones_initializer(),  
        trainable=True,
    )

    
    def call(self,x):
        rms = tf.sqrt(tf.reduce_mean(tf.square(x),axis=-1,keepdims=True)+self.epsilon)
        norm_x =x/rms
        return norm_x*self.scale
    
class CasualSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dropout=0.0):
        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )

        self.rmsnorm = RMSNorm(d_model)

    def call(self, x, padding_mask=None, training=None):
        norm_x = self.rmsnorm(x)

        # MultiHeadAttention expects mask shape: (batch, 1, 1, seq_len)
        attn = self.mha(
            query=norm_x,
            value=norm_x,
            key=norm_x,
            attention_mask=padding_mask,
            use_causal_mask=True,
            training=training
        )

        return x + attn   # residual connection


