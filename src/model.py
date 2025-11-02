"""
model.py
Llama Architecture Implementation
"""

# Decoder And Transformer Base Code

class FeedForward(tf.keras.layers.Layer):
    def __init__(self,d_model,dropout_rate=0.1):
        super().__init__()

        self.seq =tf.keras.Sequential(
            [
                SwiGLU(d_model),
                tf.keras.layers.Dropout(dropout_rate)
            ]
        )
        self.rmsnorm =RMSNorm(d_model)

    def call(self,x,training=None):
        y =self.seq(self.rmsnorm(x),training=training)  # pre-norm
        return x+y                                      # residual on raw x
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,*,d_model,num_heads,dropout_rate=0.1):
        super().__init__()
        self.causal_self_attention = CasualSelfAttention(num_heads=num_heads,d_model=d_model,dropout=dropout_rate)
        self.ffn = FeedForward(d_model)
        
    def call(self,x,padding_mask=None,training=None):
        x = self.causal_self_attention(x, padding_mask=padding_mask, training=training)
        x = self.ffn(x, training=training)
        return x
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self,*,num_layers,d_model,num_heads,vocab_size,max_len,dropout_rate=0.1): # Added max_len
        super(Decoder,self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embedding = TokenEmbedding(vocab_size=vocab_size,d_model=d_model) # Renamed
        self.pos_embedding = PositionalEmbedding(max_len=max_len, d_model=d_model)   # ADDED
        self.dropout =tf.keras.layers.Dropout(dropout_rate)

        self.dec_layers = [DecoderLayer(d_model=d_model,num_heads=num_heads,dropout_rate=dropout_rate) for _ in range(num_layers)]
        self.last_attn = None

    def call(self, x, training=None):
        # Get padding mask from the *token* embedding layer
        pad_mask = self.token_embedding.compute_mask(x)
        if pad_mask is not None:
            pad_mask = tf.cast(~pad_mask, tf.float32)[:, tf.newaxis, tf.newaxis, :]

        # Create token embeddings
        x_tok = self.token_embedding(x)
        # Create positional embeddings (shape is inferred from x_tok)
        x_pos = self.pos_embedding(x_tok) 

        # ADD the two embeddings together
        x = x_tok + x_pos
        x = self.dropout(x, training=training)

        for layer in self.dec_layers:
            x = layer(x, padding_mask=pad_mask, training=training)

        return x
    
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads,max_len, input_vocab_size, dropout_rate=0.1):
        super().__init__()

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            vocab_size=input_vocab_size,
            max_len=max_len,
            dropout_rate=dropout_rate,
        )

        self.rmsnorm = RMSNorm(d_model)
        self.final_layer = tf.keras.layers.Dense(input_vocab_size)

    def call(self, inputs,training=False):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        x = inputs

        x = self.decoder(x,training=training)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        x = self.rmsnorm(x)
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output
        return logits

