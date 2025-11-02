"""
embeddings.py
Llama Architecture Implementation
"""

# Word Embeddings 
'''A word embedding is a numerical representation of a word in a high-dimensional space.
The key idea behind this representation is that words with similar meanings will have similar numerical representations,
meaning the “distance” between their embeddings in this space will be small.'''

class TokenEmbedding(tf.keras.layers.Layer):
    """
    Args:
        vocab_size(int): Vocabulary size(number of unique tokens).
        d_model (int): Embedding dimension
    """
    def __init__(self,vocab_size,d_model):
        super().__init__()

        #Learnable token embeddings
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            mask_zero=True   # padding token is masked
        )

    def compute_mask(self,*args,**kwargs):
        # Pass through the mask from embedding layer
        return self.embedding.compute_mask(*args,**kwargs)

    def call(self,x):
        """
        Args:
            x: Tensor of token indices -> shape: (batch_size,sequence_length)
        
        Returns:
            Tensor of shape (batch_size,sequence_length,d_mask)
        """

        # COnvert token indices to dense vectorss -> (batch_size,seq_len,d_model)
        x=self.embedding(x)
        return x
    
class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Creates a learnable positional embedding.
    """
    def __init__(self, max_len, d_model):
        super().__init__()
        # Embedding layer for positions 0 to max_len-1
        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_len, output_dim=d_model)
        self.max_len = max_len

    def call(self, x):
        """
        Args:
            x: Token embeddings of shape (B, T, D)
        Returns:
            Positional embeddings of shape (B, T, D)
        """
        # Get the sequence length (T) from the input tensor
        T = tf.shape(x)[1] 
        # Create position indices: [0, 1, 2, ..., T-1]
        positions = tf.range(start=0, limit=T, delta=1)
        # Broadcast to match the batch size: (B, T)
        positions = tf.broadcast_to(positions, [tf.shape(x)[0], T])
        # Look up the positional embeddings
        return self.pos_emb(positions)

