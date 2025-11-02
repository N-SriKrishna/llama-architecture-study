"""
attention.py
Llama Architecture Implementation
"""

# Multi-head self attention
"""It captures multiple meaning in a sentence.
The philosphy is each attention head can focus on different parts of the input.
One head might focus on local context (e.g., next word)
Another might look at global structure (e.g., sentence-level relationships)."""

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Llama multi-head(scaled-dor-product) attention implemented from scratch.

    Args: 
        d_model (int) : total embedding size ( must be divisible by num_heads)
        num_heads(int): number of attention heads
        dropout(float): dropout on attenstion weights ( 0.0 = not dropout)
    
    Call signature :
        T_q = T_k = T_v

        output, attn_scores = mha(
        query,                     # (Batch_size, T_q, d_model)
        value=None,                # (B, T_v, d_model)  – defaults to query
        key=None,                  # (B, T_k, d_model)  – defaults to value
        mask=None,                 # (B, 1, T_q, T_k) or (B, T_q, T_k)
        use_causal_mask=False,     # True → autoregressive causal mask
        training=None
        )
    """

    def __init__(self,d_model,num_heads,dropout=0.0,**kwargs):
        super().__init__(**kwargs)

        if d_model % num_heads!=0 :
            raise ValueError (
                f"d_model={d_model} must be divisible by num_heads={num_heads}"
            )
        self.d_model =d_model
        self.num_heads =num_heads
        self.depth =d_model/num_heads

        # Linear projections fro Q,K,V and dinal output 
        self.wq = tf.keras.layers.Dense(d_model,use_bias=False)
        self.wk = tf.keras.layers.Dense(d_model,use_bias=False)
        self.wv = tf.keras.layers.Dense(d_model,use_bias=False)
        self.wo = tf.keras.layers.Dense(d_model,use_bias=False)

        self.dropout= tf.keras.layers.Dropout(dropout)

    # Helper Functions

    def _split_heads(self,x,B):
        """
        Reshape (Batch_size,sequence length,d_model) -> (Batch_size,num_heads,sequence length,depth)
        so we can run attention on each head in parallel
        """
        x - tf.reshape(x,(B,-1,self.num_heads,self.depth))  # x = (B, T, d_model)
        return tf.transpose(x,perm=[0,2,1,3])
    
    @staticmethod
    def _scaled_dot_product_attention(q,k,v,mask,dropout,training=None):
        """
        Core attention: softmax(QKᵀ / √d_k) V
        Returns: (B, num_heads, T_q, depth_v), (B, num_heads, T_q, T_k)
        """
        dk=tf.cast(tf.shape(k)[-1],tf.float32)
        scores=tf.matmul(q,k,transpose_b=True)/tf.math.sqrt(dk)  #(B,h,T_q,T_k)

        if mask is not None:
            # broadcast automatically if mask rank < scores rank
            scores+= (mask*-1e9)  # large negative = zero probability
    
        attn=tf.nn.softmax(scores,axis=-1) 
        attn=dropout(attn,training=training)
        output=tf.matmul(attn,v)  #(B,h,T_q,depth)
        return output
    
    # Forward Pass

    def call(
            self,
            query,
            value=None,
            key=None,
            mask=None,
            use_casual_mask=False,
            training=None
            ):
        if value is None:
            value = query
        if key is None:
            key = value
        B = tf.shape(query[0])

