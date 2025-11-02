"""
rotary.py
Llama Architecture Implementation
"""

# Rotary Positional Encoding
"""
mathematical way to incorporate postional information of words in a text directly in attention mechanism.
The same word at different positions ends up with different rotated vectors
The inner product between tokens reflects their relative distances"""

def apply_rope(x,sin,cos):
    '''
    x : (B,h,T,d)   even-sized last dim ( d must be multiple of 2)
    sin: (T,d//2)   broadcastable
    cos: (T,d//2)
    '''
    # this seperates each features vector's dimension ito 2 halves - real and imaginary
    x_even = x[...,0::2] # get even-dimension values -> shape: (B,h,T,d/2)
    x_odd = x[...,1::2]  # get odd-dimension values -> shape: (B,h,T,d/2)

    # this is 2D rotation formulas applied to each positional index and head.
    # it rotates the embedding vector in its dimensional space based on position

    x_rot_even = x_even*cos-x_odd*sin
    x_rot_odd = x_even*sin-x_odd*cos

    x_rot = tf.stack([x_rot_even,x_rot_odd],axis=-1)
    return tf.reshape(x_rot,tf.shape(x))

def make_sincos(seq_len,dim,base=10000):
    """
    Returns sin , cos with shape (seq_len,dim//2)
    """
    pos = tf.cast(tf.range(seq_len),tf.float32)
    i = tf.cast(tf.range(0,dim,2),tf.float32)/dim
    theta = pos[:,None]/(base**i[None,:])
    return tf.sin(theta),tf.cos(theta)


