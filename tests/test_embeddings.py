"""
Unit tests for embeddings module.
"""
import tensorflow as tf
from src.embeddings import TokenEmbedding, PositionalEmbedding


def test_token_embedding():
    """Test token embedding layer."""
    vocab_size = 1000
    d_model = 256
    
    layer = TokenEmbedding(vocab_size, d_model)
    input_tokens = tf.constant([[1, 2, 3, 4]])
    
    output = layer(input_tokens)
    
    assert output.shape == (1, 4, d_model)
    print("✓ TokenEmbedding test passed")


def test_positional_embedding():
    """Test positional embedding layer."""
    max_len = 100
    d_model = 256
    
    layer = PositionalEmbedding(max_len, d_model)
    input_tokens = tf.constant([[1, 2, 3, 4]])
    
    output = layer(input_tokens)
    
    assert output.shape == (1, 4, d_model)
    print("✓ PositionalEmbedding test passed")


if __name__ == '__main__':
    test_token_embedding()
    test_positional_embedding()
    print("\n✅ All tests passed!")
