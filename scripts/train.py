"""
Training script for Llama text summarization model.
"""
import tensorflow as tf
from src.model import Transformer
from src.config import *

def main():
    print("🚀 Starting training...")
    
    # Initialize model
    model = Transformer(
        num_layers=4,
        d_model=256,
        num_heads=8,
        input_vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        dropout_rate=0.1
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']
    )
    
    print("✅ Model compiled successfully!")
    
    # Add your training loop here
    # model.fit(train_ds, epochs=10, validation_data=val_ds)

if __name__ == '__main__':
    main()
