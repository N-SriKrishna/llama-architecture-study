"""
Inference script for text generation.
"""
import tensorflow as tf
from src.model import Transformer
from src.inference import *

def main():
    print("🔮 Running inference...")
    
    # Load model
    # model = tf.keras.models.load_model('checkpoints/best_model.keras')
    
    # Generate text
    # result = generate_text(model, "Your prompt here")
    # print(result)

if __name__ == '__main__':
    main()
