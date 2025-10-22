# testgpu.py
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"✅ GPU(s) detected: {len(gpus)}")
    for gpu in gpus:
        print("   ", gpu)
else:
    print("❌ No GPU detected. TensorFlow is running on CPU.")
