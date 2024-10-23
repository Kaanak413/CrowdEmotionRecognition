import tensorflow as tf
print(tf.version.VERSION
)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))