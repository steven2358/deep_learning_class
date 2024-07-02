# Training code which uses TensorFlow's enable_op_determinism. 
# "When op determinism is enabled, TensorFlow ops will be deterministic.
# This means that if an op is run multiple times with the same inputs 
# on the same hardware, it will have the exact same outputs each time."

EPOCHS = 15
BUFFER_SIZE = 500
BATCH_SIZE = 32

tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()
train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(processed_image_ds.element_spec)

unet = unet_model((img_height, img_width, num_channels))
unet.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model_history = unet.fit(train_dataset, epochs=EPOCHS)
