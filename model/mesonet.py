import tensorflow as tf

class MesoNet:
    def __init__(self, input_shape=(64, 64, 1)):
        self.input_shape = input_shape

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        # First convolutional block
        x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
        
        # Second convolutional block
        x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
        
        # Third convolutional block
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = tf.keras.layers.Dense(32, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)