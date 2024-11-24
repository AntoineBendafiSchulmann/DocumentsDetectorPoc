import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

train_dir = 'dataset/generated'

def load_images_from_directory(directory, img_size=(224, 224)):
    dataset = tf.data.Dataset.list_files(os.path.join(directory, "*.jpg"))
    dataset = dataset.map(lambda x: process_image(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def process_image(file_path, img_size):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = image / 255.0  
    label = 1 
    return image, label


full_dataset = load_images_from_directory(train_dir)
train_size = int(0.8 * tf.data.experimental.cardinality(full_dataset).numpy())
train_dataset = full_dataset.take(train_size)
val_dataset = full_dataset.skip(train_size)

train_dataset = train_dataset.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)


base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, validation_data=val_dataset, epochs=5)

model.save('document_detector_model.h5')
