import tensorflowjs as tfjs
import tensorflow as tf

model = tf.keras.models.load_model('document_detector_model.h5', compile=False)

tfjs.converters.save_keras_model(model, 'document-detector-frontend/public/js_model')
