import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image


class PredictionPipeline():
    def __init__(self) -> None:
        self.CLASS_NAMES = ['Malaria Infected cell', 'Healthy Cell']
        self.IMG_SIZE = 224

    def predict(self, input_img):
        # Loading all Deep Learning models
        resnet_152v2_model = load_model('./DL models/resnet152v2_model.h5')
        inception_resnetv2_model = load_model('./DL models/inception_resnet_model.h5')
        # Image Preprocessing
        image = Image.open(input_img)
        image = tf.cast(image, dtype=tf.float32)
        image = image / 255.0
        input_tensor = tf.expand_dims(tf.image.resize(image, [self.IMG_SIZE, self.IMG_SIZE]), axis=0)
        # Making Predictions
        try:
            resnet_152v2_y_probs = resnet_152v2_model.predict(input_tensor)
            inception_resnet_v2_probs = inception_resnetv2_model.predict(input_tensor)
        except ValueError as err:
            return [[-1]], err, err, err
        else:
            return tf.round(resnet_152v2_y_probs), resnet_152v2_y_probs, tf.round(inception_resnet_v2_probs), inception_resnet_v2_probs
