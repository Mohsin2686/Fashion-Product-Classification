import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf


class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename

    @staticmethod
    def _preprocess_image(image_path: str, IMG_HEIGHT:int=150, IMG_WIDTH:int=150):
        """Load and preprocess a single image."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        image = image / 255.0  # Normalize to [0, 1]
        image = tf.expand_dims(image, axis=0)
        return image
    


    
    def predict(self):
        ## load model
        
        model = load_model(os.path.join("artifacts","training", "model.keras"))
        #model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename

        test_image = self._preprocess_image(imagename) 
        

        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 0:
            prediction = 'Causal Shoes'
            return [{ "image" : prediction}]
        elif result[0] == 1:
            prediction = 'Handbag'
            return [{ "image" : prediction}]
        elif result[0] == 2:
            prediction = 'Heels'
            return [{ "image" : prediction}]
        elif result[0] == 3:
            prediction = 'Kurtas'
            return [{ "image" : prediction}]
        elif result[0] == 4:
            prediction = 'Shirts'
            return [{ "image" : prediction}]
        elif result[0] == 5:
            prediction = 'Sports Shoes'
            return [{ "image" : prediction}]
        elif result[0] == 6:
            prediction = 'Sunglasses'
            return [{ "image" : prediction}]
        elif result[0] == 7:
            prediction = 'Tops'
            return [{ "image" : prediction}]
        elif result[0] == 8:
            prediction = 'TShirts'
            return [{ "image" : prediction}]
        elif result[0] == 9:
            prediction = 'Watches'
            return [{ "image" : prediction}]