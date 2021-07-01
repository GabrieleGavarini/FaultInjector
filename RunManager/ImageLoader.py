import tensorflow as tf
from tensorflow.keras.preprocessing import image

class ImageLoader:

    @staticmethod
    def load_resize_center(path, resize_target=256, crop_target=224):
        """
        Load the image. Resize the image to a square image and then center-crop it.
        :param path: path of the image
        :param resize_target: target length for the image reshape
        :param crop_target: target length for the image crop
        :return: the loaded image, resized and cropped
        """
        loaded_image = image.load_img(path)
        loaded_image = image.img_to_array(loaded_image)
        loaded_image = tf.image.resize(loaded_image, (resize_target, resize_target))
        loaded_image = tf.image.central_crop(loaded_image, crop_target / resize_target)

        return loaded_image
