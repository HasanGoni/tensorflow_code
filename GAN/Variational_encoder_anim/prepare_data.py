from pathlib import Path
import tensorflow as tf

Path.ls = lambda x: sorted(list((x.iterdir())))
path = Path(r'/tmp/anime/images')


def get_dataset(image_dir):
    """
    getting image list from
    image_dir
    """
    image_dir = Path(image_dir)
    image_list = [i.as_posix() for i in image_dir.ls()]
    return image_list


def map_image(image_filename,
              im_size=64):
    """
    preprocess image files and
    convert them to tensorflow dataset
    """
    image = tf.io.read_file(image_filename)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image,
                            (im_size, im_size))
    image = image / 255.0
    image = tf.reshape(image, shape=(im_size,
                                     im_size,
                                     3,))
    return image

print(len(get_dataset(path)))
