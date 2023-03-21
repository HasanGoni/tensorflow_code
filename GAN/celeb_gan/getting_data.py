import numpy as np
import urllib.request
from pathlib import Path
import zipfile
import glob
from tqdm import tqdm
from PIL import Image
import tensorflow as tf

path = Path(r'/tmp/celeb')
path.mkdir(parents=True,
           exist_ok=True)
url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/Resources/archive.zip"
file_name = 'archive.zip'

urllib.request.urlretrieve(url,
                           file_name)


def extract_zip(path,
                filename,
                download_dir=None):
    """
    extracting zip file
    in a path
    filename: str
    path: path where zipfile is there
    download_dir: where data needs to
    be downloaded, if None
    there same directory
    """
    path = Path(path)
    zip_ref = zipfile.ZipFile(path/filename, 'r')
    if download_dir is None:
        down_dir = path
    else:
        down_dir = download_dir
    zip_ref.extractall(down_dir)
    zip_ref.close()


def load_celeb(bs,
               resize=64,
               crop_size=128):
    """
    bs: batch size
    crop_size: croping centering of the
    image. This will be centering
    resize: resize of the image
    """
    image_path =glob.glob(r'./img_align_celeba/img_align_celeba/*.jpg')
    image_number = len(image_path)
    images = np.zeros((image_number,
                      resize,
                      resize,
                      3), np.uint8)
    # Crop and resize the image
    for idx, im in enumerate(image_path):
        with Image.open(im) as ima:
            left = (ima.size[0] - crop_size) // 2
            top = (ima.size[1] - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            image = ima.crop((left,
                              top,
                              right,
                              bottom))
            image = image.resize((resize,
                                  resize),
                                 Image.LANCZOS)
            images[idx] = np.asarray(image,
                                     np.uint8)
    # Split the dataset and maybe use half
    # as label
    half_index = images.shape[0] // 2
    images1, images2 = images[:half_index], images[half_index: 2 * half_index]
    del images

    def preprocess(i):
        x = tf.cast(i, tf.float32) / 127.5 - 1.0
        return x

    dataset = tf.data.Dataset.from_tensor_slices((images1, images2))
    dataset = dataset.map(
        lambda x, y: (preprocess(x), preprocess(y))
    ).shuffle(buffer_size=4096).batch(bs,
                                      drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
