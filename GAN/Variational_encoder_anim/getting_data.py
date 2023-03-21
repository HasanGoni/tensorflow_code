from pathlib import Path
import numpy as np
import zipfile
import urllib.request

np.random.seed(51)
batch_size = bs = 2000
latent_dim = 512
image_size = 64

path = Path(r'/tmp/anime')
path.mkdir(parents=True,
           exist_ok=True)

data_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/Resources/anime-faces.zip"
file_name = 'animefaces.zip'
download_dir = path
urllib.request.urlretrieve(data_url,
                           file_name)
zip_ref = zipfile.ZipFile(file_name,
                          'r')
zip_ref.extractall(download_dir)
zip_ref.close()
