import glob
from  pathlib import Path

path = Path.cwd()

das = glob.glob(r'./img_align_celeba/img_align_celeba/*.jpg')
print(das)
