import numpy as np
from PIL import Image

def image_load(path: str):
  # This would load the image into a tensor where an index [x][y][rgb] would determin
  # the RGB component of the pixel at xy.
  # We want it a little inversed, we want [rgb][x][y], the color first, each 'frame' of the tensor
  # contains the whole image, so we reorder this tensor
  # A tensor transpose along the Z axis basically
  tensor = np.asarray(Image.open(path))
  return tensor.transpose(2, 0, 1)
def image_save(tensor: np.ndarray, path: str):
  # We need to reorder the tensor back to [x][y][rgb] order.
  Image.fromarray(tensor.transpose(1, 2, 0)).save(path)


image = image_load("apples.jpeg")
np.savetxt("apples.matrix", image[0])
print(image)
image_save(image, "output.jpeg")