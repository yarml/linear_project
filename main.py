import numpy as np
from PIL import Image
from scipy.signal import convolve2d # Apparently NumPy cannot convolve matrices

def image_load(path: str) -> np.ndarray:
  # This would load the image into a tensor where an index [y][x][rgba] would determin
  # the RGBA component of the pixel at xy.
  # We want it a little inversed, we want [rgb][y][x], the color first, each 'frame' of the tensor
  # contains the whole image, so we reorder this tensor
  # A tensor transpose along the Z axis basically
  tensor = np.asarray(Image.open(path))
  return tensor.transpose(2, 0, 1)[:3]
def image_save(tensor: np.ndarray, path: str):
  # We need to reorder the tensor back to [y][x][rgba] order.
  Image.fromarray(tensor.transpose(1, 2, 0)).save(path)
def reduce_to_grayscale(tensor: np.ndarray) -> np.array:
  row = (tensor[0] + tensor[1] + tensor[2]) / 3
  return np.asarray([row, row, row]).astype(np.uint8)
def reduce_to_bin(tensor: np.ndarray, threshold: np.uint8) -> np.ndarray:
  row = np.where((tensor[0] + tensor[1] + tensor[2]) / 3 < threshold, 0, 255)
  return np.asarray([row, row, row]).astype(np.uint8)

def diff(f0: np.ndarray, f1: np.ndarray) -> np.ndarray:
  return (f1 - f0) * 2
def add(f0: np.ndarray, f1: np.ndarray) -> np.ndarray:
  return (f1 + f0).astype(np.uint8)
def invert(tensor: np.ndarray) -> np.ndarray:
  return 255 - tensor

def image_convolve(matrix, kernel):
	width = matrix[0].shape[0]
	height = matrix[0].shape[1]
	output_matrix = np.zeros((3, width, height))
	
	for ch in range(3):
		channel = matrix[ch]
		output_matrix[ch] = convolve2d(channel, kernel, mode='same', boundary='fill', fillvalue=0)
	return output_matrix.astype(np.uint8)

# Method 1:
# Because laziness, the frames are not separated by us, they should be in the files frame0.png and frame1.png
# We simply subtract them
def method1():
  frame0 = image_load("frame0.png")
  frame1 = image_load("frame1.png")
  #print(frame0)
  diff01 = diff(frame0, frame1)
  print(diff01.dtype)
  gs = reduce_to_grayscale(diff01)
  print(gs)
  image_save(gs, "edge_m1.png")

# Method 1.1, similar idea to 1, except instead of subtracting two frames,
# we add the first frame to the inverted second frame
def method1p1():
  frame0 = image_load("frame0.png")
  frame1 = image_load("frame1.png")
  inv1 = invert(frame1)
  image_save(reduce_to_bin(add(frame0/2, inv1/2), 39), "edge_m1.1.png")

# Method 2: Subtract the image from a shifted version of itself
def method2():
  shift_amnt = 1
  rm_indices = np.concatenate((np.arange(shift_amnt), np.array([-i - 1 for i in range(shift_amnt)])))
  #print(rm_indices)
  source = image_load("frame0.png")
  pshifted = np.roll(source, (shift_amnt, shift_amnt), (1, 2))
  inv_pshifted = invert(pshifted)
  # Adding with inverse then removing borders
  pshift_org = np.delete(np.delete(add(source/2, inv_pshifted/2), rm_indices, 1), rm_indices, 2)
  nshifted = np.roll(source, (-shift_amnt, -shift_amnt), (1, 2))
  inv_nshifted = invert(nshifted)
  nshift_org = np.delete(np.delete(add(source/2, inv_nshifted/2), rm_indices, 1), rm_indices, 2)
  #image_save(pshift_org, "pshift.jpeg")
  #image_save(nshift_org, "nshift.jpeg")
  output = reduce_to_bin(add(nshift_org, pshift_org), 1)
  image_save(output, "output.jpeg")

def method3():
  source = image_load("frame0.png")
  source = reduce_to_grayscale(source)
  output = reduce_to_bin(image_convolve(source, np.array([[-0.1, 0, 0.1], [-0.2, 0, 0.2], [-0.1, 0, 0.1]])), 60)
  image_save(output, "output.jpeg")

#method1()
#method1p1()
#method2()
method3()