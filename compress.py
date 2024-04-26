from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import math

def dct2(f):
    """2D Discrete Cosine Transform
    Args:
        f: Square array
    Returns: 
        2D DCT of f
    """
    return dct(dct(f, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(f):
    """2D Inverse Discrete Cosine Transform
    Args:
        f: Square array
    Returns: 
        2D Inverse DCT of f
    """
    return idct(idct(f, axis=0 , norm='ortho'), axis=1 , norm='ortho')

BLOCK_SIZE = 8
QUALITY_FACTOR = 2
QUANTIZATION_BITS_Y = 4
QUANTIZATION_BITS_UV = 4

mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
mask[0,0] = 1
if QUALITY_FACTOR >= 2:
    mask[1,0] = 1
    mask[0,1] = 1
elif QUALITY_FACTOR >= 3:
    mask[1,1] = 1
    mask[0,2] = 1
    mask[1,2] = 1
    mask[2,0] = 1
    mask[2,1] = 1
print(f'using {np.count_nonzero(mask)}/{BLOCK_SIZE * BLOCK_SIZE} DCT coefficients')

# Function to apply mask on DCT coefficients
def apply_mask(dct_matrix):
    return dct_matrix * mask

# Load TIFF image
image = Image.open("lena_color.tif")

# Convert to YUV
yuv_image = image.convert("YCbCr")

# Convert YUV image to numpy array
yuv_data = np.array(yuv_image)

# Split YUV channels
y_channel = yuv_data[:, :, 0]
u_channel = yuv_data[:, :, 1]
v_channel = yuv_data[:, :, 2]
# MAX_Y = np.max(yuv_data[:, :, 0])
# MAX_U = np.max(yuv_data[:, :, 1])
# MAX_V = np.max(yuv_data[:, :, 2])
MAX_Y = 255  # from 8-bit RGB
MAX_U = 255  # from 8-bit RGB
MAX_V = 255  # from 8-bit RGB
print(f'MAX_Y = {MAX_Y}, MAX_U = {MAX_U}, MAX_V = {MAX_V}')

blocks = np.zeros((BLOCK_SIZE, BLOCK_SIZE, y_channel.shape[0] // BLOCK_SIZE, y_channel.shape[1] // BLOCK_SIZE, 3), int)

# Perform DCT on each 8x8 block of the Y, U, and V channels
MAX_QUANT_Y = 2 ** QUANTIZATION_BITS_Y - 1
MAX_QUANT_UV = 2 ** QUANTIZATION_BITS_UV - 1
for y in range(0, y_channel.shape[0], BLOCK_SIZE):
    for x in range(0, y_channel.shape[1], BLOCK_SIZE):
        blocks[:, :, y // BLOCK_SIZE, x // BLOCK_SIZE, 0] = np.round(dct2(y_channel[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]) / (MAX_Y * BLOCK_SIZE) * MAX_QUANT_Y)
        blocks[:, :, y // BLOCK_SIZE, x // BLOCK_SIZE, 1] = np.round(dct2(u_channel[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]) / (MAX_U * BLOCK_SIZE) * MAX_QUANT_UV)
        blocks[:, :, y // BLOCK_SIZE, x // BLOCK_SIZE, 2] = np.round(dct2(v_channel[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]) / (MAX_V * BLOCK_SIZE) * MAX_QUANT_UV)

# Perform Inverse DCT on each 8x8 block of the Y, U, and V channels
for y in range(0, y_channel.shape[0] // BLOCK_SIZE):
    for x in range(0, y_channel.shape[1] // BLOCK_SIZE):
        y_channel[y * BLOCK_SIZE : (y+1) * BLOCK_SIZE, x * BLOCK_SIZE : (x+1) * BLOCK_SIZE] = idct2(apply_mask(blocks[:, :, y, x, 0]) * (MAX_Y * BLOCK_SIZE) / MAX_QUANT_Y)
        u_channel[y * BLOCK_SIZE : (y+1) * BLOCK_SIZE, x * BLOCK_SIZE : (x+1) * BLOCK_SIZE] = idct2(apply_mask(blocks[:, :, y, x, 1]) * (MAX_U * BLOCK_SIZE) / MAX_QUANT_UV)
        v_channel[y * BLOCK_SIZE : (y+1) * BLOCK_SIZE, x * BLOCK_SIZE : (x+1) * BLOCK_SIZE] = idct2(apply_mask(blocks[:, :, y, x, 2]) * (MAX_V * BLOCK_SIZE) / MAX_QUANT_UV)

compression_factor = 8*8*8 / QUANTIZATION_BITS_Y / QUANTIZATION_BITS_UV / QUANTIZATION_BITS_UV * BLOCK_SIZE * BLOCK_SIZE / np.count_nonzero(mask)

plt.figure(figsize=(30, 10))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image [512 x 512 x 24]')
plt.axis('off')
plt.axis('tight')

# Processed image
yuv_data[:, :, 0] = y_channel
yuv_data[:, :, 1] = u_channel
yuv_data[:, :, 2] = v_channel
processed_image = Image.fromarray(yuv_data, "YCbCr").convert("RGB")
plt.subplot(1, 3, 2)
plt.imshow(processed_image)
plt.title(f'DCT Compressed Image [compression factor = {compression_factor}]')
plt.axis('off')
plt.axis('tight')

# Processed image
yuv_data[:, :, 0] = y_channel
yuv_data[:, :, 1] = 100
yuv_data[:, :, 2] = 100
processed_image = Image.fromarray(yuv_data, "YCbCr").convert("RGB")
plt.subplot(1, 3, 3)
plt.imshow(processed_image)
plt.title(f'DCT Compressed Image [compression factor = {compression_factor * (QUANTIZATION_BITS_Y + QUANTIZATION_BITS_UV) / QUANTIZATION_BITS_Y}]')
plt.axis('off')
plt.axis('tight')

plt.show()
