from PIL import Image
import numpy as np
from scipy import ndimage

path = './input.png'
image = Image.open(path)
image = image.convert('RGBA')
data = np.array(image)

# Convert RGB to grayscale
gray = np.dot(data[...,:3], [0.2989, 0.5870, 0.1140])

# Threshold to determine black or white
data[(gray > 127) & (data[..., 3] == 255)] = [255, 255, 255, 255]
data[(gray <= 127) & (data[..., 3] == 255)] = [0, 0, 0, 255]

# Label contiguous regions of black pixels
labeled_array, num_features = ndimage.label(data[..., 0] == 0)

# Find regions smaller than 10,000 pixels and fill them with white
for label in range(1, num_features + 1):
    region_size = np.sum(labeled_array == label)
    if region_size < 10000:
        data[labeled_array == label] = [255, 255, 255, 255]

# Display the modified image
img = Image.fromarray(data)
#Save image to file
img.save('output.png')
