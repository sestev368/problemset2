# problemset2

https://colab.research.google.com/drive/1es6lvKcOCRx8sBFcnA9t98MzVOd4OPan?usp=sharing

# Convolution and Feature Map Visualization in Google Colab

In this report, we will go through the steps to perform image convolution using random filters and visualize both the filters and the resulting feature maps. We will use an image from the given URL:

![Original Image](https://cdn.britannica.com/22/75922-050-D3982BD0/flowers-fruits-garden-strawberry-plant-species.jpg)

## Setup

First, let's set up the environment in Google Colab. Ensure you have the required libraries installed:

```python
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display

# Define the URL of the image you want to load
image_url = "https://cdn.britannica.com/22/75922-050-D3982BD0/flowers-fruits-garden-strawberry-plant-species.jpg"

# Send an HTTP GET request to fetch the image
response = requests.get(image_url)

if response.status_code == 200:
    # Open the image using Pillow
    original_image = Image.open(BytesIO(response.content))

    # Check if the original image is in RGB mode (convert if needed)
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')

    # Convert the image to a NumPy array
    image_array = np.array(original_image)
    # Display the image
    display(original_image)

# Define the number of random filters
num_filters = 10

# Create and display random filters
plt.figure(figsize=(15, 6))
for i in range(num_filters):
    random_filter = np.random.rand(3, 3, 3)  # Random 3x3x3 filters
    plt.subplot(2, num_filters, i + 1)
    plt.imshow(random_filter)
    plt.axis('off')
    plt.title(f'Filter {i + 1}')

# Apply the filters and display feature maps
feature_maps = []
for i in range(num_filters):
    filter_weights = np.random.rand(3, 3, 3, 1)  # Random weights for each filter
    conv_result = tf.nn.conv2d(image_array[None, ...], filter_weights, strides=1, padding="SAME")[0].numpy()
    feature_maps.append(conv_result)

for i, feature_map in enumerate(feature_maps):
    plt.subplot(2, num_filters, i + num_filters + 1)
    plt.imshow(feature_map[:, :, 0], cmap='gray')  # Display the first channel of the feature map
    plt.axis('off')
    plt.title(f'Feature Map {i + 1}')

plt.show()

