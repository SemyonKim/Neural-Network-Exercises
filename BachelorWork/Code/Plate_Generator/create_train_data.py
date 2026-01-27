import numpy as np
import pandas as pd
import cv2
import os
import pickle

# Custom module for plate generation (assumed to exist)
from genplate_advanced import *

# Mapping characters to class indices
index = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "H": 5, "K": 6,
    "M": 7, "O": 8, "P": 9, "T": 10, "X": 11, "Y": 12,
    "0": 13, "1": 14, "2": 15, "3": 16, "4": 17, "5": 18,
    "6": 19, "7": 20, "8": 21, "9": 22
}

# Character set used for generation
chars = [
    "A", "B", "C", "D", "E", "H", "K", "M", "O", "P",
    "T", "X", "Y", "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9"
]

# Random integer in range [lo, hi)
def rand_range(lo, hi):
    return lo + r(hi - lo)

# Random integer from 0 to val
def r(val):
    return int(np.random.random() * val)

def gen_rand():
    name = ""
    label = []

    # Format: 1 letter + 3 digits + 2 letters + 2 digits
    label.append(rand_range(0, 13))         # First letter
    for i in range(3):                      # Three digits
        label.append(rand_range(13, 23))
    for i in range(2):                      # Two letters
        label.append(rand_range(0, 13))
    for i in range(2):                      # Two digits
        label.append(rand_range(13, 23))

    # Convert label indices to character string
    for i in range(8):
        name += chars[label[i]]

    return name, label


def gen_sample(genplate_advanced, width, height):
    name, label = gen_rand()
    img = genplate_advanced.generate(name)       # Generate synthetic plate image
    img = cv2.resize(img, (width, height))       # Resize to model input size
    return label, name, img


def genBatch(batchSize, outputPath):
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    label_store = []

    for i in range(batchSize):
        print('create num:', i)
        label, name, img = gen_sample(genplate_advanced, 140, 30)
        label_store.append(label)

        filename = os.path.join(outputPath, f"{i:04d}.jpg")
        cv2.imwrite(filename, img)

    # Save all labels to text file
    np.savetxt('label.txt', label_store)


# Parameters
batchSize = 20000
path = './data/train_data'
font_en = './font/RoadNumbers2.0.ttf'
bg_dir = './NoPlates'
# Initialize generator with font and background directory
genplate_advanced = GenPlate(font_en, bg_dir)

# Generate synthetic dataset
genBatch(batchSize=batchSize, outputPath=path)

# Load labels from text file
a = np.loadtxt('label.txt')  # shape: (batchSize, 8)

# One-hot encode labels into 23 classes (13 letters + 10 digits)
b = np.zeros([batchSize, 23])
for i in range(batchSize):
    for j in range(8):
        b[i, int(a[i, j])] = 1  # one-hot encoding

# Load and reshape image data
img_data = np.zeros([batchSize, 30, 140, 3])  # shape: (batchSize, H, W, C)
for i in range(batchSize):
    img_path = os.path.join(path, f"{i:04d}.jpg")
    img_temp = cv2.imread(img_path)
    img_temp = np.reshape(img_temp, (30, 140, 3))
    img_data[i] = img_temp

# Save image data to pickle file
with open('train_data.pkl', 'wb') as output:
    pickle.dump(img_data, output)

print("Dataset generation complete.")

