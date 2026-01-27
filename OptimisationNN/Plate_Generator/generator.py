import os
import cv2
import pickle
import numpy as np
from PIL import Image, ImageFont
from augmenter import (
    render_large_char,
    render_small_char,
    apply_rotation,
    apply_random_distortion,
    apply_smudginess,
    blend_with_random_background,
    apply_gaussian_blur,
    apply_random_noise
)


class GenPlate:
    """Synthetic license plate generator with augmentation."""

    def __init__(self, font_path: str, no_plates_dir: str):
        # Load fonts
        self.font_large = ImageFont.truetype(font_path, 116)
        self.font_small = ImageFont.truetype(font_path, 86)

        # Base plate template
        self.img = np.array(Image.new("RGB", (520, 112), (255, 255, 255)))
        self.bg = cv2.resize(cv2.imread("./images/lpr.png"), (520, 112))
        self.smu = cv2.imread("./images/smu2.jpg")

        # Collect background images
        self.noplates_path = [
            os.path.join(parent, filename)
            for parent, _, filenames in os.walk(no_plates_dir)
            for filename in filenames
        ]

    def draw(self, chars: str) -> np.ndarray:
        """Render characters onto plate template."""
        self.img[0:94, 36:86] = render_large_char(self.font_large, chars[0])
        self.img[0:94, 100:150] = render_large_char(self.font_large, chars[1])
        self.img[0:94, 155:205] = render_large_char(self.font_large, chars[2])
        self.img[0:94, 210:260] = render_large_char(self.font_large, chars[3])
        self.img[0:94, 265:315] = render_large_char(self.font_large, chars[4])
        self.img[0:94, 320:370] = render_large_char(self.font_large, chars[5])
        self.img[0:73, 399:439] = render_small_char(self.font_small, chars[6])
        self.img[0:73, 444:484] = render_small_char(self.font_small, chars[7])
        return self.img

    def generate(self, text: str) -> np.ndarray:
        """Generate augmented license plate image from text."""
        if len(text) != 8:
            raise ValueError("License plate text must be 8 characters long.")

        fg = self.draw(text)
        com = cv2.bitwise_and(fg, self.bg)
        com = apply_rotation(com, np.random.randint(-20, 20), com.shape, 20)
        com = apply_random_distortion(com, 5, (com.shape[0], com.shape[1]))
        com = apply_smudginess(com, self.smu)
        com = blend_with_random_background(com, self.noplates_path)
        com = apply_gaussian_blur(com, 1 + np.random.randint(2))
        com = apply_random_noise(com)
        return com


# Character set
CHARS = ["A", "B", "C", "D", "E", "H", "K", "M", "O", "P",
         "T", "X", "Y", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def gen_rand():
    """Generate random license plate string and label indices."""
    label = []
    label.append(np.random.randint(0, 13))       # First letter
    label.extend(np.random.randint(13, 23, 3))  # Three digits
    label.extend(np.random.randint(0, 13, 2))   # Two letters
    label.extend(np.random.randint(13, 23, 2))  # Two digits

    name = "".join(CHARS[idx] for idx in label)
    return name, label


def gen_sample(generator: GenPlate, width: int, height: int):
    """Generate one sample image and label."""
    name, label = gen_rand()
    img = generator.generate(name)
    img = cv2.resize(img, (width, height))
    return label, name, img


def gen_batch(generator: GenPlate, batch_size: int, output_path: str):
    """Generate batch of synthetic license plates and save to disk."""
    os.makedirs(output_path, exist_ok=True)
    labels = []

    for i in range(batch_size):
        label, name, img = gen_sample(generator, 140, 30)
        labels.append(label)
        filename = os.path.join(output_path, f"{i:04d}.jpg")
        cv2.imwrite(filename, img)

    np.savetxt("label.txt", labels, fmt="%d")
    return labels


def prepare_dataset(batch_size: int, output_path: str):
    """Generate dataset and save as pickle."""
    labels = np.loadtxt("label.txt")
    one_hot = np.zeros((batch_size, 23))

    for i in range(batch_size):
        for j in range(8):
            one_hot[i, int(labels[i, j])] = 1

    img_data = np.zeros((batch_size, 30, 140, 3))
    for i in range(batch_size):
        img_path = os.path.join(output_path, f"{i:04d}.jpg")
        img_temp = cv2.imread(img_path)
        img_data[i] = np.reshape(img_temp, (30, 140, 3))

    with open("train_data.pkl", "wb") as f:
        pickle.dump(img_data, f)

    print("Dataset generation complete.")


if __name__ == "__main__":
    batch_size = 20000
    output_dir = "./data/train_data"
    font_path = "./font/RoadNumbers2.0.ttf"
    bg_dir = "./NoPlates"

    generator = GenPlate(font_path, bg_dir)
    gen_batch(generator, batch_size, output_dir)
    prepare_dataset(batch_size, output_dir)