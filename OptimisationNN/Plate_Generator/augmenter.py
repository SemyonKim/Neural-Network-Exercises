import cv2
import numpy as np
import random
from math import sin, pi
from PIL import Image, ImageDraw, ImageFont


def random_int(limit: int) -> int:
    """Return random integer in [0, limit)."""
    return np.random.randint(0, limit)


def apply_smudginess(img: np.ndarray, texture: np.ndarray) -> np.ndarray:
    """Overlay smudginess texture onto image."""
    img_h, img_w = img.shape[:2]
    rows = random_int(texture.shape[0] - img_h)
    cols = random_int(texture.shape[1] - img_w)

    overlay = texture[rows:rows + img_h, cols:cols + img_w]
    overlay = cv2.resize(overlay, (img_w, img_h))
    overlay = cv2.bitwise_not(overlay)

    alpha = random.uniform(0.0, 0.5)
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0.0)


def apply_rotation(img: np.ndarray, angle: float, shape: tuple, max_angle: float) -> np.ndarray:
    """Apply perspective rotation to image."""
    h, w = shape[:2]
    size = (w + int(h * sin(max_angle / 180 * pi)), h)
    interval = abs(int(sin(angle / 180 * pi) * h))

    pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])

    if angle > 0:
        pts2 = np.float32([
            [interval, 0], [0, size[1]],
            [size[0], 0], [size[0] - interval, h]
        ])
    else:
        pts2 = np.float32([
            [0, 0], [interval, size[1]],
            [size[0] - interval, 0], [size[0], h]
        ])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, size)


def apply_random_distortion(img: np.ndarray, factor: int, size: tuple) -> np.ndarray:
    """Apply random perspective distortion."""
    h, w = size
    pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
    pts2 = np.float32([
        [random_int(factor), random_int(factor)],
        [random_int(factor), h - random_int(factor)],
        [w - random_int(factor), random_int(factor)],
        [w - random_int(factor), h - random_int(factor)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, size)


def blend_with_random_background(img: np.ndarray, dataset: list) -> np.ndarray:
    """Blend image with random background environment."""
    env = cv2.imread(dataset[random_int(len(dataset))])
    env = cv2.resize(env, (img.shape[1], img.shape[0]))
    alpha = random.uniform(0.0, 0.4)
    return cv2.addWeighted(img, 1 - alpha, env, alpha, 0.0)


def render_large_char(font: ImageFont.FreeTypeFont, char: str) -> np.ndarray:
    """Render large character image."""
    img = Image.new("RGB", (50, 94), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 36), char, (0, 0, 0), font=font)
    return np.array(img)


def render_small_char(font: ImageFont.FreeTypeFont, char: str) -> np.ndarray:
    """Render small character image."""
    img = Image.new("RGB", (40, 73), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 26), char, (0, 0, 0), font=font)
    return np.array(img)


def apply_gaussian_blur(img: np.ndarray, level: int) -> np.ndarray:
    """Apply Gaussian blur with kernel size based on level."""
    ksize = level * 2 + 1
    return cv2.blur(img, (ksize, ksize))


def add_noise_to_channel(channel: np.ndarray) -> np.ndarray:
    """Add random noise to a single image channel."""
    diff = 255 - channel.max()
    noise = np.random.normal(0, 1 + random_int(6), channel.shape)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = (diff * noise).astype(np.uint8)
    return channel + noise


def apply_random_noise(img: np.ndarray) -> np.ndarray:
    """Apply random noise to all RGB channels."""
    for c in range(3):
        img[:, :, c] = add_noise_to_channel(img[:, :, c])
    return img