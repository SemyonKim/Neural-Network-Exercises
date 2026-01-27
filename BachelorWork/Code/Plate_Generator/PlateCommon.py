import cv2
import numpy as np
import random
from math import sin
from PIL import Image, ImageDraw, ImageFont


def r(val):
    return int(np.random.random() * val)

def AddSmudginess(img, Smu):
    img_h, img_w = img.shape[:2]
    rows = r(Smu.shape[0] - img_h)
    cols = r(Smu.shape[1] - img_w)

    adder = Smu[rows:rows + img_h, cols:cols + img_w]
    adder = cv2.resize(adder, (img_w, img_h))
    adder = cv2.bitwise_not(adder)

    val = random.random() * 0.5
    img = cv2.addWeighted(img, 1 - val, adder, val, 0.0)
    return img

def rot(img, angel, shape, max_angel):
    size_o = [shape[1], shape[0]]
    size = (
        shape[1] + int(shape[0] * sin((float(max_angel) / 180) * 3.14)),
        shape[0]
    )
    interval = abs(int(sin((float(angel) / 180) * 3.14) * shape[0]))

    pts1 = np.float32([
        [0, 0], [0, size_o[1]],
        [size_o[0], 0], [size_o[0], size_o[1]]
    ])

    if angel > 0:
        pts2 = np.float32([
            [interval, 0], [0, size[1]],
            [size[0], 0], [size[0] - interval, size_o[1]]
        ])
    else:
        pts2 = np.float32([
            [0, 0], [interval, size[1]],
            [size[0] - interval, 0], [size[0], size_o[1]]
        ])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    return dst

def rotRandom(img, factor, size):
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([
        [r(factor), r(factor)],
        [r(factor), shape[0] - r(factor)],
        [shape[1] - r(factor), r(factor)],
        [shape[1] - r(factor), shape[0] - r(factor)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    return dst

def random_enviroment(img, data_set):
    index = r(len(data_set))
    env = cv2.imread(data_set[index])
    env = cv2.resize(env, (img.shape[1], img.shape[0]))
    val = random.random() * 0.4
    img = cv2.addWeighted(img, 1 - val, env, val, 0.0)
    return img

def GenCh1(f, val):
    img = Image.new("RGB", (50, 94), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 36), val, (0, 0, 0), font=f)
    return np.array(img)

def GenCh2(f, val):
    img = Image.new("RGB", (40, 73), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 26), val, (0, 0, 0), font=f)
    return np.array(img)

def AddGauss(img, level):
    # Apply Gaussian blur using a square kernel of size (2*level + 1)
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1))

def r(val):
    return int(np.random.random() * val)

def AddNoiseSingleChannel(single):
    # Add random noise to a single image channel
    diff = 255 - single.max()
    noise = np.random.normal(0, 1 + r(6), single.shape)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = diff * noise
    noise = noise.astype(np.uint8)
    dst = single + noise
    return dst

def addNoise(img, sdev=0.5, avg=10):
    # Apply noise to each RGB channel
    img[:, :, 0] = AddNoiseSingleChannel(img[:, :, 0])
    img[:, :, 1] = AddNoiseSingleChannel(img[:, :, 1])
    img[:, :, 2] = AddNoiseSingleChannel(img[:, :, 2])
    return img