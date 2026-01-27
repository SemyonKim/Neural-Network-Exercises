import os
import argparse
from math import *
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
from PlateCommon import *  # Assumed to contain helper functions

class GenPlate:
    def __init__(self, fontEng, NoPlates):
        # Load fonts for rendering characters
        self.fontE = ImageFont.truetype(fontEng, 116, 0)
        self.fontE2 = ImageFont.truetype(fontEng, 86, 0)

        # Create blank white plate image
        self.img = np.array(Image.new("RGB", (520, 112), (255, 255, 255)))

        # Load background plate template and resize
        self.bg = cv2.resize(cv2.imread("./images/lpr.png"), (520, 112))

        # Load noise texture
        self.smu = cv2.imread("./images/smu2.jpg")

        # Collect background images from NoPlates directory
        self.noplates_path = []
        for parent, _, filenames in os.walk(NoPlates):
            for filename in filenames:
                path = os.path.join(parent, filename)
                self.noplates_path.append(path)
	

    def draw(self, val):
        space1 = 14
        space2 = 5
        space3 = 0

        self.img[0:94, 36:86] = GenCh1(self.fontE, val[0])
        self.img[0:94, 100:150] = GenCh1(self.fontE, val[1])
        self.img[0:94, 155:205] = GenCh1(self.fontE, val[2])
        self.img[0:94, 210:260] = GenCh1(self.fontE, val[3])
        self.img[0:94, 265:315] = GenCh1(self.fontE, val[4])
        self.img[0:94, 320:370] = GenCh1(self.fontE, val[5])
        self.img[0:73, 399:439] = GenCh2(self.fontE2, val[6])
        self.img[0:73, 444:484] = GenCh2(self.fontE2, val[7])

        return self.img

    def generate(self, text):
        if len(text) == 8:
            fg = self.draw(text)
            com = cv2.bitwise_and(fg, self.bg)
            com = rot(com, r(40) - 20, com.shape, 20)
            com = rotRandom(com, 5, com.shape[1], com.shape[0])
            com = AddSmudginess(com, self.smu)
            com = random_enviroment(com, self.noplates_path)
            com = AddGauss(com, 1 + r(2))
            com = addNoise(com)
            return com
