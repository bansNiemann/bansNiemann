# importing the module
import cv2
import pyautogui as pg
import numpy as np
import time
from PIL import Image

class RecordMouseInputs():
    def __init__(self):
        self.board_size = None
        self.cell_size = None
        self.board_top_coord = None
        self.board_left_coord = None
        self.board_bot_coord = None
        self.board_right_coord = None



    def record(self,event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            self.board_left_coord = y
            self.board_top_coord = x
        # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            self.board_right_coord = y
            self.board_bot_coord = x

    def get_coords(self):
        return self.board_left_coord,self.board_top_coord,self.board_right_coord,self.board_bot_coord

    def calc_board_settings(self):
        self.board_size = int(self.board_right_coord - self.board_left_coord)
        self.cell_size = int(self.board_size / 8)

class ScreenshotCropper():
    def __init__(self,
                 board_left_coord,
                 board_top_coord,
                 board_right_coord,
                 board_bot_coord,
                 img_path = 'screenshot.png'):
        self.board_left_coord = board_left_coord
        self.board_top_coord = board_top_coord
        self.board_right_coord = board_right_coord
        self.board_bot_coord = board_bot_coord
        self.img_path = img_path

    def crop(self):
        img = cv2.imread(self.img_path)
        croppedImg = img[self.board_left_coord:self.board_right_coord,
                         self.board_top_coord:self.board_bot_coord]
        return croppedImg


# driver function
if __name__ == "__main__":
    recorder = RecordMouseInputs()
    # take a screenshot and store it locally
    pg.screenshot('screenshot.png')

    # load local screenshot
    img = cv2.imread('screenshot.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # record (make sure to left click and right click)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', recorder.record)
    cv2.waitKey(0)

    # calc and save
    cv2.destroyAllWindows()
