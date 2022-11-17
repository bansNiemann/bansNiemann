import sys
import cv2
import numpy as np
import pyautogui as pg
import chess
from stockfish import Stockfish
import time
import copy
import argparse

import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt

from loadImage import ImageLoader
from getCoords import *
from transform import transformations
from model import ChessPositionNet
from fen import fen2matrix, matrix2fen

from config import target_dim
from config import device
from config import state_file_name


def calcBoardSettings(left, top, right, bot):
    boardSize = int(right - left)
    cellSize = int(boardSize / 8)
    return boardSize,cellSize

def initialScreenshot():
    # take a board snapshot
    pg.screenshot('screenshot.png')
    screenshot = cv2.imread('screenshot.png')
    # screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # record (make sure to left click and right click)
    recorder = RecordMouseInputs()
    cv2.imshow('Left click the top-left corner and right click the bottom-right corner', screenshot)
    cv2.setMouseCallback('Left click the top-left corner and right click the bottom-right corner', recorder.record)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    left, top, right, bot = recorder.get_coords()

    croppedImg = getCroppedBoard(left, top, right, bot)
    cv2.imshow('Cropped Image', croppedImg)
    print('Cropped image shape:', croppedImg.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return left, top, right, bot

def getCroppedBoard(left, top, right, bot):
    # calc and save
    pg.screenshot('screenshot.png')
    print('Left ({}), Top ({}), Right ({}), Bot ({})'.format(left, top, right, bot))
    cropper = ScreenshotCropper(left, top, right, bot)
    croppedImg = cropper.crop()
    cv2.imwrite('board.png', croppedImg)
    return croppedImg




# search position for a best move
def search(fen,depth = 7):
    # create chess board instance and set position from FEN string
    print('Searching best move for this position:')
    print(fen)
    board = chess.Board(fen=fen)
    print(board)
    stockfish.set_depth(depth)
    stockfish.set_fen_position(fen)
    print('Evaluation:',stockfish.get_evaluation())
    # best_move = stockfish.get_top_moves(3)[0]['Move']
    best_move = stockfish.get_best_move_time(1000)

    # # close engine
    # engine.quit()

    # search for the best move
    return best_move


def getFenPrediciton(screenshotPath = 'board.png'):
    # ImageLoader
    imgLoader = ImageLoader(img_size = [3,400,400])
    board = imgLoader.load(img_path = screenshotPath)

    print('Using device', device)
    model = ChessPositionNet(target_dim=target_dim).to(device)
    model.load_state_dict(torch.load(state_file_name, map_location=device))

    predicted = torch.zeros(8,8)
    for i in range(8):
        for j in range(8):
            x = board[i][j]
            _label = F.log_softmax(model(x), dim=1).argmax()
            predicted[i][j] = _label
    predictedFen = matrix2fen(predicted)
    return predictedFen

def flipFen(fen):
    rows = fen.split('/')
    flippedFen = []
    for row in rows[::-1]:
        flippedFen.append(row[::-1])
    flippedFen = '/'.join(flippedFen)
    return flippedFen

def formatFen(predictedFen,sideToStart,whiteAlwaysBottom = True):
    fen = predictedFen.replace('-', '/')

    if sideToStart and not whiteAlwaysBottom:
        fen = flipFen(fen)

    # add side to move to fen
    fen += ' ' + 'b' if sideToStart else ' w'
    board = chess.Board(fen=fen)
    fen = board.fen()
    return fen

def initializeMouseFrameOfReference(left, top, right, bot, sideToStart, whiteAlwaysBottom = True, offset = [-68,88]):
    # square to coords
    square_to_coords = []

    if sideToStart and not whiteAlwaysBottom:
        # array to convert board square indices to coordinates (black)
        get_square = [
            'h1', 'g1', 'f1', 'e1', 'd1', 'c1', 'b1', 'a1',
            'h2', 'g2', 'f2', 'e2', 'd2', 'c2', 'b2', 'a2',
            'h3', 'g3', 'f3', 'e3', 'd3', 'c3', 'b3', 'a3',
            'h4', 'g4', 'f4', 'e4', 'd4', 'c4', 'b4', 'a4',
            'h5', 'g5', 'f5', 'e5', 'd5', 'c5', 'b5', 'a5',
            'h6', 'g6', 'f6', 'e6', 'd6', 'c6', 'b6', 'a6',
            'h7', 'g7', 'f7', 'e7', 'd7', 'c7', 'b7', 'a7',
            'h8', 'g8', 'f8', 'e8', 'd8', 'c8', 'b8', 'a8',
        ]
    else:
        # array to convert board square indices to coordinates (black)
        get_square = [
            'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8',
            'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',
            'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6',
            'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5',
            'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4',
            'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3',
            'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',
            'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
        ]

    # board top left corner coords
    x = copy.deepcopy(left)
    y = copy.deepcopy(top)

    # loop over board rows
    for row in range(8):
        # loop over board columns
        for col in range(8):
            # init square
            square = row * 8 + col

            # associate square with square center coordinates
            square_to_coords.append((int(x + cellSize / 2), int(y + cellSize / 2)))

            # increment x coord by cell size
            x += cellSize

        # restore x coord, increment y coordinate by cell size
        x = copy.deepcopy(left)
        y += cellSize

    for i in range(len(square_to_coords)):
        square_to_coords[i] = (square_to_coords[i][0] + offset[0], square_to_coords[i][1] + offset[1])

    return square_to_coords, get_square

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sideToStart', type=str, required=True)
    parser.add_argument('--manualTimings', default=False, action='store_true')
    parser.add_argument('--whiteAlwaysBottom', default=True, action='store_false')

    args = parser.parse_args()
    sideToStart = args.sideToStart.lower()
    if sideToStart[0] == 'w':
        sideToStart = 0
    elif sideToStart[0] == 'b':
        sideToStart = 1
    else:
        print('Invalid starting side!')
        sys.exit(0)

    left, top, right, bot = initialScreenshot()
    boardSize, cellSize = calcBoardSettings(left,top,right,bot)

    square_to_coords, get_square = initializeMouseFrameOfReference(left,
                                                                   top,
                                                                   right,
                                                                   bot,
                                                                   sideToStart,
                                                                   args.whiteAlwaysBottom)

    ################################
    #
    #          Main driver
    #
    ################################

    # load Stockfish engine
    stockfish = Stockfish('../stockfish_eng/stockfish_14.1_win_x64_avx2')

    while True:
        try:
            getCroppedBoard(left, top, right, bot)

            # convert piece image coordinates to FEN string
            predictedFen = getFenPrediciton()
            fen = formatFen(predictedFen,sideToStart,args.whiteAlwaysBottom)

            best_move = search(fen)
            print('Best move:', best_move)

            # extract source and destination square coordinates
            from_sq = square_to_coords[get_square.index(best_move[0] + best_move[1])]
            to_sq = square_to_coords[get_square.index(best_move[2] + best_move[3])]
            print('fromsq {}, coord ({})'.format(get_square.index(best_move[0] + best_move[1]),from_sq))
            print('fromsq {}, coord ({})'.format(get_square.index(best_move[2] + best_move[3]),to_sq))

            # make move on board
            pg.moveTo(from_sq)
            pg.click()
            pg.moveTo(to_sq)
            pg.click()

            if args.manualTimings:
                input("Press Enter to continue...")
            else:
                time.sleep(10)

        except:
            sys.exit(0)











