import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

BOUNDARYINK = 5
WINDOWSIZEX = 640
WINDOWSIZEY = 480

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

IMAGESAVE = False

MODEL = load_model("best_model.h5")

LABELS = {
    0:'Zero',
    1:'One',
    2:'Two',
    3:'Three',
    4:'Four',
    5:'Five',
    6:'Six',
    7:'Seven',
    8:'Eight',
    9:'Nine',
}

# Initialize our pygame

pygame.init()


FONT = pygame.font.Font("freesansbold.ttf", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX,WINDOWSIZEY))


pygame.display.set_caption("Digit Recognition")

iswriting = False

number_xcord = []
number_ycord = []

image_cnt=1

PREDICT = True

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARYINK, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDARYINK)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDARYINK, 0), min(WINDOWSIZEY, number_ycord[-1] + BOUNDARYINK)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(
                np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                image_cnt += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])
                
                # Add padding around the green rectangle
                padding = 5
                pygame.draw.rect(DISPLAYSURF, GREEN, (rect_min_x - padding, rect_min_y - padding,
                                                     rect_max_x - rect_min_x + 2 * padding, rect_max_y - rect_min_y + 2 * padding), 2)

                # Render the label text
                textSurface = FONT.render(label, True, RED, WHITE)
                textRectObj = textSurface.get_rect()
                textRectObj.left, textRectObj.bottom = rect_min_x, rect_max_y

                # Draw the label text inside the rectangle
                DISPLAYSURF.blit(textSurface, textRectObj)

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()