"""
HandTracker or HandTrakingModule
================================

It is a simple module to detect hands and do simple projects with hand recognition and machine learning.

Hand Detector
-------------

Uses:
  1. To find and detect hand.
  2. To find 21 landmarks in each hand.
  3. To find the distance between any 2 landmarks.
  4. To find if a finger is up or not.
  5. Can create a rectangle for the best hand detection zone.

Functions and theris least requirements:
  1. HandDetector.FindHands:
      i. Requires `image` (with atleast one hand)
     ii. Returns an `image` with hand landmarks drawn on it
  2. HandDetector.FindLocation:
      i. Requires `image` (with atleast one hand)
     ii. Returns location of hand landmarks `lmloc` (dict), location of hand `handloc` (dict)
  3. HandDetector.DrawLandmarks:
      i. Requires `image` (with atleast one hand), `index` int value more than -1 but less than 21
     ii. Returns None
  4. HandDetector.fingersUp:
      i. Requires `image` (with atleast one hand)
     ii. Returns info dict [fingername: bool]
  5. HandDetector.fingersUp_PROTO:
      i. Requires None
     ii. Returns info dict [fingername: bool]
  6. HandDetector.findDistance:
      i. Requires `image` (with atleast one hand), `id` numbers of any two landmarks
     ii. Returns `image` with those landmarks drawn on it and a line connection those and the center point of that line, `length` the disance between 
  7. HandDetector.FindingZone:
      i. Requires `image` (with atleast one hand)
     ii. Returns location of rectangle `FindingZonedim` for the best hand detection zone

Other Uses
----------
  1. It can provide all the finger names used in this module, which is stored in `Fingers`.
  2. It can provide all the hand landmarks, which is stored in `HandLandmark`.
  3. It can provide all the ways to flip an image by 'opencv-python', which is stored in `CVFlipCodes`.
  4. It can provide all the corner point names used in this module, which is stored in `CornerPoints`.
  5. It can put text at any given coordinate on the image screen with a background, with the help of `PutText` function.

For further information about any module, please see for the docs provided in each of the modules.

Thank You 🙂
------------
"""


# Importing modules
from cProfile import label
import setup

import cv2 as cv
import time as tm
import numpy as np
from math import hypot as hpt
import mediapipe.python.solutions.hands as mphands
import mediapipe.python.solutions.drawing_utils as mpdraw


# Finger names
class Fingers ():
    """Fingers names."""
    THUMB = 'Thumb'
    INDEX_FINGER = 'Index'
    MIDDLE_FINGER = 'Middle'
    RING_FINGER = 'Ring'
    PINKY = 'Pinky'


# Hand landmarks
class HandLandmark ():
    """The 21 hand landmarks."""
    WRIST = 0

    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_DIP = 3
    THUMB_TIP = 4

    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8

    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12

    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    MCPs = [1, 5, 9, 13, 17]
    PIPs = [2, 6, 10, 14, 18]
    DIPs = [3, 7, 11, 15, 19]
    TIPs = [4, 8, 12, 16, 20]


# Ways to flip an image
class CVFlipCodes ():
    """The 3 modes to flip the image."""
    flip_vertically = 0
    flip_horizontally = 1
    flip_vertically_and_horizontally = -1


# Corner point names
class CornerPoints ():
    """The 4 corners."""
    Top_Left_Corner = 'xy'
    Top_Right_Corner = 'Xy'
    Bottom_Left_Corner = 'xY'
    Bottom_Right_Corner = 'XY'


# Hand Detector Module
class HandDetector ():
    def __init__(self, mode:bool = False, MaxHands: int = 2, complexity: int = 1, detectconf: float = 0.5, trackconf: float = 0.5, linethikness: int = 2, filled = cv.FILLED, radius: int = 5) -> None:
        """
        Initialises the hand detector.
        """

        self.mode = mode
        self.MaxHands = MaxHands
        self.complexity = complexity
        self.detectconf = detectconf
        self.trackconf = trackconf
        self.linethikness = linethikness
        self.filled = filled
        self.radius = radius

        self.hands = mphands.Hands (self.mode, self.MaxHands, self.complexity, self.detectconf, self.trackconf)

    # Find the hand
    def FindHands (self, image: np.ndarray, draw: bool = True) -> tuple [np.ndarray, bool]:
        """
        Detects the hand.
        """

        imgRGB = cv.cvtColor (image, cv.COLOR_BGR2RGB)
        self.results = self.hands.process (imgRGB)
        inf = False

        if self.results.multi_hand_landmarks != None:
            inf = True

            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    mpdraw.draw_landmarks (image, handlms, mphands.HAND_CONNECTIONS, col = (52, 255, 48))

        return image, inf

    # Find location of hand landmarks
    def FindLocation (self, image: np.ndarray, Hands = "one", hbox: bool = True, col: tuple [int, int, int] = (52, 255, 48)) -> tuple [dict [int, tuple [int, int]], dict [str, tuple [int, int]]]:
        """
        Finds the location of the hand and the hand landmarks.
        """

        self.lmloc = {}
        handloc = {}

        xloc = []
        yloc = []

        # Creating landmark: coordinate dictionary
        if self.results.multi_hand_landmarks != None:
            if Hands.lower () == "one":
                hand = self.results.multi_hand_landmarks [0]

            if Hands.lower () == "both":
                hand = self.results.multi_hand_landmarks

            for id, lm in enumerate (hand.landmark):
                h, w, c = image.shape
                cx, cy = int (lm.x * w), int (lm.y * h)

                self.lmloc [id] = (cx, cy)

        # finding the hand and packing it in a box
        if len (self.lmloc) >= 1:
            for i in range (21):
                xloc.append (self.lmloc [i][0])
                yloc.append (self.lmloc [i][1])

            box_R_x = min (xloc)
            box_R_y = min (yloc)
            handloc [CornerPoints.Top_Left_Corner] = (box_R_x, box_R_y)

            box_L_X = max (xloc)
            box_L_y = min (yloc)
            handloc [CornerPoints.Top_Right_Corner] = (box_L_X, box_L_y)

            box_r_x = min (xloc)
            box_r_Y = max (yloc)
            handloc [CornerPoints.Bottom_Left_Corner] = (box_r_x, box_r_Y)

            box_l_X = max (xloc)
            box_l_Y = max (yloc)
            handloc [CornerPoints.Bottom_Right_Corner] = (box_l_X, box_l_Y)

        # Making the box
        if hbox:
            cv.rectangle (image, (handloc [CornerPoints.Top_Left_Corner][0], handloc [CornerPoints.Top_Left_Corner][1]), (handloc [CornerPoints.Bottom_Right_Corner][0], handloc [CornerPoints.Bottom_Right_Corner][1]), col, self.linethikness)

        return self.lmloc, handloc

    # Draw hand landmarks
    def DrawLandmarks (self, image: np.ndarray, index: int, loclist: dict [int, tuple [int, int]] = False, prnt: bool = False, color: tuple [int, int, int] = (255, 0, 255)) -> None:
        if not loclist:
            loclist = self.lmloc
            
        cv.circle (image, (loclist [index][0], loclist [index][1]), self.radius, color, self.filled)

        if not index:
            for i in len (loclist):
                cv.circle (image, (loclist [i][0], loclist [i][1]), self.radius, color, self.filled)

        if prnt:
            print (f"{index}: {loclist [index]}")

    # Detect if any finger is up !
    def fingersUp (self, image: np.ndarray, loclist: dict [int, tuple [int, int]] = False, Draw: bool = False) -> dict [str, bool]:
        """
        Working
        -------
        It takes the distance between the `TIP` [4, 8, 12, 16, 20] of a finger and the `MCP` [1, 5, 9, 13, 17] point. Then after a certain range it detects it as the `finger is up`.

        Returns
        -------
        It returns the information in the format of a dict consist of five `str` (finger names) and `bool` values.
        """

        inf = {Fingers.THUMB: False, Fingers.INDEX_FINGER: False, Fingers.MIDDLE_FINGER: False, Fingers.RING_FINGER: False, Fingers.PINKY: False}

        if not loclist:
            loclist = self.lmloc

        # Thumb
        image, dist = self.findDistance (image, loclist, HandLandmark.THUMB_TIP, HandLandmark.THUMB_MCP, Draw)
        dist = int (dist)

        if dist >= 69:
            inf [Fingers.THUMB] = True

        else:
            inf [Fingers.THUMB] = False

        # Index finger
        image, dist = self.findDistance (image, loclist, HandLandmark.INDEX_FINGER_TIP, HandLandmark.INDEX_FINGER_MCP, Draw)
        dist = int (dist)

        if dist >= 100:
            inf [Fingers.INDEX_FINGER] = True

        else:
            inf [Fingers.INDEX_FINGER] = False

        # Middle finger
        image, dist = self.findDistance (image, loclist, HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.MIDDLE_FINGER_MCP, Draw)
        dist = int (dist)

        if dist >= 110:
            inf [Fingers.MIDDLE_FINGER] = True

        else:
            inf [Fingers.MIDDLE_FINGER] = False

        # Ring finger
        image, dist = self.findDistance (image, loclist, HandLandmark.RING_FINGER_TIP, HandLandmark.RING_FINGER_MCP, Draw)
        dist = int (dist)

        if dist >= 100:
            inf [Fingers.RING_FINGER] = True

        else:
            inf [Fingers.RING_FINGER] = False

        # Pinky
        image, dist = self.findDistance (image, loclist, HandLandmark.PINKY_TIP, HandLandmark.PINKY_MCP, Draw)
        dist = int (dist)

        if dist >= 70:
            inf [Fingers.PINKY] = True

        else:
            inf [Fingers.PINKY] = False

        return inf

    # Detect if any finger is up !
    def fingersUp_PROTO (self, loclist: dict [int, tuple [int, int]] = False) -> dict [str, bool]:
        """
        Working
        -------
        It takes the location of the `TIP` [8, 12, 16, 20] of a finger and the `DIP` [7, 11, 15, 19]. Then after a certain range it detects it as the `finger is up`. For the thumb, it takes the location of the `DIP` [3] and the `MCP` [2].

        Returns
        -------
        It returns the information in the format of a dict consist of five `str` (finger names) and `bool` values.
        """

        fingers = {Fingers.THUMB: False, Fingers.INDEX_FINGER: False, Fingers.MIDDLE_FINGER: False, Fingers.RING_FINGER: False, Fingers.PINKY: False}

        if not loclist:
            loclist = self.lmloc

        # Thumb
        if loclist [HandLandmark.THUMB_DIP][0] > loclist [HandLandmark.THUMB_DIP - 1][0]:
            fingers [Fingers.THUMB] = True

        else:
            fingers [Fingers.THUMB] = False

        # Index finger
        if loclist [HandLandmark.INDEX_FINGER_TIP][1] < loclist [HandLandmark.INDEX_FINGER_TIP - 1][1]:
            fingers [Fingers.INDEX_FINGER] = True

        else:
            fingers [Fingers.INDEX_FINGER] = False

        # Middle finger
        if loclist [HandLandmark.MIDDLE_FINGER_TIP][1] < loclist [HandLandmark.MIDDLE_FINGER_TIP - 1][1]:
            fingers [Fingers.MIDDLE_FINGER] = True

        else:
            fingers [Fingers.MIDDLE_FINGER] = False

        # Ring finger
        if loclist [HandLandmark.RING_FINGER_TIP][1] < loclist [HandLandmark.RING_FINGER_TIP - 1][1]:
            fingers [Fingers.RING_FINGER] = True

        else:
            fingers [Fingers.RING_FINGER] = False

        # Pinky
        if loclist [HandLandmark.PINKY_TIP][1] < loclist [HandLandmark.PINKY_TIP - 1][1]:
            fingers [Fingers.PINKY] = True

        else:
            fingers [Fingers.PINKY] = False
 
        return fingers

    # Finds distance
    def findDistance (self, image: np.ndarray, loclist: dict [int, tuple [int, int]], id1: int, id2: int, draw: bool = True, col1: tuple [int, int, int] = (255, 0, 255)) -> tuple [np.ndarray, float]:
        """
        Working
        -------
        It takes the locations of two points and finds the ditance between them by the hypot function of the math module.

        Returns
        -------
        Image with three circles drawn on it, and the length between those two points.
        """

        point1 = (loclist [id1][0], loclist [id1][1])
        point3 = (loclist [id2][0], loclist [id2][1])

        point2 = ((point1 [0] + point3 [0])//2, (point1 [1] + point3 [1])//2)

        length = hpt (point1 [0] - point3 [0], point1 [1] - point3 [1])

        # To draw
        if draw:
            col2 = (255 - col1 [0], 255 - col1 [1], 255 - col1 [2])

            cv.line (image, point1, point3, col1, self.linethikness)

            cv.circle (image, point1, self.radius, col1, self.filled)
            cv.circle (image, point2, self.radius, col2, self.filled)
            cv.circle (image, point3, self.radius, col1, self.filled)
 
        return image, length

    # Safe working zone
    def FindingZone (self, image: np.ndarray, space: int = 100, camdim: tuple [int, int] = (640, 480), color: tuple [int, int, int] = (0, 0, 255)) -> dict [str, tuple [int, int]]:
        FindingZonedim = {CornerPoints.Top_Left_Corner: (space, space), 
                          CornerPoints.Top_Right_Corner: (camdim [0] - space, space), 
                          CornerPoints.Bottom_Left_Corner: (space, camdim [1] - space), 
                          CornerPoints.Bottom_Right_Corner: (camdim [0] - space, camdim [1] - space)}

        cv.rectangle (image, FindingZonedim [CornerPoints.Top_Left_Corner], FindingZonedim [CornerPoints.Bottom_Right_Corner], color, self.linethikness)

        return FindingZonedim


# Put text with a background
def PutText (image: np.ndarray, txt: str, loc: tuple [int, int], font = cv.FONT_HERSHEY_COMPLEX, fontscale: int = 1, FontColor: tuple [int, int, int] = (255, 255, 255), BGColor: tuple [int, int, int] = (0, 0, 0), BorderColor: tuple [int, int, int] = (0, 0, 255), FontThikness: int = 2, BorderThikness: int = 2, BoardWidth: int = 200, BoardHeight: int = 55) -> None:
    # Rectangle dimentions
    rectdim = {CornerPoints.Top_Left_Corner: (loc [0], loc [1]), 
               CornerPoints.Top_Right_Corner: (loc [0] + BoardWidth, loc [1]), 
               CornerPoints.Bottom_Left_Corner: (loc [0], loc [1] + BoardHeight), 
               CornerPoints.Bottom_Right_Corner: (loc [0] + BoardWidth, loc [1] + BoardHeight)}

    # Rectangle
    cv.rectangle (image, (rectdim [CornerPoints.Top_Left_Corner][0], rectdim [CornerPoints.Top_Left_Corner][1]), (rectdim [CornerPoints.Bottom_Right_Corner][0], rectdim [CornerPoints.Bottom_Right_Corner][1]), BGColor, cv.FILLED)

    # Border
    cv.rectangle (image, (rectdim [CornerPoints.Top_Left_Corner][0], rectdim [CornerPoints.Top_Left_Corner][1]), (rectdim [CornerPoints.Bottom_Right_Corner][0], rectdim [CornerPoints.Bottom_Right_Corner][1]), BorderColor, BorderThikness)

    # Text
    cv.putText (image, txt, (loc [0] + 12, loc [1] + 37), font, fontscale, FontColor, FontThikness)


# Draw volume colomn
def DrawVolColomn (image: np.ndarray, cord: tuple [int, int], volume_level: int, gap: int = False, width: int = 50, camdim: tuple [int, int] = (640, 480), thikness: int = 2, bdcol: tuple [int, int, int] = (0, 0, 0), bgcol: tuple [int, int, int] = (255, 255, 255), volume_color_nrml: tuple [int, int, int] = (0, 255, 0), volume_color_abnrml: tuple [int, int, int] = (0, 136, 255)) -> int:
    # Volume bar dimentions
    VolColDim = {CornerPoints.Top_Left_Corner: cord,
                 CornerPoints.Top_Right_Corner: (cord [0] + width, cord [1]),
                 CornerPoints.Bottom_Left_Corner: (cord [0], camdim [1] - gap),
                 CornerPoints.Bottom_Right_Corner: (cord [0] + width, camdim [1] - gap)}

    if not gap:
        gap = cord [1]

    if volume_level <= 100:
        volume_color = volume_color_nrml
    
    if volume_level > 100 and volume_level <= 150:
        volume_color = volume_color_abnrml

    volume_level_cord = (cord [0], np.interp (volume_level, (0, 150), (VolColDim [CornerPoints.Top_Left_Corner][1], VolColDim [CornerPoints.Bottom_Left_Corner][1])))
    LabelCord = (cord [0], VolColDim [CornerPoints.Top_Left_Corner][1] - VolColDim [CornerPoints.Bottom_Left_Corner][1])

    # Volume bar
    cv.rectangle (image, VolColDim [CornerPoints.Top_Left_Corner], VolColDim [CornerPoints.Bottom_Right_Corner], bgcol, cv.FILLED)

    # volume level
    cv.rectangle (image, VolColDim [CornerPoints.Bottom_Right_Corner], volume_level_cord, volume_color, cv.FILLED)

    # Volume bar border
    cv.rectangle (image, VolColDim [CornerPoints.Top_Left_Corner], VolColDim [CornerPoints.Bottom_Right_Corner], bdcol, thikness)

    # Volume percentage
    cv.putText (image, f"{volume_level}%", LabelCord, cv.FONT_HERSHEY_COMPLEX, 1, bgcol, 2)

    return volume_level


# main
def main ():
    ptime = 0
    cam = cv.VideoCapture (0)
    detector = HandDetector ()

    while True:
        try:
            success, img = cam.read ()
            img = detector.FindHands (img)
            loc, hbox = detector.FindLocation (img)

            if len (loc) != 0:
                detector.DrawLandmarks (img)

            ctime = tm.time ()
            fps = 1/(ctime - ptime)
            ptime = ctime

            PutText (img, f"FPS: {round (fps, 2)}", loc = (20, 40))

            cv.imshow ("Webcam", img)
            cv.waitKey (1)

        except KeyboardInterrupt:
            print ('')
            exit ()


if __name__ == "__main__":
    main ()


# THE END
