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

Hand Landmarks
--------------
![Hand Landmarks](https://user-images.githubusercontent.com/78896721/151704786-dce200a9-30f0-4b12-ae59-a061e60a833a.jpg)

Acknowledgement
---------------
For further information about any module, please see for the docs provided in each of the modules.

Thank You 🙂
------------
