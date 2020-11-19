import cv2
import pyautogui

# Get width and height of the screen
width, height = pyautogui.size()
# The coordenate to click to play and pause video
target_x = 0.15 * width
target_y = 0.42 * height

# Load cascades
# Cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('./Cascades/haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eyes_cascade = cv2.CascadeClassifier('./Cascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

# True if eyes are detected
eyes_detected = False
changed = False

while True:

    # Read video camara
    ret, img = cap.read()
    # Convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate through all detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Get the area where the face is located
        area_gray = gray[y:y + h, x:x + w]  # Gray image
        area_color = img[y:y + h, x:x + w]  # Color image

        # Detect eyes within a face area
        eyes = eyes_cascade.detectMultiScale(area_gray)

        # Iterate through all the eyes
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eyes
            cv2.rectangle(area_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Check if any faces were detected
    if len(faces) > 0:
        # If two or more eyes were detected
        if len(eyes) >= 2:
            if eyes_detected == False: changed = True
            eyes_detected = True
    else:
        if eyes_detected == True: changed = True
        eyes_detected = False

    # Show the image
    cv2.imshow('Eyes detector', img)

    # Click screen when detection state is changed
    if changed:
        changed = False
        # Click on screen
        pyautogui.click(target_x, target_y)

    # Detect clicked keyboard
    k = cv2.waitKey(30) & 0xff
    # Stop the program when escape key is clicked
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()