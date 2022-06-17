# importing libraries
import numpy as np
import cv2
import os
import smtplib, ssl

# importing Open AI cascade for facial recognition
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

# one of the parameters of recognizing face
counter = 0

while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()

    # setting parameters 
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for face in faces:

        (x, y, w, h) = face
        # parameters for if a face has been detected
        if(face.any() and counter ==0):
            
            # code for sending email
            port = 465  # For SSL
            smtp_server = "smtp.gmail.com"
            sender_email = "EMAIL@gmail.com"  
            # Rogers email address for sending emails as texts
            receiver_email = "PHONENUMBER@pcs.rogers.com"  
            password = "EMAIL PASSWORD"
            message = "Face Detected"

            # Sends email
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, message)

        cv2.imshow("Face", frames)
        counter = 1 

        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('Video', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#releases video feed
video_capture.release()
cv2.destroyAllWindows()