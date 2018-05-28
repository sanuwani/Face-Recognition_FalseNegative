import cv2

#Setup the enviorment by linking to the Haar Cascades Models

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Identify the image of interest to import. Ensure that when you import a file path
#that you do not use / in front otherwise it will return empty.
img = cv2.imread('jj.jpg')

# Resize the image to save space and be more manageable.
# We do this by calculating the ratio of the new image to the old image
r = 500.0 / img.shape[1]
dim = (500, int(img.shape[0] * r))

# Perform the resizing and show
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

#Display the image
#cv2.imshow('image',resized)
cv2.waitKey(0) #Before moving on, wait for a keyboard click.


#Process the image - convert to BRG to grey
grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

#cv2.imshow('image',grey)
cv2.waitKey(0) #Before moving on, wait for a keyboard click.

#Identify the face and eye using the haar-based classifiers.
faces = face_cascade.detectMultiScale(grey, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(resized,(x,y),(x+w,y+h),(255,0,0),2)
    roi_grey = grey[y:y+h, x:x+w]
    roi_color = resized[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_grey)
for (ex,ey,ew,eh) in eyes:
    print ex,ey,ew,eh
    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

#Display the bounding box for the face and eyes
cv2.imshow('img',resized)
cv2.waitKey(0)
