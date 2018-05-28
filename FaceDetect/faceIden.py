import cv2
import math

name=raw_input('Enter the id of image: ')
centers = []
nose=[]
def facechop(image):  
    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)

    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        face_file_name = name+'.jpg'
        cv2.imwrite(face_file_name, sub_face)

    #cv2.imshow(image, img)

    return

def detectEyes(name):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    image = cv2.imread(name+'.jpg')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # iterating over the face detected
    for (x, y, w, h) in faces:
        print 'inside loop1'
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

    # Store the coordinates of eyes in the image to the 'center' array
        for (ex, ey, ew, eh) in eyes:
            print 'inside loop2'
            print ex,ey,ew,eh
            centers.append(x + int(ex + 0.5 * ew))
            centers.append(y + int(ey + 0.5 * eh))
            
    print centers

def detectNose(name):
    nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
    image = cv2.imread(name+'.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in nose_rects:
        nose.append(int(x + 0.5 * w))
        nose.append(int(y + 0.5 * h))

    print nose

def getImage(name):
        cap = cv2.VideoCapture(0)


        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        while(True):
                
                ret, frame = cap.read()

                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image
                faces = faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                        
                )

                

                
                for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


                
                cv2.imshow('frame', frame)
                #cv2.imwrite('flf.jpg',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.imwrite(name+'.jpg',frame)
                        break
        cap.release()        
        cv2.destroyAllWindows()
        
def recFace():
    if (len(centers)==4):
        #cen=centers[0]
        eye_x_len=abs(centers[0]-centers[2])
        eye_y_len=abs(centers[1]-centers[3])
        val=(eye_x_len**2)+(eye_y_len**2)
        val=math.sqrt(val)
        if (len(nose)==2):
            x=eye_x_len/2.0
            y=eye_y_len/2.0
            dis_to_nose_x=abs(x-nose[0])
            dis_to_nose_y=abs(x-nose[1])
            d=(dis_to_nose_x**2)+(dis_to_nose_y**2)
            d=math.sqrt(d)
            print val,d
            ratio=d/val
            print ratio
            f= open("data.txt","a+")
            f.write(name + "," + str(ratio)+ "\n")
            f.close()
    else:
        print 'Error; Try again!!!'


print("Press'q' to caapture the image")
getImage(name)

detectEyes(name)
detectNose(name)
recFace()
facechop(name+'.jpg')
