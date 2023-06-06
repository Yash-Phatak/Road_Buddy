import pyrebase
import imutils
import easyocr
import cv2
import re
import joblib
import numpy as np
from ultralytics import YOLO

config = {
    "apiKey": "AIzaSyDtOjH2M0Ps0zyQxGjNXGnZb4BxdkNLYQY",
    "authDomain": "roadbuddy-bdaeb.firebaseapp.com",
    "databaseURL": "https://roadbuddy-bdaeb-default-rtdb.firebaseio.com",
    "projectId": "roadbuddy-bdaeb",
    "storageBucket": "roadbuddy-bdaeb.appspot.com",
    "messagingSenderId": "942672445949",
    "appId": "1:942672445949:web:55e9abddd5a2dc53139994",
    "measurementId": "G-22S937Z6HE",
    "serviceAccount": "configfirebase.json"
}

firebase = pyrebase.initialize_app(config)

storage = firebase.storage()

# all_files = storage.list_files()
# # print(all_files)
# for file in all_files:
#     print(file.name)

all_files = storage.list_files()
directory_pathCar = "car"
car_files = [file for file in all_files if file.name.startswith("car")]

all_files = storage.list_files()
directory_pathSign = "sign"
sign_files = [files for files in all_files if files.name.startswith("sign")]

# print(car_files)
# print(sign_files)

for fileCar in car_files:
    # print(fileCar.name)
    file_nameCar = fileCar.name[4:]
for fileSign in sign_files:
    # print(fileSign.name)
    file_nameSign = fileSign.name[5:]

# print(file_nameCar)
# print(fileCar.name)
# print(file_nameSign)
# print(fileSign.name)

pathCar = "Car/" + file_nameCar
pathSign = "Sign/" + file_nameSign

storage = firebase.storage()
storage.download(fileCar.name, pathCar)

storage = firebase.storage()
storage.download(fileSign.name, pathSign)

print("Files Downloaded")

#Database Realtime

db = firebase.database()

tokenId = file_nameCar[:-4]

#FILE DOWNLOADED



##########################################################################
# Number Detection

# image = cv2.imread(pathCar)
image = cv2.imread('5.jpeg')
#-------------------------------------------------------------------------------------------


image = imutils.resize(image, width=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cascade_file = 'C:/Users/jatin/Desktop/Hackstory/haarcascades/haarcascade_russian_plate_number.xml'
cascade = cv2.CascadeClassifier(cascade_file)
plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 25))

for (x, y, w, h) in plates:
    plate_region = image[y:y + h, x:x + w]

    gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # _, thresh = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_OTSU)

    reader = easyocr.Reader(['en'])
    result = reader.readtext(thresh)

    number = ""
    for i in result:
        number += i[1]

    print(number)

    pattern = r"[A-Z]{2}\d{2}[A-Z]{2}\d{4}"
    plate = re.search(pattern, number)

    if plate:
        license_plate = plate.group()
        print("License plate:", license_plate)

        db.child("tickets").child(tokenId).update({"Plate_number": license_plate})
        print("Updated number to RTDB")
    else:
        print("No license plate found.")

    #UPDATE TO RTDB


################################################################################
# No Parking detection

model = joblib.load('model_filename.pkl')

# test_image_path = pathSign
test_image_path = 'no_parking1.jpg'
#-------------------------------------------------------------------------------------------
test_image = cv2.imread(test_image_path)
test_image = cv2.resize(test_image, (100, 100))
test_image = np.array(test_image) / 255.0
test_image = test_image.reshape(1, -1)
test_prediction = model.predict(test_image)

parkingBoard = False

if test_prediction == 1:
    print("Parking Board Detected")
    parkingBoard = True
else:
    print("No Parking Board")
    parkingBoard = False

db.child("tickets").child(tokenId).update({"Parking_Board": parkingBoard})

print("Updated parking board to RTDB")

##################################################################################
#yolo car bike detection

model = YOLO('yolov8l.pt')

object_names = model.names

# results = model(pathCar, show=True)
results = model('5.jpeg', show=True)
#-------------------------------------------------------------------------------------------

for r in results:
    boxes = r.boxes
    for b in boxes:
        id = int(b.cls[0])
    if(id in [2, 3, 7]):
        vehicle = object_names[id]
        print("Vehicle: " + vehicle)

        db.child("tickets").child(tokenId).update({"Vehicle_Type": vehicle})
        print("Updated vehicle to RTDB")



####################################################################################################

all_values = db.get().val()
for value in all_values.values():
    dictionary = value[tokenId]
    # print(dictionary)
    car_lat = round(float(dictionary['car_latitude']), 3)
    car_long = round(float(dictionary['car_longitude']), 3)
    car_time = int(dictionary['car_timestamp'])

    sign_lat = round(float(dictionary['sign_latitude']), 3)
    sign_long = round(float(dictionary['sign_longitude']), 3)
    sign_time = int(dictionary['sign_timestamp'])

    plate_number = dictionary['Plate_number']
    park_board = dictionary['Parking_Board']

    # print(car_lat, car_long, car_time, sign_lat, sign_long, sign_time, plate_number, park_board)

    time_diff = (sign_time - car_time)/1000
    lat_diff = (sign_lat - car_lat)
    long_diff = (sign_long - sign_long)

    # print(time_diff, lat_diff, long_diff)

    if(time_diff <= 30.0):
        if(lat_diff <= 0.005 and long_diff <= 0.005):
            print("AI - Verified")
            db.child("tickets").child(tokenId).update({"AI_verified": "verified"})
            print("Updated AI Verified to RTDB")