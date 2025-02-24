import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import f1_score

# นิยาม custom_f1score function (หรือใช้ฟังก์ชันจาก sklearn)
def custom_f1score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

# โหลดโมเดลพร้อมกับ custom_objects ที่ต้องการ
model = tf.keras.models.load_model(
    'C:/Users/pog12/Downloads/elysian01/car-number-plate-detection/model_detection.h5',
    custom_objects={'custom_f1score': custom_f1score}  # กำหนด custom metric ที่ใช้
)

# ฟังก์ชันเพื่อแปลงภาพจากตัวอักษรเป็นตัวเลข
def fix_dimension(img): 
    new_img = np.zeros((28, 28, 3))
    for i in range(3):
        new_img[:, :, i] = img
    return new_img

# ฟังก์ชันในการทำนายแผ่นป้ายทะเบียน
def show_results(characters):
    dic = {}
    chars = '0123456789กขฃคฆงจฉชซฅฆฌญฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลวศษสหฬอฮ'
    for i, c in enumerate(chars):
        dic[i] = c

    output = []
    for ch in characters:
        img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)  # เตรียมภาพสำหรับโมเดล
        y_ = model.predict(img)[0]  # ทำนายตัวอักษร
        character = dic[np.argmax(y_)]
        output.append(character)  # เก็บผลลัพธ์ในลิสต์

    plate_number = ''.join(output)
    return plate_number

# ฟังก์ชันในการตรวจจับป้ายทะเบียน
def detect_plate(img, plate_cascade):
    plate_img = img.copy()
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=7)
    for (x, y, w, h) in plate_rect:
        plate = plate_img[y:y+h, x:x+w]
        return plate, (x, y, w, h)
    return None, None

# ฟังก์ชันในการแยกตัวอักษรจากป้ายทะเบียน
def segment_characters(plate_image):
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_plate, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 20:  # ปรับขนาดตามความเหมาะสม
            char_image = thresh[y:y+h, x:x+w]
            char_images.append(char_image)

    char_images = sorted(char_images, key=lambda x: x[0])
    return char_images

# ฟังก์ชันบันทึกภาพที่จับได้
def save_plate_image(plate_image, plate_number):
    # สร้างโฟลเดอร์สำหรับบันทึกภาพ (หากยังไม่มี)
    folder_path = "C:/Users/pog12/Downloads/elysian01/car-number-plate-detection/saved_plates"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # กำหนดชื่อไฟล์ตามหมายเลขทะเบียน
    file_name = f"{plate_number}.jpg"
    file_path = os.path.join(folder_path, file_name)

    # บันทึกภาพ
    cv2.imwrite(file_path, plate_image)
    print(f"Saved plate image: {file_path}")

# เริ่มต้นการเปิดกล้อง
cap = cv2.VideoCapture(0)  # ใช้กล้องที่ index 0
plate_cascade = cv2.CascadeClassifier('indian_license_plate.xml')  # เปลี่ยน path ให้ถูกต้อง

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # ตรวจจับป้ายทะเบียนจากภาพ
    plate, plate_coords = detect_plate(frame, plate_cascade)
    if plate is not None:
        # แสดงผลป้ายทะเบียนที่ตรวจจับได้
        cv2.imshow("Detected Plate", plate)

        # แยกตัวอักษรจากป้ายทะเบียน
        characters = segment_characters(plate)

        # ทำนายและแสดงผลตัวอักษรที่อ่านได้
        plate_number = show_results(characters)
        print("Plate Number: ", plate_number)

        # บันทึกภาพป้ายทะเบียน
        save_plate_image(plate, plate_number)

    # แสดงภาพจากกล้อง
    cv2.imshow('Camera Feed', frame)
    
    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่าง OpenCV
cap.release()
cv2.destroyAllWindows()
