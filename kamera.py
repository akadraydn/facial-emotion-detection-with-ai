import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Eğitilmiş modelin yolu
model_path = 'DUYGU_DURUMU_TANIMA.h5'
emotion_model = load_model(model_path)

# Duygu etiketleri
emotion_labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Yüz tespiti için OpenCV yüz tanıma modelini yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kamera başlatma
cap = cv2.VideoCapture(0)

# Pencereyi oluştur ve boyutunu ayarla (resizable)
cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)

while True:
    # Kameradan görüntü al
    ret, frame = cap.read()
    
    # Gri tonlamaya çevir
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Yüz tespiti
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Tespit edilen yüzleri çerçeve içine al
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_roi = frame[y:y+h, x:x+w]  # Tespit edilen yüz bölgesini al
    
        # Görüntüyü istediğiniz boyuta yeniden boyutlandır
        resized_frame = cv2.resize(face_roi, (48, 48))
        
        # Görüntüyü yeniden şekillendir, normalleştir ve preprocess_input fonksiyonunu kullan
        img_array = img_to_array(resized_frame)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Duygu tahmini
        predictions = emotion_model.predict(img_array)
        
        # En yüksek tahminin indeksi
        predicted_label_index = np.argmax(predictions)
        
        # Tahmin edilen duygu etiketi
        predicted_emotion = emotion_labels[predicted_label_index]
        
        # Görüntü üzerine sonucu yazdır
        cv2.putText(frame, f'Emotion: {predicted_emotion}', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    
    # Görüntüyü göster
    cv2.imshow('Emotion Detection', frame)
    
    # Pencere boyutunu ayarla (resizable)
    cv2.resizeWindow('Emotion Detection', 800, 600)  # Ayarladığınız boyutu kullanabilirsiniz
    
    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera kapatma
cap.release()
cv2.destroyAllWindows()