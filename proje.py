from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Veri yolu tanımları
train_path = "C:/Users/ahmet/dataset/train"
validation_path = "C:/Users/ahmet/dataset/validation"
test_path = "C:/Users/ahmet/dataset/test"

# Görüntü boyutları
img_size = (48, 48)
camera_img_size = (224, 224)  # Kamera görüntüsünün boyutu
batch_size = 64

# Veri artırma (data augmentation) için ImageDataGenerator kullanma
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Veri yükleyici oluşturma
train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=img_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_path,
                                                              target_size=img_size,
                                                              batch_size=batch_size,
                                                              class_mode='categorical')

# Model oluşturma
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(6, activation='softmax'))  # 7 sınıf için çıkış katmanı

# Modeli derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model özeti
model.summary()

# Modeli eğitme
egitimTakip = model.fit(train_generator, epochs=20, validation_data=validation_generator)

# Modeli test etme
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_path,
                                                  target_size=img_size,
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

model.save('DUYGU_DURUMU_TANIMA.h5')


#***************************************************************************************

import matplotlib.pyplot as plt

plt.plot(egitimTakip.history['accuracy'])
plt.plot(egitimTakip.history['val_accuracy'])
plt.title("Model Başarımları")
plt.ylabel("Başarım")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim","Doğrulama"], loc = "upper left")
plt.show()

plt.plot(egitimTakip.history["loss"])
plt.plot(egitimTakip.history["val_loss"])
plt.title("Model Kayıpları")
plt.ylabel("Kayıp")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim","Doğrulama"], loc = "upper left")
plt.show()