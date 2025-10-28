import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# keras, bu veriyi train ve test tupleları olarak döndüğü için bu şekilde ayırdık.
# 60.000 train 10.000 test
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"X_train (eğitim resimleri) şekli: {X_train.shape}")
print(f"y_train (eğitim etiketleri) şekli: {y_train.shape}")
print(f"İlk resmin etiketi: {y_train[0]}")

#Şu anda X_train veri setimiz (60000, 28, 28) şeklinde. 60k tane 28x28 matris.
# Bizim Dense katmanımız 2D matris alamaz.
# input_shape: (784, ) şeklinde düz bir vektör bekler.
# reshape ile düz bir vektöre çevireceğiz.

X_train_processed = X_train.reshape(60000, 784)
X_test_processed = X_test.reshape(10000, 784)
print(f"Düzleştirme sonrası X_train şekli: {X_train_processed.shape}")

# resim pikseller, 0 ile 255 arasında değer alır.
# relu aktivasyonuna o kadar büyük sayı girmek optimizasyonu yavaşlatır.
# normalize etmemiz gerek. 255'e bölerek bunu yapabiliriz.

X_train_processed = X_train_processed.astype('float32')/255.0
X_test_processed = X_test_processed.astype('float32')/255.0
print(f"Normalizasyon sonrası bir pikselin max değeri: {X_train_processed.max()}")



# etiketleri kategorik hale getirme: one-hot-encoding yapıyoruz.
# modelimizin çıkış katmanı softmax olacak. 0-1 arası olasılıklar.
# categorical_crossentropy loss func bu olasılık listesini bir tamsayı ile karşılaştıramaz.
# bu yüzden etiketleri one hot vektörlere dönüştürmemiz gerek.
y_train_processed = to_categorical(y_train, 10)
y_test_processed = to_categorical(y_test, 10)
print(f"One-hot encoding sonrası ilk etiket (5): {y_train_processed[0]}")

model = Sequential([
    Dense(units=128, activation='relu', input_shape=(784,)),
    Dense(units=64, activation='relu'),
    Dense(units=10, activation='softmax')
])

# modelin mimarisini ve öğrenilecek parametre sayısınnı ağırlıklar + bias olarak gösterir.
model.summary()


# compile modeli öğrenmeye hazırlar.
# optimize adam gradient descent varyantı
# loss=categorical_crossentropy softmax çıkışı ve one-hot etiketler için matematiksel olarak doğu olan loss func

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Eğitimi Başlıyor...")

history= model.fit(
    X_train_processed, y_train_processed, epochs=10,
    batch_size=128,
    # her epoch sonunda modeli hiç görmediği test verisiyle sınıyoruz.
    validation_data=(X_test_processed,y_test_processed)
)

print("Model Eğitimi Tamamlandı.")
print("\nTest Veriseti Üzerinde Değerlendirme:")

# score [final_loss, final_Accuracy] şeklinde bir liste olarak döndürülür.
score = model.evaluate(X_test_processed,y_test_processed, verbose=0)
print(f"Test Kaybı (Loss): {score[0]:.4f}")
print(f"Test Başarısı (Accuracy): {score[1]:.4f}")

# history.history bir dictionary döndürür.
print(f"\nEğitim geçmişinde kaydedilenler: {history.history.keys()}")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1) # 1 satır 2 sütun 1.grafik
plt.plot(history.history['accuracy'], label='Eğitim Başarısı  (Training Acc)')
plt.plot(history.history['val_accuracy'], label='Doğrulama Başarısı (validation Accuracy)')
plt.title('Epoch\'lara Göre Başarı (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Eğitim Kaybı (Training Loss)')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı (Validation Loss)')
plt.title('Epoch\'lara Göre Kayıp (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()