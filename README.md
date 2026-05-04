
## ⚙️ Kurulum ve Kullanım

Bir bilgisayar mühendisliği projesi standartlarında, kurulumu şu adımlarla gerçekleştirebilirsin:

1.  **Depoyu Klonla:**
```bash
git clone [https://github.com/TunahanYlcnn/Yolov8-Kalman.git](https://github.com/TunahanYlcnn/Yolov8-Kalman.git)
cd Yolov8-Kalman
```
---

# 🛰️ YOLOv8-Kalman: Gelişmiş Nesne Takibi ve Durum Tahmini

Bu proje, **YOLOv8** (You Only Look Once) nesne dedektörü ile **Kalman Filtresi**'nin (Kalman Filter) gücünü birleştirerek, videolardaki nesneler için daha kararlı ve kesintisiz bir takip sistemi sunar. Sadece anlık tespitle yetinmeyip, nesnenin bir sonraki konumunu matematiksel olarak tahmin ederek gürültülü verileri temizler ve kısa süreli kayıpları (occlusion) telafi eder.

---

## 🧠 Çalışma Mantığı ve Matematiksel Arka Plan

Sistem, **"Recursive Bayesian Estimation"** prensibiyle çalışır. Her karede şu iki aşamalı döngü tekrarlanır:

### 1. Tahmin (Predict)
Nesnenin mevcut hızını ve konumunu kullanarak, bir sonraki karede nerede olacağını tahmin eder. Bu aşamada sistemin durum vektörü şu şekilde güncellenir:

$$\hat{x}_{k|k-1} = F_k \hat{x}_{k-1|k-1} + B_k u_k$$

Burada $F_k$ durum geçiş matrisini, $\hat{x}$ ise nesnenin koordinatlarını ($x, y, w, h$) ve hız bileşenlerini temsil eder.

### 2. Güncelleme (Update / Correct)
YOLOv8'den gelen yeni ölçüm (measurement) verisi ile tahmin edilen veri karşılaştırılır. **Kalman Kazancı (Kalman Gain)** hesaplanarak en doğru konum belirlenir:

$$K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R_k)^{-1}$$

Bu sayede, dedektör nesneyi anlık olarak kaçırsa bile Kalman Filtresi nesnenin hareket vektörünü koruyarak takibi sürdürür.

---

## 🛠️ Teknik Özellikler

*   **YOLOv8 Entegrasyonu:** Gerçek zamanlı, yüksek doğruluklu nesne tespiti.
*   **Doğrusal Tahminleme:** Nesne hareketlerini normalize ederek sarsıntıları (jitter) azaltır.
*   **Kaybolma Toleransı:** Nesne bir engelin arkasına girdiğinde, filtrenin tahmin yeteneği sayesinde takip kutusu nesneyi beklemeye devam eder.
*   **Hız Vektörü Analizi:** Nesnenin sadece konumu değil, yönelim ve hızı da takip edilir.

---

## 📋 Gereksinimler

Sistemi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekir:
```bash
pip install ultralytics opencv-python numpy filterpy
```
