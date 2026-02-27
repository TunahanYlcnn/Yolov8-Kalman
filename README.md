# 🛰️ Nesne Algılama ve Gelişmiş Takip Sistemleri (Object Tracking)

Bu depo, modern bilgisayar görüsü tekniklerini kullanarak nesne algılama (Object Detection) ve çoklu nesne takibi (Multi-Object Tracking) konularında geliştirilmiş kapsamlı Python projelerini içermektedir. Projeler, Windows 11 Pro ortamında ve NVIDIA GPU desteğiyle (Lenovo Gaming PC) optimize edilmiştir.


## 📁 Proje Modülleri ve Algoritmalar

### 1. YOLOv8 + Kalman Filtresi (SORT Yaklaşımı)
**Dosya:** `yolov8_kalman_sort.py`
* **Teknoloji:** YOLOv8 Algılama + Kalman Filtresi + Macar Algoritması (Linear Sum Assignment).
* **İşlev:** Her nesneye benzersiz bir ID atar. Kalman Filtresi sayesinde nesnenin hızını ve yönünü matematiksel olarak tahmin ederek akıcı bir takip sağlar.

### 2. Re3 Tracker (Derin Öğrenme & LSTM)
**Dosya:** `re3Algoritması.py`
* **Teknoloji:** Real-Time Recurrent Regression (Re3) mimarisi.
* **İşlev:** LSTM katmanları kullanarak nesneyi görsel hafızasıyla takip eder. Nesne kısa süreliğine engellerin arkasında kalsa dahi takibi sürdürme kabiliyetine sahiptir.

### 3. Karşılaştırmalı Analiz (Dual Display)
**Dosya:** `yolov8KalmanVsYolov8.py`
* **İşlev:** Standart YOLOv8 algılaması ile Kalman Filtresi entegre edilmiş sistemi aynı anda iki ayrı pencerede kıyaslar. Takip algoritmalarının kararlılık üzerindeki etkisini görselleştirir.


## 🛠️ Teknik Gereksinimler

Sisteminizde aşağıdaki kütüphanelerin yüklü olduğundan emin olun:

```bash
pip install ultralytics torch opencv-python numpy filterpy scipy