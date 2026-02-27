# 🤖 Makine Öğrenmesi Algoritmaları Koleksiyonu

Bu depo, makine öğrenmesinin temel taşlarını oluşturan çeşitli algoritmaların Python ve Scikit-Learn kullanılarak gerçekleştirilmiş uygulamalarını içerir. Proje; sınıflandırma, regresyon, kümeleme ve veri görselleştirme tekniklerini kapsamaktadır.

## 🛠️ Kullanılan Teknolojiler
* **Python 3.x**
* **Scikit-Learn:** Algoritma modelleri ve veri işleme.
* **Pandas & NumPy:** Veri manüpilasyonu ve matris işlemleri.
* **Matplotlib & Seaborn:** Veri görselleştirme.

## 📁 Proje İçeriği ve Algoritmalar

### 1. Denetimli Öğrenme (Sınıflandırma & Regresyon)
* **Karar Ağaçları (`decision_tree_...`):** Karmaşıklık analizi ve budama teknikleri ile modelleme.
* **Lojistik Regresyon (`logistic_regression_...`):** İkili sınıflandırma problemleri üzerine uygulamalar.
* **Elastic Net (`elastic_net_diabetes.py`):** L1 ve L2 regülarizasyonu ile diyabet verisi tahmini.
* **Naïve Bayes (`iris_naive_bayes_siniflandirma.py`):** Iris veri seti üzerinde olasılıksal sınıflandırma.
* **KNN & SVM:** `knn_dt_svm_hyperparam_search.py` ile hiperparametre optimizasyonu.

### 2. Denetimsiz Öğrenme (Kümeleme)
* **K-Means (`kmeans_kumeleme.py`):** Veri noktalarını benzerliklerine göre gruplandırma.
* **DBSCAN (`dbscan_circles.py`):** Yoğunluk tabanlı kümeleme ile iç içe geçmiş halka verileri ayrıştırma.
* **Hiyerarşik Kümeleme (`hiyerarsik_kumeleme.py`):** Dendrogram yapısı ile küme analizi.

### 3. Boyut İndirgeme ve Görselleştirme
* **PCA (`iris_pca_2d_3d.py`):** Yüksek boyutlu verilerin 2B ve 3B uzayda incelenmesi.
* **t-SNE (`mnist_tsne_visualization.py`):** Karmaşık veri setlerinin (MNIST vb.) düşük boyutlu görselleştirilmesi.

## 🚀 Kurulum ve Çalıştırma

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install numpy pandas matplotlib scikit-learn seaborn