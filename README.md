# 🎯 BTK Akademi Datahon | Kaggle Playground Series - S3E16 Çözümü

Bu repoda, [Kaggle Playground Series - Season 3, Episode 16](https://www.kaggle.com/competitions/playground-series-s3e16/overview) yarışması kapsamında geliştirilen çözüm yer almaktadır. Proje, **BTK Akademi Datahon Yarışması** çerçevesinde yürütülmüştür.

## 🧠 Proje Amacı

Yarışmanın amacı, verilen yapay bir veri seti üzerinden en yüksek doğruluğu sağlayacak sınıflandırma modelini geliştirmektir. Katılımcılar, veri ön işleme, özellik mühendisliği, model seçimi ve hiperparametre optimizasyonu gibi süreçleri kullanarak en iyi performansı elde etmeye çalışmaktadır.

Harika! Görseldeki klasör ve dosya yapısına bakarak projenin oldukça yapılandırılmış ve modüler bir şekilde organize edildiğini söyleyebilirim. Aşağıda bu proje yapısını açıklıyorum:

---

## 📁 Proje Yapısı Açıklaması

### 📂 `catboost_info/`
CatBoost modelinin çalışması sırasında otomatik olarak oluşturulan ve eğitim süreçlerine dair bilgileri içeren sistem klasörü.

### 📂 `dataset/`
Veri dosyalarının veya veriye erişim için kullanılan yardımcı dosyaların yer aldığı klasör. Genelde `train.csv`, `test.csv`, `sample_submission.csv` gibi dosyalar burada bulunur.

### 📂 `predictions/`
Model çıktılarına ait tahmin dosyalarının (`submission.csv` gibi) veya farklı modellerin tahmin sonuçlarının kaydedildiği klasör.

---

### 🧠 Ana Python Script Dosyaları

- **`config.py`**  
  Proje boyunca kullanılan sabit yapılandırma ayarları (veri yolu, sabitler, parametreler vb.) bu dosyada tanımlanmış olabilir.

- **`data_preprocessing.py`**  
  Verinin temizlenmesi, dönüştürülmesi, eksik verilerin işlenmesi, kategorik/sayısal değişken dönüşümleri gibi ön işleme adımları burada gerçekleştirilir.

- **`dataset.py`**  
  Veri kümesinin yüklenmesi ve eğitim/test veri setlerinin hazırlanması gibi işlemler bu dosyada tanımlanmış olabilir.

- **`helpers.py`**  
  Küçük yardımcı fonksiyonların (örneğin: loglama, skor hesaplama, zaman ölçümü gibi) yer aldığı modül.

- **`HyperTuner.py`**  
  Model hiperparametre optimizasyon süreci (örneğin: GridSearchCV, Optuna, RandomSearch) burada ele alınır.

- **`models.py`**  
  Farklı regresyon modellerinin (CatBoost, LightGBM, XGBoost, vs.) tanımlandığı ve eğitildiği modül.

- **`main.py`**  
  Tüm pipeline’ın çalıştığı ana script. Genellikle veri yükleme, model eğitme, tahmin yapma ve sonuçları kaydetme burada birleştirilir.

- **`WeightedAssemble.py`**  
  Birden fazla modelin çıktılarının ağırlıklı ortalamasıyla daha iyi sonuç veren bir topluluk (ensemble) tahmini üreten dosya.

---

### 📝 Diğer Belgeler

- **`Notes.md`**  
  Proje boyunca alınan notlar, fikirler, denemeler veya model sonuçlarının not edildiği bir belge olabilir.

- **`README.md`**  
  Proje hakkında genel bilgi, kullanım yönergeleri ve açıklamaların yer aldığı açıklayıcı dosya.

---

Bu yapı, Kaggle gibi yarışma tabanlı projelerde model geliştirme, deneme ve çıktıları düzenli bir şekilde yönetmek için oldukça ideal.

İstersen bu yapıya özel bir `README.md` dosyası da hazırlayabilirim. Hazır mıyız?
## 🔍 Kullanılan Yöntemler

- Eksik veri analizi ve doldurma
- Etiketleme, one-hot encoding gibi kategorik değişken dönüşümleri
- Özellik mühendisliği (örnek: sayısal kombinasyonlar, etkileşimli özellikler)
- Model eğitimi:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- Hiperparametre optimizasyonu (GridSearchCV / Optuna)
- Model karşılaştırma ve değerlendirme (Confusion Matrix, Accuracy, ROC-AUC)
- En iyi model ile tahminlerin `submission.csv` dosyasına kaydı

## 🛠️ Kullanılan Teknolojiler

- Python 3.10+
- Pandas, NumPy, Scikit-Learn
- XGBoost, LightGBM
- Matplotlib, Seaborn
- Jupyter Notebook

## 📊 Sonuçlar

En iyi sonuç veren model, test verisi üzerinde %XX doğruluk oranına ulaşmıştır. (Not: Kaggle üzerinde "Public Leaderboard" skoru XX.X şeklindedir.)

## 📎 Yararlı Bağlantılar

- [Yarışma Sayfası (Kaggle)](https://www.kaggle.com/competitions/playground-series-s3e16)
- [BTK Akademi Resmi Sitesi](https://www.btkakademi.gov.tr/)

## 📌 Not

Bu proje, **BTK Akademi** tarafından düzenlenen **Yapay Zeka ve Veri Bilimi Yarışmaları (Datahon)** kapsamında gerçekleştirilmiştir. Tamamen eğitim ve geliştirme amaçlıdır.

