# Mendeteksi, Melacak, dan Menghitung Jumlah Kendaraan pada Rekaman Lalu Lintas Secara Real-Time

## Penjelasan Direktori

### Analysis

Berisi data-data hasil analisis dan prediksi dalam format json. Data-data analisis mencakup:
- Ground truth label
- Hasil prediksi

### Lib

Berisi _modified_ library dengan fungsionalitas dan sumber berikut:

    -lib/kito.py
        Original Author: Roman Solovyev (ZFTurbo)
        Link: https://github.com/ZFTurbo/Keras-inference-time-optimizer
        Kegunaan:
            Kito digunakan untuk mereduksi Conv2D dan Batchnorm menjadi 1 layer
    - lib/metrics.py
        Original Author: Timothy C. Arlen
        Link: https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734
        Kegunaan:
            Metrics digunakan untuk menghitung MAP hasil dari prediksi
    - lib/detector/YOLO/yad2k
        Original Author: Allan Zelener
        Link: https://github.com/allanzelener/YAD2K
        kegunaan:
            Yad2k digunakan untuk konversi YOLOv2 dari darknet kedalam keras
    - lib/detector/YOLO/train/preprocessing.py
        Original Author: Ngoc Anh Huynh
        Link: https://github.com/experiencor/keras-yolo2/blob/master/preprocessing.py
        kegunaan: 
            Preprocessing digunakan untuk membuat batch training data generator dan mengkonversi fitur
    - lib/detector/YOLO/train/trainer.py
        Original Author: Ngoc Anh Huynh
        Link: https://github.com/experiencor/keras-yolo2/blob/master/frontend.py
        kegunaan: 
            Trainer berisi helper function yang digunakan dalam proses training

### Notebook

Berisi kumpulan notebook untuk melakukan training, evaluation, dsb.

### Train Val Config

Berisi konfigurasi training data beserta labelnya dan referensi pada image terkait.

## Owner

Aldi Fahrezi

Muhammad

Muhammad Ayaz Dzulfikar
