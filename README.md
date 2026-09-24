# 🪐 NASA Exoplanet Taxonomy Explorer

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.14%2B-3F4F75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Aplikasi eksplorasi data interaktif dan klasterisasi *unsupervised machine learning* (**K-Means Clustering**) untuk mengelompokkan dan memetakan taksonomi planet di luar Tata Surya (*exoplanet*) berdasarkan dataset **NASA Exoplanet Archive**.

---

## 📌 Deskripsi Proyek

Proyek ini bertujuan untuk mengklasifikasikan ribuan planet luar surya secara otomatis menggunakan parameter fisis & orbital utama:
- **Orbital Period (Hari)**
- **Orbit Semi-Major Axis (AU)**
- **Massa Planet (Massa Bumi / $M_\oplus$)**

Dengan menerapkan **transformasi logaritmik**, **Feature Scaling (StandardScaler)**, dan **K-Means Clustering ($k=3$)**, model berhasil mengidentifikasi 3 kategori utama taksonomi astrofisika:

1. 🔥 **Hot Jupiters**: Planet gas raksasa bermassa tinggi yang mengorbit sangat dekat dengan bintang induknya (periode orbit singkat ~3.98 hari).
2. ❄️ **Cold Gas Giants**: Planet raksasa masif di wilayah orbit luar yang dingin (mirip Jupiter & Saturnus).
3. 🌍 **Terrestrial & Super-Earths**: Kelompok planet kebumian bermassa rendah hingga sedang yang menjadi target utama pencarian zona laik huni (*Habitable Zone*).

---

## ✨ Fitur Utama

- 🌌 **Dashboard Interaktif Streamlit**: Antarmuka modern ber-tema ruang angkasa (*Modern Dark Space Theme*).
- 🚀 **Ruang Fitur 3D Interaktif**: Plot 3 Dimensi Plotly untuk memetakan distribusi 3 variabel fisis utama secara simultan.
- 📊 **Scatter Plot 2D & Pemfilteran**: Visualisasi interaktif Massa vs Periode Orbit dengan filter taksonomi dan tahun penemuan.
- 🧪 **Prediktor Kategori Planet Baru**: Form interaktif untuk memasukkan data planet baru dan memprediksi klasternya secara *real-time*.
- 📈 **Evaluasi Klaster Otomatis**: Menampilkan metrik validasi klaster seperti **Silhouette Score**, **Davies-Bouldin Index**, dan **Calinski-Harabasz Index**.

---

## 📁 Struktur Direktori

```text
NASA Exoplanet Archive/
├── Exoplanet_Clustering_KMeans.ipynb  # Notebook eksplorasi & eksperimen model ML
├── all_exoplanets_2021.csv            # Dataset NASA Exoplanet Archive (2021)
├── streamlit.py                       # Kode utama aplikasi web Streamlit
├── requirements.txt                   # Daftar dependensi Python
└── README.md                          # Dokumentasi proyek
```

---

## 🛠️ Panduan Instalasi & Penggunaan

### 1. Prasyarat
Pastikan Anda telah menginstal **Python 3.9** atau versi yang lebih baru di sistem Anda.

### 2. Kloning Repositori
```bash
git clone https://github.com/username-anda/nasa-exoplanet-taxonomy-explorer.git
cd nasa-exoplanet-taxonomy-explorer
```

### 3. Buat & Aktifkan Virtual Environment (Opsional tapi Direkomendasikan)
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Instal Dependensi
```bash
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi Streamlit
```bash
streamlit run streamlit.py
```
Aplikasi secara otomatis akan terbuka di peramban web Anda pada alamat `http://localhost:8501`.

---

## 📊 Metrik Evaluasi Model

| Metrik Validasi | Keterangan |
| :--- | :--- |
| **Silhouette Score** | Menunjukkan pemisahan klaster yang cukup jelas & terdefinisi baik |
| **Davies-Bouldin Index** | Skor lebih rendah menandakan tingkat separasi antar klaster yang baik |
| **Calinski-Harabasz Index** | Menunjukkan variansi internal klaster yang kompak dibanding variansi antar klaster |

---

## 🛠️ Teknologi yang Digunakan

- **Bahasa Pemrograman**: Python
- **Machine Learning**: `scikit-learn` (K-Means, StandardScaler, Metrics)
- **Pengolahan Data**: `pandas`, `numpy`
- **Visualisasi Data**: `plotly` (Express & Graph Objects), `matplotlib`, `seaborn`
- **Framework Web**: `streamlit`

---

## 📜 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---
*Dikembangkan untuk eksplorasi data astronomi & machine learning.*
