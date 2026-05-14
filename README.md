# 🚗 Prediksi Harga Mobil — Final Project Sains Data

Aplikasi web prediksi harga mobil menggunakan **Linear Regression** berbasis dataset Car Sales.

## File yang diperlukan
```
├── app.py                  ← Aplikasi Streamlit utama
├── requirements.txt        ← Daftar library Python
├── car_price_model.pkl     ← Model yang sudah dilatih (dari train_and_save.py)
├── model_stats.json        ← Statistik & data untuk visualisasi
└── train_and_save.py       ← Script pelatihan model (jalankan di Colab)
```

## Cara Deploy ke Streamlit Cloud (Gratis)

### Langkah 1 — Buat akun
- Daftar di [streamlit.io](https://streamlit.io) (gunakan akun GitHub)

### Langkah 2 — Upload ke GitHub
1. Buat repository GitHub baru (contoh: `prediksi-harga-mobil`)
2. Upload semua file di atas ke repository tersebut

### Langkah 3 — Deploy
1. Login ke [share.streamlit.io](https://share.streamlit.io)
2. Klik **New app**
3. Pilih repository GitHub yang tadi dibuat
4. Set **Main file path** → `app.py`
5. Klik **Deploy!**
6. Tunggu beberapa menit → URL aplikasi siap diakses publik 🎉

## Cara Jalankan Lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fitur Aplikasi
- 🔮 **Prediksi Harga** — Input 9 spesifikasi teknis, output estimasi harga
- 🎯 **Preset Spesifikasi** — Ekonomis / Menengah / Premium
- 📊 **Analisis Pasar** — Chart top 10 mobil terlaris + insight koefisien model
- 📈 **Gauge Bar** — Posisi harga di rentang pasar
- 📋 **Ringkasan Spesifikasi** — Tampilan spesifikasi yang diinput
