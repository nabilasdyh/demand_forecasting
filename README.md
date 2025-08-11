

🛒 Retail Demand Forecasting Dashboard

🚀 Mengubah Data Menjadi Keputusan Bisnis Cerdas
Selamat datang di proyek Retail Demand Forecasting Dashboard! Proyek ini adalah solusi berbasis Machine Learning yang dirancang untuk membantu manajer ritel mengoptimalkan manajemen stok. Dashboard interaktif ini memungkinkan kita untuk memprediksi permintaan produk di masa depan, mencegah stockout yang merugikan, dan menghindari overstock yang memboroskan biaya.

🎯 Masalah yang Dipecahkan
Manajemen stok yang tidak akurat bisa menjadi mimpi buruk. Kekurangan produk (stockout) berarti kehilangan penjualan dan kepuasan pelanggan, sementara kelebihan stok (overstock) mengikat modal dan menambah biaya penyimpanan. Dashboard ini bertujuan untuk memecahkan masalah ini dengan menyediakan prediksi permintaan yang akurat dan rekomendasi stok yang optimal.

✨ Fitur-Fitur Utama

Prediksi Demand 📈: Memanfaatkan model SARIMAX untuk memprediksi permintaan harian produk secara akurat.

Simulasi Skenario 🎲: Lakukan prediksi dengan skenario bisnis yang berbeda, seperti adanya promosi atau tingkat diskon tertentu.

Rekomendasi Stok Optimal 📦: Dashboard secara otomatis merekomendasikan jumlah stok optimal yang harus disiapkan, dengan mempertimbangkan metrik akurasi model (MAPE).

Visualisasi Interaktif 📊: Semua hasil disajikan dalam grafik interaktif yang mudah dipahami, sehingga pengambilan keputusan menjadi lebih cepat.


🛠️ Teknologi yang Digunakan

Bahasa: Python 🐍

Dashboard: Streamlit ✨

Model: Statsmodels (SARIMAX) 📈

Data & Preprocessing: Pandas, NumPy, Scikit-learn

Visualisasi: Matplotlib


📂 Struktur Proyek
Repositori ini disusun dengan rapi untuk memudahkan navigasi:

.
├── .gitignore

├── requirements.txt

├── demand_forecast.ipynb      # Skrip EDA dan pelatihan model xgboost dan sarimax

├── demand_forecast.py         # Skrip utama aplikasi Streamlit

├── df_model_sarimax.csv       # Dataset historis untuk pelatihan model

├── sarimax_models.pkl.gz      # File terkompresi berisi model SARIMAX dan preprocessor data yang sudah dilatih

├── eval_dict.pkl              # File metrik evaluasi model (MAE & MAPE)

├── frequency_dict.pkl         # File frekuensi untuk fitur eksogen

└── README.md                  # File ini


LinkedIn: (https://www.linkedin.com/in/nabilasadiyahh/)
