#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import date, timedelta
import os
import re
import pickle
import gzip

st.set_page_config(page_title="Retail Demand Forecasting Dashboard", page_icon="📦", layout="wide")

# load model data
@st.cache_data
def load_raw_data():
    """Loads and preprocesses the raw data from a CSV file."""
    try:
        df = pd.read_csv('df_model_sarimax.csv')
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        else:
            df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        return df
    except FileNotFoundError:
        st.error("File 'df_model_sarimax.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data mentah: {e}")
        st.stop()

df_model_sarimax = load_raw_data()
last_historical_date = df_model_sarimax.index.max().date()

@st.cache_resource
def load_objects():
    try:
        models_path = 'sarimax_final_models.pkl.gz'
        eval_path = 'eval_dict.pkl'
        freq_path = 'frequency_dict.pkl'

        if not os.path.exists(models_path):
            st.error(f"File model '{models_path}' tidak ditemukan.")
            return None, None, None, None
        if not os.path.exists(eval_path):
            st.error(f"File evaluasi '{eval_path}' tidak ditemukan.")
            return None, None, None, None
        if not os.path.exists(freq_path):
            st.error(f"File frekuensi '{freq_path}' tidak ditemukan.")
            return None, None, None, None
        
        with gzip.open(models_path, 'rb') as f:
            final_models_dict = pickle.load(f)

        with open(eval_path, 'rb') as f:
            eval_dict = pickle.load(f)
        
        with open(freq_path, 'rb') as f:
            frequency_dict = pickle.load(f)
        
        # Memisahkan preprocessor dan model dari dictionary
        preprocessor_dict = {cat: data['preprocessor'] for cat, data in final_models_dict.items()}
        sarimax_models = {cat: data['model'] for cat, data in final_models_dict.items()}
        
        return preprocessor_dict, eval_dict, sarimax_models, frequency_dict
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat objek: {e}. Pastikan file .pkl tidak rusak.")
        return None, None, None, None

preprocessor_dict, eval_dict, sarimax_models, frequency_dict = load_objects()

# Hentikan aplikasi jika ada objek yang gagal dimuat
if preprocessor_dict is None or frequency_dict is None:
    st.stop()
    
categories = list(preprocessor_dict.keys())
onehot_features = ['Weather Condition', 'Region', 'Store ID', 'discount_level', 'price_level']
ordinal_features = ['Seasonality']
exog_passthrough_all = ['Promotion', 'Epidemic', 'Units Ordered', 'Price_Diff', 'dayofweek_sin', 'dayofweek_cos']

missing_exog_features = ['lag_7', 'rolling_mean_7']

# Gunakan daftar kolom lengkap untuk membuat DataFrame X_future_raw
exog_cols_raw = onehot_features + ordinal_features + exog_passthrough_all + missing_exog_features

st.title("🛒 Retail Demand Forecasting Dashboard")
st.markdown("Dashboard ini memprediksi permintaan harian dan merekomendasikan stok optimal berdasarkan data historis dan skenario masa depan.")

st.sidebar.header("⚙️ Pengaturan Prediksi")

with st.sidebar.form("prediction_form"):
    st.subheader("Rentang Tanggal")
    min_date_pred = last_historical_date + timedelta(days=1)
    # PERBAIKAN: Batasi tanggal maksimal menjadi 180 hari setelah data historis terakhir
    max_date_pred = last_historical_date + timedelta(days=180)
    start_date = st.date_input("Tanggal Mulai Prediksi", value=min_date_pred, min_value=min_date_pred, max_value=max_date_pred)
    end_date = st.date_input("Tanggal Akhir Prediksi", value=min(start_date + timedelta(days=30), max_date_pred), min_value=start_date, max_value=max_date_pred)
    if start_date >= end_date:
        st.error("Tanggal mulai harus lebih awal dari tanggal akhir.")
        st.stop()
    
    st.subheader("Pilihan Kategori")
    selected_category = st.selectbox("Pilih Kategori", categories)
    
    st.subheader("Skenario Bisnis")
    promo_scenario = st.selectbox("Promosi?", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    discount_scenario = st.selectbox("Tingkat Diskon", options=['rendah', 'sedang', 'tinggi'])
    submitted = st.form_submit_button("Generate Forecast")

if submitted:
    with st.container(height=600, border=True):
        st.header(f"Hasil Prediksi untuk Kategori: **{selected_category}**")

        model = sarimax_models[selected_category]
        preprocessor = preprocessor_dict[selected_category]
        mape = eval_dict[selected_category]['MAPE_test']

        forecast_days = (end_date - start_date).days + 1
        future_index = pd.date_range(start=start_date, periods=forecast_days, freq='D')
        
        current_cat_freq = frequency_dict[selected_category]
        current_subset = df_model_sarimax[df_model_sarimax['Category'] == selected_category].copy()
        
        # Buat DataFrame X_future_raw dengan semua kolom yang diperlukan
        X_future_raw = pd.DataFrame(index=future_index, columns=exog_cols_raw)
        
        for col in X_future_raw.columns:
            if col in onehot_features + ordinal_features:
                col_freq = current_cat_freq.get(col, {})
                if col == 'discount_level':
                    X_future_raw[col] = discount_scenario
                elif col_freq:
                    mode_value = max(col_freq, key=col_freq.get)
                    X_future_raw[col] = mode_value
                else:
                    X_future_raw[col] = current_subset[col].mode()[0]
            elif col == 'Promotion':
                X_future_raw[col] = promo_scenario
            elif col == 'Epidemic':
                X_future_raw[col] = 0
            # PERBAIKAN UTAMA DI SINI
            # Pastikan kolom-kolom yang diharapkan preprocessor diisi dengan nilai yang valid
            elif col == 'lag_7':
                X_future_raw[col] = current_subset['Units Ordered'].iloc[-1]
            elif col == 'rolling_mean_7':
                X_future_raw[col] = current_subset['Units Ordered'].iloc[-7:].mean()
            elif col in ['Units Ordered', 'Price_Diff']:
                X_future_raw[col] = current_subset[col].mean()
            elif col == 'dayofweek_sin':
                X_future_raw[col] = np.sin(2 * np.pi * X_future_raw.index.dayofweek / 7)
            elif col == 'dayofweek_cos':
                X_future_raw[col] = np.cos(2 * np.pi * X_future_raw.index.dayofweek / 7)
        
        # Lakukan transformasi pada data mentah
        X_future_encoded = preprocessor.transform(X_future_raw)
        
        # Dapatkan nama-nama kolom yang dihasilkan oleh preprocessor
        preprocessor_names = preprocessor.get_feature_names_out()
        
        # Hapus awalan (prefix) dari nama kolom yang dihasilkan preprocessor
        renamed_names = [re.sub(r'^\w+__', '', col) for col in preprocessor_names]

        # Dapatkan nama-nama kolom eksogen yang dibutuhkan oleh model
        required_exog_names = model.model.exog_names

        # Buat DataFrame dari hasil encoding dan nama kolom yang sudah diperbaiki
        X_future_full_df = pd.DataFrame(X_future_encoded, index=X_future_raw.index, columns=renamed_names).fillna(0)
        
        # Filter DataFrame yang sudah lengkap menggunakan nama kolom yang dibutuhkan oleh model.
        X_future_final = X_future_full_df[required_exog_names]
            
        forecast_result = model.get_forecast(steps=forecast_days, exog=X_future_final.astype(float))
        forecast_df = forecast_result.predicted_mean.to_frame('predicted_demand')

        forecast_df['optimal_stock'] = forecast_df['predicted_demand'] * (1 + mape)
        forecast_df['optimal_stock'] = forecast_df['optimal_stock'].apply(lambda x: max(0, x))
        forecast_df = forecast_df.round(2)

        st.subheader("Grafik Prediksi Demand & Stok Optimal")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(forecast_df.index, forecast_df['predicted_demand'], label='Predicted Demand', color='green', marker='o', markersize=3)
        ax.plot(forecast_df.index, forecast_df['optimal_stock'], label='Optimal Stock', color='orange', linestyle='--')
        ax.set_title(f"Prediksi Harian {selected_category} ({start_date} s/d {end_date})")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Jumlah Unit")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig) 

        st.subheader("Rekomendasi Stok Optimal (Tabel)")
        st.dataframe(forecast_df, use_container_width=True)

st.sidebar.header("💡 Informasi")
st.sidebar.markdown("Dasboard ini hanya untuk predict forecast 6 bulan setelah data historis")
st.sidebar.markdown("Ketika akan mengaktifkan tingkat diskon maka ubah input Promosi menjadi 'ya'")

