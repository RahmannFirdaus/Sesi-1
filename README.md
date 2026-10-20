# Studi Kasus: Prediksi Keterlambatan Pembayaran (Late Payment) PT Maju Bersama

Repositori ini berisi implementasi *end-to-end* studi kasus analitik data untuk memprediksi probabilitas keterlambatan pembayaran (*late payment*) faktur/invoice oleh pelanggan. Proyek ini diinspirasi dari materi perkuliahan "Make Sense of Data" oleh Bapak Andria Arisal (BRIN).

Melalui repositori ini, kita mensimulasikan alur kerja Data Science mulai dari pembuatan dataset, pembersihan data (terutama penanganan kebocoran data/ *data leakage*), analisis deskriptif, pemodelan prediktif (Machine Learning), hingga rekomendasi preskriptif.

## 📂 Struktur Repositori

Terdapat 3 file utama dalam repositori ini:

1. **`generate_dataset.py`**
   Script Python untuk menghasilkan dataset sintetis `pt_maju_bersama_invoices.csv`. Script ini membuat profil pelanggan, menerbitkan 4000 invoice, dan mensimulasikan logika terjadinya *late payment* (misalnya: rasio kredit tinggi dan segmen SME memiliki probabilitas gagal bayar lebih tinggi). Script ini juga sengaja menyertakan variabel *data leakage*.

2. **`case_study_late_payment.py`**
   Script utama yang menjalankan seluruh *pipeline* analitik. Script ini membaca dataset, melakukan pembersihan fitur, membagi data, melatih beberapa model Machine Learning, dan mengevaluasi performanya.

3. **`pt_maju_bersama_invoices.csv`**
   Merupakan dataset hasil *generate* berisi data riwayat invoice, detail pelanggan, dan status pembayaran.

## 🔍 Alur Analitik (Data Pipeline)

Proyek ini mempraktikkan 4 tahap analitik utama:

### 1. Data Understanding & Preparation
* Membaca dan memahami distribusi target kelas (`is_late_payment`).
* **Pencegahan Data Leakage (Krusial):** Script ini menghapus kolom `actual_payment_date`, `days_overdue`, dan `collection_status`. Ketiga fitur ini adalah jebakan *data leakage* karena secara logis belum diketahui pada saat invoice baru saja diterbitkan (saat prediksi dilakukan di dunia nyata).

### 2. Descriptive & Diagnostic Analytics
* Menganalisis tingkat gagal bayar berdasarkan segmen pelanggan (Corporate, SME, Retail) untuk mencari *insight* diagnostik awal.

### 3. Predictive Analytics (Machine Learning)
* Melakukan *encoding* variabel kategorikal (seperti *customer segment* dan *payment method*).
* Melatih 3 model klasifikasi: **Logistic Regression, Decision Tree, dan Random Forest**.
* Mengevaluasi model menggunakan metrik *Accuracy, Recall, Precision*, dan *ROC-AUC*.

### 4. Prescriptive Analytics & Human-in-the-loop
* Memberikan rekomendasi aturan bisnis (*Business Rules*) berdasarkan hasil pemodelan, seperti pengetatan limit kredit otomatis dan peringatan dini (*early collection trigger*).
* Menerapkan konsep *Human-in-the-Loop* dimana keputusan final yang berisiko tinggi (seperti pembekuan kredit) tetap memerlukan verifikasi manusia.

## 🚀 Cara Menjalankan Proyek

**Persyaratan Sistem:**
instal Python 3.7+ dan pustaka berikut:
* `pandas`
* `numpy`
* `scikit-learn`

**Langkah-langkah Eksekusi:**

1. **(Opsional) Generate ulang dataset:**
   Jika ingin membuat dataset baru dengan *seed* yang sama atau mengubah parameter jumlah data, jalankan:
   ```bash
   python generate_dataset.py
   ```

2. **Jalankan Studi Kasus Analitik:**
   Untuk memproses data, melihat analisis, dan melatih model Machine Learning, jalankan:
   ```bash
   python case_study_late_payment.py
   ```

## ⚠️ Konsep Penting: Data Leakage
Salah satu pelajaran utama dari repositori ini adalah kehati-hatian terhadap **Data Leakage** (Kebocoran Data). Model Machine Learning tidak boleh dilatih menggunakan variabel yang di dunia nyata terjadi **setelah** target diprediksi (contoh: kita tidak bisa memprediksi apakah orang akan telat bayar dengan menggunakan variabel "jumlah hari telat" yang baru diketahui setelah tanggal jatuh tempo lewat).
