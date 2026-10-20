import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, classification_report

def run_case_study():
    print("="*60)
    print("STUDI KASUS: PREDIKSI LATE PAYMENT PT MAJU BERSAMA")
    print("Berdasarkan Materi Perkuliahan 'Make Sense of Data' (Andria Arisal - BRIN)")
    print("="*60)

    # 1. DATA UNDERSTANDING & PREPARATION
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "pt_maju_bersama_invoices.csv")
    df = pd.read_csv(csv_path)
    print(f"\n[1] DATA UNDERSTANDING: Total {len(df)} transaksi loaded.")
    print(f"Target distribution (is_late_payment):\n{df['is_late_payment'].value_counts(normalize=True)*100}\n")

    # Eliminasi Data Leakage
    # Seperti dijelaskan Pak Andria: tanggal bayar riil, overdue days, & collection status
    # belum ada saat transaksi diterbitkan, sehingga HARUS dibuang agar tidak bocor!
    leakage_cols = ["actual_payment_date", "days_overdue", "collection_status"]
    df_clean = df.drop(columns=leakage_cols + ["invoice_id", "customer_id", "invoice_date", "due_date"])
    print(f"[2] DATA PREPARATION & LEAKAGE PREVENTION:")
    print(f"Mengeliminasi fitur kebocoran data: {leakage_cols}")
    print(f"Fitur terbersih yang digunakan: {list(df_clean.columns[:-1])}\n")

    # 2. DESCRIPTIVE & DIAGNOSTIC ANALYTICS
    print("[3] ANALISIS DESKRIPTIF & DIAGNOSTIK:")
    segment_summary = df.groupby("customer_segment")["is_late_payment"].agg(["count", "mean"])
    segment_summary["mean"] = segment_summary["mean"] * 100
    segment_summary.columns = ["Jumlah Transaksi", "Persentase Late Payment (%)"]
    print(segment_summary)
    print()

    # 3. PREDICTIVE ANALYTICS
    print("[4] ANALISIS PREDIKTIF (MACHINE LEARNING):")
    # Encoding categorical variables
    df_encoded = pd.get_dummies(df_clean, columns=["customer_segment", "payment_method"], drop_first=True)
    
    X = df_encoded.drop(columns=["is_late_payment"])
    y = df_encoded["is_late_payment"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Recall": round(rec, 4),
            "Precision": round(prec, 4),
            "ROC-AUC": round(roc, 4)
        })

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print()

    # 4. PRESCRIPTIVE ANALYTICS & HUMAN-IN-THE-LOOP
    print("[5] ANALISIS PRESKRIPTIF & REKOMENDASI BUSINESS RULES (HUMAN-IN-THE-LOOP):")
    print("- Rule 1: Jika credit_ratio > 0.70 pada segmen SME, berlakukan pengetatan limit kredit otomatis.")
    print("- Rule 2: Faktur berisiko tinggi (probabilitas prediksi > 0.60) dipicu untuk penagihan dini (early collection trigger) 7 hari sebelum due date.")
    print("- Rule 3: Keputusan pembekuan kredit tetap membutuhkan verifikasi manual manajer keuangan (Human-in-the-Loop).")
    print("="*60)

if __name__ == "__main__":
    run_case_study()
