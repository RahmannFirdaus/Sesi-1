import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Resolve path relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "pt_maju_bersama_invoices.csv")

# Set seed for reproducibility
np.random.seed(42)

# Parameters matching Pak Andria Arisal's lecture case study (PT Maju Bersama)
n_customers = 120
n_products = 24
n_invoices = 4000
start_date = datetime(2024, 7, 1)

# Generate Customer Master Data
customer_ids = [f"CUST_{i+1:03d}" for i in range(n_customers)]
customer_segments = np.random.choice(["Corporate", "SME", "Retail"], size=n_customers, p=[0.2, 0.5, 0.3])
credit_limits = np.random.choice([50000000, 150000000, 500000000], size=n_customers, p=[0.4, 0.4, 0.2])

cust_df = pd.DataFrame({
    "customer_id": customer_ids,
    "customer_segment": customer_segments,
    "credit_limit": credit_limits
})

# Generate Invoices Data
invoices = []
payment_methods = ["Bank Transfer", "Credit Terms 30D", "Credit Terms 60D", "Giro"]

for i in range(1, n_invoices + 1):
    cust = cust_df.sample(1).iloc[0]
    inv_date = start_date + timedelta(days=int(np.random.randint(0, 700)))
    due_date = inv_date + timedelta(days=30 if "30D" in cust["customer_segment"] else 45)
    
    # Amount between 5M and 150M IDR
    amount = float(np.random.exponential(scale=25000000) + 2000000)
    credit_ratio = float(min(1.2, amount / cust["credit_limit"]))
    
    # Probability of late payment correlates with high credit ratio, SME segment, and longer credit terms
    prob_late = 0.05
    if credit_ratio > 0.7:
        prob_late += 0.25
    if cust["customer_segment"] == "SME":
        prob_late += 0.15
    if "Credit" in payment_methods[i % 4]:
        prob_late += 0.10
        
    is_late = 1 if np.random.rand() < prob_late else 0
    
    # Data leakage variables (present in historical records but NOT available at prediction time!)
    if is_late:
        days_overdue = int(np.random.randint(31, 120))
        actual_payment_date = due_date + timedelta(days=days_overdue)
        collection_status = "Collection Agency" if days_overdue > 60 else "Late Follow-up"
    else:
        days_overdue = int(np.random.randint(-10, 5))
        actual_payment_date = due_date + timedelta(days=days_overdue)
        collection_status = "Paid On Time"
        
    invoices.append({
        "invoice_id": f"INV-{i:05d}",
        "customer_id": cust["customer_id"],
        "customer_segment": cust["customer_segment"],
        "payment_method": payment_methods[i % 4],
        "invoice_amount": round(amount, 2),
        "credit_ratio": round(credit_ratio, 4),
        "invoice_date": inv_date.strftime("%Y-%m-%d"),
        "due_date": due_date.strftime("%Y-%m-%d"),
        # Target variable
        "is_late_payment": is_late,
        # LEAKAGE FEATURES (must be dropped before model training)
        "actual_payment_date": actual_payment_date.strftime("%Y-%m-%d"),
        "days_overdue": days_overdue,
        "collection_status": collection_status
    })

df = pd.DataFrame(invoices)
df.to_csv(CSV_PATH, index=False)
print(f"Dataset generated successfully with {len(df)} rows.")
print(f"Late payment rate: {df['is_late_payment'].mean()*100:.2f}%")
