"""
Data Preparation Script - Sample and Clean Your Dataset
"""
import pandas as pd
import numpy as np
from sklearn.utils import resample

def prepare_training_data():
    """
    Prepare your dataset: 956 suspicious + 168k normal (sampled from 1047619)
    Your columns: Time, Date, Sender_account, Receiver_account, Amount, Payment_currency, 
    Received_currency, Sender_bank_location, Receiver_bank_location, Payment_type, Is_laundering, Laundering_type
    """
    print("📖 Loading your AML1.csv dataset...")
    
    # Load your full dataset
    df = pd.read_csv("AML1.csv")  # Your actual dataset file
    
    print(f"Original dataset: {len(df)} transactions")
    print(f"Columns found: {list(df.columns)}")
    
    # Check current distribution
    suspicious_mask = df['Is_laundering'] == 1
    suspicious_txns = df[suspicious_mask]
    normal_txns = df[~suspicious_mask]
    
    print(f"Suspicious transactions: {len(suspicious_txns)}")
    print(f"Normal transactions: {len(normal_txns)}")
    
    # Keep ALL suspicious transactions (956)
    print("✅ Keeping all suspicious transactions")
    
    # Sample 168,000 normal transactions randomly
    print("🎲 Sampling 168,000 normal transactions...")
    
    # Make sure we don't sample more than available
    sample_size = min(168000, len(normal_txns))
    normal_sampled = resample(
        normal_txns, 
        n_samples=sample_size, 
        random_state=42, 
        replace=False
    )
    
    # Combine suspicious + sampled normal
    final_dataset = pd.concat([suspicious_txns, normal_sampled], ignore_index=True)
    
    # Shuffle the dataset
    final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Final dataset: {len(final_dataset)} transactions")
    print(f"   - Suspicious: {len(suspicious_txns)} ({len(suspicious_txns)/len(final_dataset)*100:.2f}%)")
    print(f"   - Normal: {len(normal_sampled)} ({len(normal_sampled)/len(final_dataset)*100:.2f}%)")
    print(f"   - Imbalance ratio: 1:{len(normal_sampled)//len(suspicious_txns)}")
    
    # Save prepared dataset
    final_dataset.to_csv("data/training_data.csv", index=False)
    print("💾 Saved to data/training_data.csv")
    
    # Display sample
    print("\n📋 Sample of prepared data:")
    print(final_dataset.head())
    print("\n📋 Column info:")
    print(final_dataset.info())
    
    return final_dataset

if __name__ == "__main__":
    # Create data directory
    import os
    os.makedirs('data', exist_ok=True)
    
    # Prepare the data
    prepare_training_data()