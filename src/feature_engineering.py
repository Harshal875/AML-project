"""
Custom Feature Engineering for Your AML Dataset
Creates 25+ velocity-based features from your exact data structure
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder

class AMLFeatureEngineer:
    
    def __init__(self):
        self.encoders = {}
    
    def create_features(self, df):
        """
        Create 25+ features from your data structure:
        Time, Date, Sender_account, Receiver_account, Amount, Payment_currency, 
        Received_currency, Sender_bank_location, Receiver_bank_location, Payment_type
        """
        print(f"🔧 Creating features for {len(df)} transactions...")
        
        # Create datetime column
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d-%m-%Y %H:%M:%S')
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # 1-6: Basic Time Features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['month'] = df['datetime'].dt.month
        
        # 7-12: Amount Features
        df['amount_log'] = np.log1p(df['Amount'])
        df['is_round_amount'] = (df['Amount'] % 1000 == 0).astype(int)
        df['is_just_under_10k'] = ((df['Amount'] >= 9000) & (df['Amount'] < 10000)).astype(int)
        df['is_large_amount'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)
        df['is_small_amount'] = (df['Amount'] < df['Amount'].quantile(0.05)).astype(int)
        df['amount_category'] = pd.cut(df['Amount'], bins=5, labels=[0,1,2,3,4]).astype(int)
        
        # 13-16: Geographic Features
        df['is_cross_border'] = (df['Sender_bank_location'] != df['Receiver_bank_location']).astype(int)
        df['is_currency_exchange'] = (df['Payment_currency'] != df['Received_currency']).astype(int)
        df['same_country'] = (df['Sender_bank_location'] == df['Receiver_bank_location']).astype(int)
        
        # High-risk countries (based on your data)
        high_risk_countries = ['Pakistan', 'Albania', 'Morocco', 'Nigeria']
        df['sender_high_risk'] = df['Sender_bank_location'].isin(high_risk_countries).astype(int)
        df['receiver_high_risk'] = df['Receiver_bank_location'].isin(high_risk_countries).astype(int)
        
        # 17-19: Payment Type Features
        cash_types = ['Cash Deposit', 'Cash Withdrawal']
        df['is_cash_transaction'] = df['Payment_type'].isin(cash_types).astype(int)
        df['is_cross_border_payment'] = (df['Payment_type'] == 'Cross-border').astype(int)
        df['is_card_payment'] = df['Payment_type'].isin(['Credit card', 'Debit card']).astype(int)
        
        # 20-25+: Velocity Features (KEY FOR DETECTING PATTERNS)
        df = self._create_velocity_features(df)
        
        # Encode categorical variables
        df = self._encode_categorical_features(df)
        
        print(f"✅ Created {len(self.get_feature_columns())} features")
        return df
    
    def _create_velocity_features(self, df):
        """
        Create velocity-based features to detect smurfing and layering
        This is the most important part for detecting suspicious patterns!
        """
        print("🕒 Creating velocity features...")
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Initialize velocity features
        velocity_cols = [
            'txn_count_1h', 'txn_count_24h', 'txn_count_7d',
            'amount_sum_1h', 'amount_sum_24h', 'amount_sum_7d',
            'unique_receivers_1h', 'unique_receivers_24h', 'unique_receivers_7d',
            'avg_amount_1h', 'avg_amount_24h', 'avg_amount_7d',
            'max_amount_1h', 'max_amount_24h', 'max_amount_7d',
            'minutes_since_last_txn'
        ]
        
        for col in velocity_cols:
            df[col] = 0.0
        
        # Process each sender account
        for sender in df['Sender_account'].unique():
            sender_mask = df['Sender_account'] == sender
            sender_txns = df[sender_mask].copy()
            
            if len(sender_txns) <= 1:
                continue
                
            for idx, row in sender_txns.iterrows():
                current_time = row['datetime']
                
                # Get transactions before current time
                before_mask = sender_txns['datetime'] < current_time
                before_txns = sender_txns[before_mask]
                
                if len(before_txns) == 0:
                    continue
                
                # Time windows
                window_1h = current_time - timedelta(hours=1)
                window_24h = current_time - timedelta(hours=24)
                window_7d = current_time - timedelta(days=7)
                
                # 1-hour window
                mask_1h = before_txns['datetime'] >= window_1h
                txns_1h = before_txns[mask_1h]
                
                df.loc[idx, 'txn_count_1h'] = len(txns_1h)
                if len(txns_1h) > 0:
                    df.loc[idx, 'amount_sum_1h'] = txns_1h['Amount'].sum()
                    df.loc[idx, 'unique_receivers_1h'] = txns_1h['Receiver_account'].nunique()
                    df.loc[idx, 'avg_amount_1h'] = txns_1h['Amount'].mean()
                    df.loc[idx, 'max_amount_1h'] = txns_1h['Amount'].max()
                
                # 24-hour window
                mask_24h = before_txns['datetime'] >= window_24h
                txns_24h = before_txns[mask_24h]
                
                df.loc[idx, 'txn_count_24h'] = len(txns_24h)
                if len(txns_24h) > 0:
                    df.loc[idx, 'amount_sum_24h'] = txns_24h['Amount'].sum()
                    df.loc[idx, 'unique_receivers_24h'] = txns_24h['Receiver_account'].nunique()
                    df.loc[idx, 'avg_amount_24h'] = txns_24h['Amount'].mean()
                    df.loc[idx, 'max_amount_24h'] = txns_24h['Amount'].max()
                
                # 7-day window
                mask_7d = before_txns['datetime'] >= window_7d
                txns_7d = before_txns[mask_7d]
                
                df.loc[idx, 'txn_count_7d'] = len(txns_7d)
                if len(txns_7d) > 0:
                    df.loc[idx, 'amount_sum_7d'] = txns_7d['Amount'].sum()
                    df.loc[idx, 'unique_receivers_7d'] = txns_7d['Receiver_account'].nunique()
                    df.loc[idx, 'avg_amount_7d'] = txns_7d['Amount'].mean()
                    df.loc[idx, 'max_amount_7d'] = txns_7d['Amount'].max()
                
                # Time since last transaction
                last_txn_time = before_txns['datetime'].max()
                minutes_diff = (current_time - last_txn_time).total_seconds() / 60
                df.loc[idx, 'minutes_since_last_txn'] = minutes_diff
        
        # Additional ratio features
        df['velocity_ratio_24h_1h'] = np.where(df['txn_count_1h'] > 0, 
                                              df['txn_count_24h'] / df['txn_count_1h'], 0)
        df['amount_velocity_ratio'] = np.where(df['amount_sum_24h'] > 0,
                                              df['Amount'] / df['amount_sum_24h'], 1)
        df['receiver_diversity_ratio'] = np.where(df['txn_count_24h'] > 0,
                                                 df['unique_receivers_24h'] / df['txn_count_24h'], 0)
        
        print("✅ Velocity features created")
        return df
    
    def _encode_categorical_features(self, df):
        """Encode categorical variables"""
        categorical_cols = ['Payment_type', 'Sender_bank_location', 'Receiver_bank_location', 
                           'Payment_currency', 'Received_currency']
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                known_categories = self.encoders[col].classes_
                df[f'{col}_encoded'] = df[col].astype(str).apply(
                    lambda x: self.encoders[col].transform([x])[0] if x in known_categories else -1
                )
        
        return df
    
    def get_feature_columns(self):
        """Return list of all feature columns for ML model"""
        return [
            'Amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend', 'is_night', 
            'is_business_hours', 'month', 'is_round_amount', 'is_just_under_10k', 
            'is_large_amount', 'is_small_amount', 'amount_category', 'is_cross_border',
            'is_currency_exchange', 'same_country', 'sender_high_risk', 'receiver_high_risk',
            'is_cash_transaction', 'is_cross_border_payment', 'is_card_payment',
            'txn_count_1h', 'txn_count_24h', 'txn_count_7d', 'amount_sum_1h', 
            'amount_sum_24h', 'amount_sum_7d', 'unique_receivers_1h', 'unique_receivers_24h',
            'unique_receivers_7d', 'avg_amount_1h', 'avg_amount_24h', 'avg_amount_7d',
            'max_amount_1h', 'max_amount_24h', 'max_amount_7d', 'minutes_since_last_txn',
            'velocity_ratio_24h_1h', 'amount_velocity_ratio', 'receiver_diversity_ratio',
            'Payment_type_encoded', 'Sender_bank_location_encoded', 'Receiver_bank_location_encoded',
            'Payment_currency_encoded', 'Received_currency_encoded'
        ]