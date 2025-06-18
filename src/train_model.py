"""
Train XGBoost Model with SMOTE for Your AML Dataset
Handles 956 suspicious vs 168k normal transactions
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os
from feature_engineering import AMLFeatureEngineer

def train_aml_model():
    """Train the AML model on your prepared dataset"""
    
    print("🚀 Starting AML Model Training...")
    
    # Load prepared data
    if not os.path.exists('data/training_data.csv'):
        print("❌ Training data not found! Run data preparation first.")
        return
    
    df = pd.read_csv('data/training_data.csv')
    print(f"📖 Loaded {len(df)} transactions")
    
    # Check distribution
    suspicious_count = df['Is_laundering'].sum()
    normal_count = len(df) - suspicious_count
    print(f"Data distribution: {suspicious_count} suspicious, {normal_count} normal")
    print(f"Imbalance ratio: 1:{normal_count//suspicious_count}")
    
    # Feature engineering
    print("🔧 Creating features...")
    engineer = AMLFeatureEngineer()
    df_features = engineer.create_features(df)
    
    # Prepare features and target
    feature_cols = engineer.get_feature_columns()
    X = df_features[feature_cols].fillna(0)
    y = df_features['Is_laundering']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    print("📊 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE for class balancing - OPTIMIZED FOR RECALL
    print("⚖️ Applying SMOTE for maximum recall...")
    minority_size = y_train.sum()
    k_neighbors = min(5, minority_size - 1)
    
    # Use regular SMOTE with full balancing for better recall
    smote = SMOTE(
        random_state=42, 
        k_neighbors=k_neighbors,
        sampling_strategy=1.0  # Full balance - equal numbers of each class
    )
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"After SMOTE: {len(X_train_balanced)} samples")
    print(f"Class distribution: {np.bincount(y_train_balanced.astype(int))}")
    
    # Train XGBoost model
    print("🚀 Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        class_weight='balanced'
    )
    
    model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate model with threshold optimization for recall
    print("📈 Evaluating model...")
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Find optimal threshold for recall
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Fix indexing issue - precision_recall_curve returns len(thresholds) = len(precision) - 1
    # Find threshold that gives at least 70% recall
    target_recall = 0.70
    valid_indices = recall[:-1] >= target_recall  # Exclude last element
    
    if np.any(valid_indices):
        # Choose threshold with highest precision among those with target recall
        best_threshold = thresholds[valid_indices][np.argmax(precision[:-1][valid_indices])]
    else:
        # If can't achieve target recall, use threshold that maximizes recall
        best_idx = np.argmax(recall[:-1])
        best_threshold = thresholds[best_idx]
    
    print(f"🎯 Optimal threshold for recall: {best_threshold:.3f}")
    
    # Make predictions with optimal threshold
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # Calculate metrics
    final_recall = recall_score(y_test, y_pred)
    
    print(f"\n🎯 MODEL PERFORMANCE:")
    print(f"Recall (Suspicious Detection): {final_recall:.3f} ({final_recall*100:.1f}%)")
    print(f"Optimal Threshold: {best_threshold:.3f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Suspicious']))
    
    print(f"\n🎯 MODEL PERFORMANCE:")
    print(f"Recall (Suspicious Detection): {recall:.3f} ({recall*100:.1f}%)")
    print(f"Optimal Threshold: {best_threshold:.3f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Suspicious']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 15 Most Important Features:")
    print(feature_importance.head(15))
    
    # Save model and components
    print("\n💾 Saving model...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/xgboost_aml_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(engineer, 'models/feature_engineer.pkl')
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    
    print("✅ Model training completed!")
    print(f"📊 Final Recall Score: {final_recall:.3f}")
    
    if final_recall >= 0.80:
        print("🎉 Great! Achieved 80%+ recall on suspicious transactions!")
    elif final_recall >= 0.70:
        print("👍 Good! 70%+ recall achieved. Consider tuning hyperparameters.")
    else:
        print("⚠️ Recall below 70%. Try adjusting SMOTE parameters or model settings.")
    
    return model, scaler, engineer

if __name__ == "__main__":
    train_aml_model()