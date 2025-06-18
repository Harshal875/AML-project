"""
Simple FastAPI Backend for AML Transaction Screening
Single file with all endpoints
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import io
import os
from typing import List

# Add src directory to Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Load trained model and components
try:
    model = joblib.load('models/xgboost_aml_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_engineer = joblib.load('models/feature_engineer.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    MODEL_LOADED = True
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Model not found: {e}")
    MODEL_LOADED = False

# FastAPI app
app = FastAPI(title="AML Transaction Screening", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for validation
class TransactionInput(BaseModel):
    """Validate individual transaction from CSV"""
    Time: str
    Date: str  
    Sender_account: str
    Receiver_account: str
    Amount: float
    Payment_currency: str
    Received_currency: str
    Sender_bank_location: str
    Receiver_bank_location: str
    Payment_type: str

class TransactionResult(BaseModel):
    sender_account: str
    receiver_account: str
    amount: float
    risk_score: float
    is_suspicious: bool
    risk_level: str

class BatchResult(BaseModel):
    total_transactions: int
    suspicious_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    predictions: List[TransactionResult]

class Stats(BaseModel):
    total_transactions: int
    suspicious_transactions: int
    detection_rate: float
    risk_distribution: dict

class ValidationError(BaseModel):
    row: int
    column: str
    error: str
    value: str

class ValidationResult(BaseModel):
    is_valid: bool
    total_rows: int
    valid_rows: int
    errors: List[ValidationError]

# Global stats storage (in production, use a database)
PROCESSED_STATS = {
    "total_transactions": 0,
    "suspicious_transactions": 0,
    "risk_distribution": {"high_risk": 0, "medium_risk": 0, "low_risk": 0}
}

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "🏦 AML Transaction Screening API",
        "model_loaded": MODEL_LOADED,
        "endpoints": {
            "upload": "/upload-csv",
            "stats": "/stats",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED
    }

def validate_csv_data(df):
    """Validate CSV data using Pydantic"""
    errors = []
    valid_rows = 0
    
    required_columns = ['Time', 'Date', 'Sender_account', 'Receiver_account', 'Amount', 
                       'Payment_currency', 'Received_currency', 'Sender_bank_location', 
                       'Receiver_bank_location', 'Payment_type']
    
    # Check for missing columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return ValidationResult(
            is_valid=False,
            total_rows=len(df),
            valid_rows=0,
            errors=[ValidationError(row=0, column=', '.join(missing_cols), 
                                  error="Missing required columns", value="")]
        )
    
    # Validate each row
    for idx, row in df.iterrows():
        try:
            # Try to create TransactionInput object (validates data types)
            transaction = TransactionInput(
                Time=str(row['Time']).strip(),
                Date=str(row['Date']).strip(),
                Sender_account=str(row['Sender_account']).strip(),
                Receiver_account=str(row['Receiver_account']).strip(),
                Amount=float(row['Amount']),
                Payment_currency=str(row['Payment_currency']).strip(),
                Received_currency=str(row['Received_currency']).strip(),
                Sender_bank_location=str(row['Sender_bank_location']).strip(),
                Receiver_bank_location=str(row['Receiver_bank_location']).strip(),
                Payment_type=str(row['Payment_type']).strip()
            )
            
            # Additional validation rules
            if transaction.Amount <= 0:
                errors.append(ValidationError(
                    row=idx + 1, column="Amount", 
                    error="Amount must be positive", value=str(row['Amount'])
                ))
                continue
                
            # Validate date format (DD-MM-YYYY)
            try:
                pd.to_datetime(transaction.Date, format='%d-%m-%Y')
            except:
                errors.append(ValidationError(
                    row=idx + 1, column="Date", 
                    error="Date must be in DD-MM-YYYY format", value=transaction.Date
                ))
                continue
                
            # Validate time format (HH:MM:SS)
            try:
                pd.to_datetime(transaction.Time, format='%H:%M:%S')
            except:
                errors.append(ValidationError(
                    row=idx + 1, column="Time", 
                    error="Time must be in HH:MM:SS format", value=transaction.Time
                ))
                continue
            
            valid_rows += 1
            
        except Exception as e:
            errors.append(ValidationError(
                row=idx + 1, column="validation", 
                error=str(e), value=""
            ))
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        total_rows=len(df),
        valid_rows=valid_rows,
        errors=errors[:10]  # Return only first 10 errors
    )

@app.post("/validate-csv", response_model=ValidationResult)
async def validate_csv(file: UploadFile = File(...)):
    """Validate CSV file before processing"""
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    
    try:
        # Read CSV
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Validate data
        validation_result = validate_csv_data(df)
        return validation_result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

@app.post("/upload-csv", response_model=BatchResult)
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV and get AML predictions with validation"""
    
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not available. Train the model first.")
    
    # Validate file
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    
    try:
        # Read CSV with better parsing
        content = await file.read()
        
        # Try different separators and encodings
        csv_text = content.decode('utf-8-sig')  # Handle BOM
        
        # Try comma first, then semicolon
        try:
            df = pd.read_csv(io.StringIO(csv_text), sep=',')
            if len(df.columns) == 1:  # If only 1 column, try semicolon
                df = pd.read_csv(io.StringIO(csv_text), sep=';')
        except:
            df = pd.read_csv(io.StringIO(csv_text), sep=';')
        
        print(f"📊 CSV loaded: {len(df)} rows, columns: {list(df.columns)}")
        
        # Validate data first
        validation_result = validate_csv_data(df)
        if not validation_result.is_valid:
            error_details = []
            for error in validation_result.errors[:5]:  # Show first 5 errors
                error_details.append(f"Row {error.row}: {error.error} in {error.column}")
            
            print(f"❌ CSV validation failed: {'; '.join(error_details)}")
            raise HTTPException(
                status_code=400, 
                detail=f"CSV validation failed. Errors: {'; '.join(error_details)}"
            )
        
        print(f"✅ CSV validation passed. Processing {len(df)} transactions...")
        
        # Feature engineering
        df_features = feature_engineer.create_features(df)
        
        # Prepare features for prediction
        X = df_features[feature_columns].fillna(0)
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Process results
        results = []
        risk_counts = {"High": 0, "Medium": 0, "Low": 0}
        
        for i, (_, row) in enumerate(df.iterrows()):
            prob = float(probabilities[i])
            is_suspicious = bool(predictions[i])
            
            # Determine risk level
            if prob >= 0.7:
                risk_level = "High"
            elif prob >= 0.3:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            risk_counts[risk_level] += 1
            
            result = TransactionResult(
                sender_account=str(row['Sender_account']),
                receiver_account=str(row['Receiver_account']),
                amount=float(row['Amount']),
                risk_score=prob,
                is_suspicious=is_suspicious,
                risk_level=risk_level
            )
            results.append(result)
        
        # Update global stats
        PROCESSED_STATS["total_transactions"] += len(df)
        PROCESSED_STATS["suspicious_transactions"] += int(predictions.sum())
        PROCESSED_STATS["risk_distribution"]["high_risk"] += risk_counts["High"]
        PROCESSED_STATS["risk_distribution"]["medium_risk"] += risk_counts["Medium"]
        PROCESSED_STATS["risk_distribution"]["low_risk"] += risk_counts["Low"]
        
        print(f"✅ Processed {len(df)} transactions. Found {int(predictions.sum())} suspicious.")
        
        # Return batch results
        return BatchResult(
            total_transactions=len(df),
            suspicious_count=int(predictions.sum()),
            high_risk_count=risk_counts["High"],
            medium_risk_count=risk_counts["Medium"],
            low_risk_count=risk_counts["Low"],
            predictions=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/stats", response_model=Stats)
def get_stats():
    """Get processing statistics"""
    detection_rate = 0
    if PROCESSED_STATS["total_transactions"] > 0:
        detection_rate = (PROCESSED_STATS["suspicious_transactions"] / 
                         PROCESSED_STATS["total_transactions"]) * 100
    
    return Stats(
        total_transactions=PROCESSED_STATS["total_transactions"],
        suspicious_transactions=PROCESSED_STATS["suspicious_transactions"],
        detection_rate=detection_rate,
        risk_distribution=PROCESSED_STATS["risk_distribution"]
    )

@app.get("/model-info")
def get_model_info():
    """Get model information"""
    if not MODEL_LOADED:
        return {"status": "Model not loaded"}
    
    return {
        "status": "Active",
        "model_type": "XGBoost Classifier",
        "features_count": len(feature_columns),
        "target_recall": "85%",
        "training_data": "956 suspicious + 168k normal transactions",
        "techniques": ["SMOTE", "Velocity Features", "Feature Engineering"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)