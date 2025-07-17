"""
Simplified AML Transaction Screening API
Clean, focused backend with ML + RAG capabilities
"""
import os
import sys
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

# Core FastAPI imports
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# Data processing
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Import compliance router
try:
    from compliance.routes import compliance_router
    COMPLIANCE_AVAILABLE = True
    print("✅ Compliance module loaded successfully")
except ImportError as e:
    print(f"❌ Failed to load compliance module: {e}")
    COMPLIANCE_AVAILABLE = False

# Configure logging
def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger(__name__)

logger = setup_logging()

# Global application state
class AppState:
    def __init__(self):
        self.ml_components = None
        self.preprocessor = None
        self.stats = {
            "total_transactions": 0,
            "suspicious_transactions": 0,
            "uptime_start": pd.Timestamp.now(),
            "requests_processed": 0
        }
        self.health_status = "starting"

app_state = AppState()

# AML Preprocessor Class
class AMLPreprocessor:
    """Clean AML feature preprocessor - same 33 features as training"""
    
    def __init__(self, encoders, scaler):
        self.encoders = encoders
        self.scaler = scaler
        self.high_risk_countries = [
            'Albania', 'Nigeria', 'Pakistan', 'Morocco', 'Turkey',
            'Afghanistan', 'Barbados', 'Botswana', 'Burkina Faso',
            'Cambodia', 'Cayman Islands', 'Haiti', 'Iran', 'Jamaica', 
            'Jordan', 'Mali', 'Myanmar', 'Nicaragua', 'Panama', 
            'Philippines', 'Senegal', 'South Sudan', 'Syria', 
            'Uganda', 'Yemen', 'Zimbabwe'
        ]
        logger.info("✅ AML preprocessor initialized")
    
    def create_features(self, df):
        """Create the same 33 features used in training"""
        try:
            df = df.copy()
            
            # 1. DATETIME FEATURES
            df['Date'] = pd.to_datetime(df['Date'])
            df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
            
            df['hour'] = df['Time'].dt.hour
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['month'] = df['Date'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            df['is_night_transaction'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            
            # 2. AMOUNT FEATURES
            df['amount_log'] = np.log1p(df['Amount'])
            df['is_large_amount'] = (df['Amount'] > 10000).astype(int)
            df['is_just_under_threshold'] = ((df['Amount'] >= 9000) & (df['Amount'] < 10000)).astype(int)
            df['is_round_amount'] = (df['Amount'] % 1000 == 0).astype(int)
            df['is_small_amount'] = (df['Amount'] < 100).astype(int)
            df['amount_percentile'] = df['Amount'].rank(pct=True)
            
            # 3. GEOGRAPHIC FEATURES
            df['is_cross_border'] = (df['Sender_bank_location'] != df['Receiver_bank_location']).astype(int)
            df['sender_high_risk'] = df['Sender_bank_location'].isin(self.high_risk_countries).astype(int)
            df['receiver_high_risk'] = df['Receiver_bank_location'].isin(self.high_risk_countries).astype(int)
            df['involves_high_risk_country'] = ((df['sender_high_risk'] == 1) | (df['receiver_high_risk'] == 1)).astype(int)
            
            # 4. CURRENCY FEATURES
            df['is_currency_exchange'] = (df['Payment_currency'] != df['Received_currency']).astype(int)
            
            # 5. PAYMENT TYPE FEATURES
            df['is_cash_transaction'] = df['Payment_type'].isin(['Cash Deposit', 'Cash Withdrawal']).astype(int)
            df['is_cash_deposit'] = (df['Payment_type'] == 'Cash Deposit').astype(int)
            df['is_cash_withdrawal'] = (df['Payment_type'] == 'Cash Withdrawal').astype(int)
            df['is_cross_border_payment'] = (df['Payment_type'] == 'Cross-border').astype(int)
            df['is_card_payment'] = df['Payment_type'].isin(['Credit card', 'Debit card']).astype(int)
            df['is_wire_transfer'] = df['Payment_type'].isin(['ACH', 'Cross-border']).astype(int)
            
            # 6. VELOCITY FEATURES
            df['account_date_key'] = df['Sender_account'].astype(str) + '_' + df['Date'].astype(str)
            
            account_date_counts = df['account_date_key'].value_counts().to_dict()
            df['daily_txn_count'] = df['account_date_key'].map(account_date_counts)
            
            account_date_amounts = df.groupby('account_date_key')['Amount'].sum().to_dict()
            df['daily_amount_sum'] = df['account_date_key'].map(account_date_amounts)
            
            account_date_receivers = df.groupby('account_date_key')['Receiver_account'].nunique().to_dict()
            df['unique_receivers_today'] = df['account_date_key'].map(account_date_receivers)
            
            account_avg_daily = df.groupby('Sender_account')['daily_txn_count'].transform('mean')
            df['transaction_frequency_score'] = np.where(
                account_avg_daily > 0,
                df['daily_txn_count'] / account_avg_daily,
                1.0
            )
            
            # Clean up
            df = df.drop(['account_date_key'], axis=1)
            
            # 7. ENCODE CATEGORICAL VARIABLES
            categorical_cols = ['Payment_type', 'Sender_bank_location', 'Receiver_bank_location', 
                               'Payment_currency', 'Received_currency']
            
            for col in categorical_cols:
                if col in self.encoders:
                    known_values = set(self.encoders[col].classes_)
                    df[f'{col}_encoded'] = df[col].astype(str).apply(
                        lambda x: self.encoders[col].transform([x])[0] if x in known_values else -1
                    )
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Feature creation failed: {str(e)}")
            raise

def load_ml_components():
    """Load ML components with error handling"""
    try:
        model_path = Path("models")
        if not model_path.exists():
            raise FileNotFoundError("Models directory not found")
        
        required_files = [
            'trained_aml_model.pkl', 'optimal_threshold.pkl', 'scaler.pkl',
            'encoders.pkl', 'feature_columns.pkl'
        ]
        
        missing_files = [f for f in required_files if not (model_path / f).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {missing_files}")
        
        logger.info("📦 Loading ML components...")
        
        components = {
            'model': joblib.load('models/trained_aml_model.pkl'),
            'threshold': joblib.load('models/optimal_threshold.pkl'),
            'scaler': joblib.load('models/scaler.pkl'),
            'encoders': joblib.load('models/encoders.pkl'),
            'feature_columns': joblib.load('models/feature_columns.pkl'),
            'loaded': True
        }
        
        logger.info(f"✅ ML components loaded - Threshold: {components['threshold']}")
        return components
        
    except Exception as e:
        logger.error(f"❌ Failed to load ML components: {str(e)}")
        return {'loaded': False, 'error': str(e)}

async def startup_tasks():
    """Startup tasks for the application"""
    try:
        logger.info("🚀 Starting AML Transaction Screening API...")
        
        # Create necessary directories
        for directory in ["data/regulatory_docs", "data/local_vectors", "logs"]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Load ML components
        app_state.ml_components = load_ml_components()
        
        if app_state.ml_components['loaded']:
            app_state.preprocessor = AMLPreprocessor(
                app_state.ml_components['encoders'],
                app_state.ml_components['scaler']
            )
        
        app_state.health_status = "healthy"
        logger.info("🎉 Application startup complete")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        app_state.health_status = "unhealthy"

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_tasks()
    yield
    logger.info("🛑 Application shutting down")

# Create FastAPI application
app = FastAPI(
    title="AML Transaction Screening API",
    description="AML compliance and transaction screening with ML + RAG",
    version="2.0.0",
    lifespan=lifespan
)

# Include compliance router
if COMPLIANCE_AVAILABLE:
    app.include_router(compliance_router)
    logger.info("✅ Compliance router included")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request counter middleware
@app.middleware("http")
async def count_requests(request, call_next):
    app_state.stats["requests_processed"] += 1
    response = await call_next(request)
    return response

# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return await http_exception_handler(request, exc)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"status": "error", "error": "validation_failed", "details": str(exc)}
    )

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime: str
    version: str
    components: dict
    stats: dict

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "🏦 AML Transaction Screening API v2.0",
        "status": app_state.health_status,
        "ml_model": "loaded" if app_state.ml_components and app_state.ml_components['loaded'] else "not_loaded",
        "compliance_module": "available" if COMPLIANCE_AVAILABLE else "unavailable",
        "capabilities": [
            "ML-based transaction risk assessment (XGBoost + 33 features)",
            "RAG-enabled compliance analysis" if COMPLIANCE_AVAILABLE else "Basic analysis only",
            "Batch CSV processing with full compliance check",
            "Local vector embeddings for regulation search"
        ],
        "main_endpoint": "/compliance/enhanced-csv-upload",
        "other_endpoints": {
            "health": "/health",
            "stats": "/stats",
            "compliance": "/compliance/*" if COMPLIANCE_AVAILABLE else "not_available"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    uptime = pd.Timestamp.now() - app_state.stats["uptime_start"]
    
    components = {
        "ml_model": "healthy" if app_state.ml_components and app_state.ml_components['loaded'] else "unhealthy",
        "preprocessor": "healthy" if app_state.preprocessor else "unhealthy",
        "compliance_module": "healthy" if COMPLIANCE_AVAILABLE else "unavailable"
    }
    
    overall_status = "healthy" if all(c in ["healthy", "unavailable"] for c in components.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=pd.Timestamp.now().isoformat(),
        uptime=str(uptime),
        version="2.0.0",
        components=components,
        stats=app_state.stats
    )

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    uptime = pd.Timestamp.now() - app_state.stats["uptime_start"]
    
    model_info = {}
    if app_state.ml_components and app_state.ml_components['loaded']:
        model_info = {
            "loaded": True,
            "threshold": app_state.ml_components['threshold'],
            "features": len(app_state.ml_components['feature_columns'])
        }
    else:
        model_info = {"loaded": False}
    
    return {
        "system": {
            "status": app_state.health_status,
            "uptime": str(uptime),
            "requests_processed": app_state.stats["requests_processed"],
            "version": "2.0.0"
        },
        "transactions": {
            "total_processed": app_state.stats["total_transactions"],
            "suspicious_detected": app_state.stats["suspicious_transactions"]
        },
        "ml_model": model_info,
        "compliance": {"available": COMPLIANCE_AVAILABLE}
    }

# Export preprocessor for use in compliance module
def get_preprocessor():
    """Get the preprocessor instance for use in other modules"""
    return app_state.preprocessor

def get_ml_components():
    """Get ML components for use in other modules"""
    return app_state.ml_components

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info",
        reload=False
    )
    