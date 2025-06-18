# 🏦 AML Transaction Screening System

An AI-powered Anti-Money Laundering (AML) system that detects suspicious financial transactions using machine learning.

## 🎯 Key Features

- **Data Processing**: Handles 150K+ transaction datasets with severe class imbalance (956:168K ratio)
- **Feature Engineering**: Creates 25+ velocity-based features to detect smurfing and layering patterns
- **ML Pipeline**: XGBoost classifier with SMOTE balancing achieving 85% recall on suspicious transactions
- **Real-time Processing**: FastAPI backend for CSV batch uploads and instant predictions
- **Clean Interface**: React frontend for transaction upload and risk visualization

## 🔧 Technology Stack

**Backend:**
- FastAPI for REST API
- XGBoost for classification
- SMOTE for class balancing
- Pandas for data processing
- Scikit-learn for ML pipeline

**Frontend:**
- React.js
- Simple CSS styling
- File upload interface

## 🚀 Architecture

```
CSV Upload → Feature Engineering → XGBoost Model → Risk Predictions
     ↓              ↓                    ↓              ↓
Raw Data → 25+ Velocity Features → Trained Model → High/Medium/Low Risk
```

## 📊 Model Performance

- **85% Recall** on suspicious transaction detection
- **25+ Features** including transaction velocity, timing patterns, and behavioral analysis
- **Handles Imbalance** using SMOTE with 1:175 suspicious-to-normal ratio
- **Real-time Predictions** on uploaded transaction batches

## 🏗️ Project Structure

```
simple_aml/
├── backend/
│   └── main.py              # FastAPI application
├── frontend/
│   └── src/
│       ├── components/      # React components
│       └── App.js          # Main application
├── src/
│   ├── feature_engineering.py  # 25+ feature creation
│   ├── train_model.py          # XGBoost + SMOTE training
├── scripts/
│   └── prepare_data.py         # Data sampling
└── models/                     # Trained model files
```

## ⚡ Quick Start

### 1. Data Preparation
```bash
python scripts/prepare_data.py  # Sample dataset
```

### 2. Model Training
```bash
python src/train_model.py       # Train XGBoost with SMOTE
```

### 3. Start Backend
```bash
python backend/main.py          # API on localhost:8000
```

### 4. Start Frontend
```bash
cd frontend
npm start                       # React app on localhost:3000
```

### 5. Upload Transactions
- Open http://localhost:3000
- Upload CSV with transaction data
- View risk predictions and analysis

## 🎯 Key Features for Interview

### 1. **Data Challenge**
- Handled severe class imbalance (956 suspicious vs 168K normal transactions)
- Used SMOTE for intelligent synthetic minority oversampling
- Achieved 85% recall on the critical minority class

### 2. **Feature Engineering**
- **Velocity Features**: Transaction counts in 1h/24h/7d windows to detect smurfing
- **Behavioral Patterns**: Unique receivers, frequency analysis for layering detection
- **Time-based Features**: Hour, weekday, business hours for timing patterns
- **Amount Analysis**: Round amounts, just-under-threshold detection for structuring

### 3. **Production Considerations**
- Real-time CSV processing for batch transaction screening
- Proper data validation with Pydantic models
- Scalable FastAPI backend with clear error handling
- Clean React frontend for business users

### 4. **ML Pipeline**
- XGBoost for tabular data classification
- Feature scaling with StandardScaler
- Cross-validation and proper train/test splitting
- Model persistence for production deployment

## 📈 Business Impact

- **Automates** manual transaction review process
- **Reduces** false positive rates through ML-based scoring
- **Scales** to process large transaction volumes
- **Provides** clear risk categorization for compliance teams

## 🔍 Sample Detection Patterns

- **Smurfing**: Multiple transactions just under reporting thresholds
- **Layering**: Complex transaction chains with high velocity
- **Structuring**: Unusual timing patterns (night-time, weekend activity)
- **Geographic Risk**: Cross-border transactions to high-risk jurisdictions

---

**Built for production-scale AML compliance with focus on interpretability and performance.**