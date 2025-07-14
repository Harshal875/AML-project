"""
Simple data models for compliance features
Think of these as the "shape" of our data
"""
from pydantic import BaseModel
from typing import List, Optional

class RegulationUpload(BaseModel):
    """When someone uploads a regulation document"""
    filename: str
    title: str
    jurisdiction: str  # Like "USA", "EU", "Global"
    regulation_type: str  # Like "AML", "KYC", "BSA"

class ComplianceQuery(BaseModel):
    """When someone asks a question about regulations"""
    question: str

class TransactionCompliance(BaseModel):
    """Compliance check result for a transaction"""
    transaction_id: str
    compliance_status: str  # "compliant", "needs_review", "violation"
    risk_level: str  # "low", "medium", "high"
    required_actions: List[str]  # ["File SAR", "Enhanced monitoring"]
    applicable_regulations: List[str]  # Which regulations apply
    explanation: str  # Human-readable explanation

class ComplianceSearchResult(BaseModel):
    """Result when searching regulations"""
    regulation_text: str
    source_document: str
    relevance_score: float


# Add these new models to your existing file

class EnhancedTransactionResult(BaseModel):
    """Transaction result with both ML and compliance data"""
    sender_account: str
    receiver_account: str
    amount: float
    
    # ML Results (existing)
    risk_score: float
    is_suspicious: bool
    risk_level: str
    
    # NEW: Compliance Results
    compliance_status: str  # "compliant", "needs_review", "violation"
    compliance_risk_level: str  # "low", "medium", "high"
    required_actions: List[str]
    applicable_regulations: List[str]
    compliance_explanation: str

class EnhancedBatchResult(BaseModel):
    """Enhanced batch result with ML + compliance data"""
    # Existing ML stats
    total_transactions: int
    suspicious_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    
    # NEW: Compliance stats
    compliance_compliant_count: int
    compliance_needs_review_count: int
    compliance_violation_count: int
    compliance_rate: float
    
    # Enhanced predictions
    predictions: List[EnhancedTransactionResult]
    
    # NEW: Compliance summary
    compliance_summary: dict