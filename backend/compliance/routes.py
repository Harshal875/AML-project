"""
Fixed Compliance Routes - All Numpy Serialization Issues Resolved
Replace your compliance/routes.py with this complete version
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import numpy as np
import io
import logging
from pydantic import BaseModel
from pathlib import Path

# Import professional components
from .compliance_engine import compliance_engine
from .document_processor import document_processor
from .vector_store import vector_store

# Configure logging
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj

# Pydantic Models
class ComplianceQuery(BaseModel):
    question: str
    max_results: Optional[int] = 5

class ComplianceSearchResult(BaseModel):
    regulation_text: str
    source_document: str
    title: str
    jurisdiction: str
    regulation_type: str
    relevance_score: float

class DocumentUploadResponse(BaseModel):
    status: str
    message: str
    chunks_created: int
    document_info: dict

class EnhancedTransactionResult(BaseModel):
    sender_account: str
    receiver_account: str
    amount: float
    risk_score: float
    is_suspicious: bool
    risk_level: str
    compliance_status: str
    compliance_risk_level: str
    required_actions: List[str]
    applicable_regulations: List[str]
    compliance_explanation: str
    confidence_score: Optional[float] = None

class EnhancedBatchResult(BaseModel):
    total_transactions: int
    suspicious_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    compliance_compliant_count: int
    compliance_needs_review_count: int
    compliance_violation_count: int
    compliance_rate: float
    predictions: List[EnhancedTransactionResult]
    compliance_summary: dict
    processing_metadata: dict

class SystemHealth(BaseModel):
    status: str
    components: dict
    database_stats: dict

# Create router
compliance_router = APIRouter(prefix="/compliance", tags=["compliance"])

def validate_csv_data(df):
    """Basic CSV validation"""
    required_columns = ['Time', 'Date', 'Sender_account', 'Receiver_account', 'Amount', 
                       'Payment_currency', 'Received_currency', 'Sender_bank_location', 
                       'Receiver_bank_location', 'Payment_type']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Basic data validation
    if df['Amount'].isna().any() or (df['Amount'] <= 0).any():
        raise ValueError("Invalid amounts detected")
    
    return True

@compliance_router.get("/health", response_model=SystemHealth)
async def compliance_health_check():
    """Health check for compliance system components"""
    try:
        # Import ML components from main
        from main import get_ml_components
        ml_components = get_ml_components()
        
        vector_stats = vector_store.get_stats()
        processor_stats = document_processor.get_processing_stats()
        
        components = {
            "ml_model": "loaded" if ml_components and ml_components.get('loaded') else "error",
            "vector_database": vector_stats.get("status", "unknown"),
            "document_processor": processor_stats.get("status", "unknown"),
            "compliance_engine": "active"
        }
        
        overall_status = "healthy" if all(c in ["loaded", "active", "healthy"] for c in components.values()) else "degraded"
        
        return SystemHealth(
            status=overall_status,
            components=components,
            database_stats=vector_stats
        )
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        return SystemHealth(
            status="error",
            components={"error": str(e)},
            database_stats={}
        )

@compliance_router.post("/upload-regulation", response_model=DocumentUploadResponse)
async def upload_regulation_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    jurisdiction: str = Form("Global"),
    regulation_type: str = Form("AML")
):
    """Upload regulatory document with RAG processing"""
    try:
        logger.info(f"📄 Processing document upload: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.txt', '.md')):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF, TXT, and MD files are supported"
            )
        
        # Read and process file
        file_content = await file.read()
        
        metadata = {
            "title": title,
            "jurisdiction": jurisdiction,
            "regulation_type": regulation_type
        }
        
        # Process document
        processed_chunks = document_processor.process_document(
            file_content=file_content,
            filename=file.filename,
            metadata=metadata,
            splitting_strategy="recursive"
        )
        
        if not processed_chunks:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from the document"
            )
        
        # Add to vector database
        success = vector_store.add_documents(processed_chunks)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to add documents to vector database"
            )
        
        logger.info(f"✅ Successfully processed {file.filename} - {len(processed_chunks)} chunks")
        
        return DocumentUploadResponse(
            status="success",
            message=f"Successfully processed {file.filename}",
            chunks_created=len(processed_chunks),
            document_info={
                "title": title,
                "jurisdiction": jurisdiction,
                "regulation_type": regulation_type,
                "file_size": len(file_content)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error uploading regulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@compliance_router.post("/search-regulations")
async def search_regulations_semantic(query_data: ComplianceQuery):
    """Semantic search through regulations using vector database"""
    try:
        logger.info(f"🔍 Searching for: {query_data.question}")
        
        search_results = vector_store.search_regulations(
            query=query_data.question,
            max_results=query_data.max_results
        )
        
        if not search_results:
            return {
                "status": "success",
                "query": query_data.question,
                "results_found": 0,
                "results": [],
                "message": "No relevant regulations found. Upload regulatory documents first."
            }
        
        formatted_results = []
        for result in search_results:
            search_result = ComplianceSearchResult(
                regulation_text=result["text"],
                source_document=result["source_file"],
                title=result.get("title", "Unknown"),
                jurisdiction=result.get("jurisdiction", "Unknown"),
                regulation_type=result.get("regulation_type", "Unknown"),
                relevance_score=convert_numpy_types(result["relevance_score"])
            )
            formatted_results.append(search_result)
        
        logger.info(f"✅ Found {len(formatted_results)} relevant regulations")
        
        return {
            "status": "success",
            "query": query_data.question,
            "results_found": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"❌ Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@compliance_router.post("/enhanced-csv-upload", response_model=EnhancedBatchResult)
async def enhanced_csv_upload(file: UploadFile = File(...)):
    """
    MAIN ENDPOINT: Enhanced CSV upload with full ML + RAG compliance analysis
    This combines all functionality from basic upload + compliance analysis
    """
    try:
        # Import shared components from main
        from main import get_ml_components, get_preprocessor
        
        ml_components = get_ml_components()
        preprocessor = get_preprocessor()
        
        if not ml_components or not ml_components.get('loaded'):
            raise HTTPException(
                status_code=503, 
                detail=f"ML model not available: {ml_components.get('error', 'Unknown error') if ml_components else 'Not loaded'}"
            )
        
        if not preprocessor:
            raise HTTPException(status_code=503, detail="Preprocessor not available")
        
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files allowed")
        
        logger.info(f"🚀 Processing enhanced CSV: {file.filename}")
        
        # Read CSV with encoding handling
        content = await file.read()
        
        # Try different encodings
        for encoding in ['utf-8-sig', 'utf-8', 'latin-1']:
            try:
                csv_text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise HTTPException(status_code=400, detail="Could not decode CSV file")
        
        # Try different separators
        try:
            df = pd.read_csv(io.StringIO(csv_text), sep=',')
            if len(df.columns) == 1:
                df = pd.read_csv(io.StringIO(csv_text), sep=';')
        except:
            df = pd.read_csv(io.StringIO(csv_text), sep=';')
        
        # Validate CSV
        validate_csv_data(df)
        
        logger.info(f"✅ CSV validated. Processing {len(df)} transactions...")
        
        # ML Analysis using shared preprocessor
        df_features = preprocessor.create_features(df.copy())
        X = df_features[ml_components['feature_columns']].fillna(0)
        X_scaled = ml_components['scaler'].transform(X)
        
        # Get ML predictions
        ml_probabilities = ml_components['model'].predict_proba(X_scaled)[:, 1]
        ml_predictions = (ml_probabilities >= ml_components['threshold']).astype(int)
        
        # Prepare ML results for compliance analysis - WITH NUMPY CONVERSION
        ml_results_list = []
        for i in range(len(df)):
            prob = convert_numpy_types(ml_probabilities[i])
            is_suspicious = convert_numpy_types(ml_predictions[i])
            
            if prob >= 0.7:
                risk_level = "High"
            elif prob >= 0.3:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            ml_results_list.append({
                'risk_score': prob,
                'is_suspicious': bool(is_suspicious),
                'risk_level': risk_level
            })
        
        # COMPLIANCE ANALYSIS using RAG
        logger.info("🏛️ Running compliance analysis with RAG...")
        compliance_results = compliance_engine.check_batch_compliance(df, ml_results_list)
        
        # Combine ML + Compliance Results - WITH NUMPY CONVERSION
        enhanced_results = []
        compliance_counts = {"compliant": 0, "needs_review": 0, "violation": 0, "error": 0}
        risk_counts = {"High": 0, "Medium": 0, "Low": 0}
        
        for i, (_, row) in enumerate(df.iterrows()):
            ml_result = ml_results_list[i]
            compliance_result = compliance_results[i]
            
            # Count results
            comp_status = compliance_result.get('compliance_status', 'needs_review')
            compliance_counts[comp_status] = compliance_counts.get(comp_status, 0) + 1
            risk_counts[ml_result['risk_level']] += 1
            
            # Create enhanced result - ALL VALUES CONVERTED
            enhanced_result = EnhancedTransactionResult(
                sender_account=str(row['Sender_account']),
                receiver_account=str(row['Receiver_account']),
                amount=convert_numpy_types(row['Amount']),
                risk_score=convert_numpy_types(ml_result['risk_score']),
                is_suspicious=convert_numpy_types(ml_result['is_suspicious']),
                risk_level=ml_result['risk_level'],
                compliance_status=compliance_result.get('compliance_status', 'needs_review'),
                compliance_risk_level=compliance_result.get('risk_level', 'medium'),
                required_actions=compliance_result.get('required_actions', []),
                applicable_regulations=compliance_result.get('applicable_regulations', []),
                compliance_explanation=compliance_result.get('explanation', 'Analysis completed'),
                confidence_score=convert_numpy_types(compliance_result.get('confidence_score')) if compliance_result.get('confidence_score') is not None else None
            )
            enhanced_results.append(enhanced_result)
        
        # Generate summary
        compliance_summary = compliance_engine.get_compliance_summary(compliance_results)
        
        # Calculate rates - WITH NUMPY CONVERSION
        total_transactions = len(df)
        compliance_rate = convert_numpy_types((compliance_counts.get('compliant', 0) / total_transactions * 100) if total_transactions > 0 else 0)
        
        # Processing metadata
        processing_metadata = {
            "ml_threshold": convert_numpy_types(ml_components['threshold']),
            "features_used": len(ml_components['feature_columns']),
            "compliance_engine": "rag_enabled",
            "regulations_in_db": vector_store.get_stats().get("total_chunks", 0),
            "processing_timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Create response - ALL VALUES CONVERTED
        response = EnhancedBatchResult(
            total_transactions=total_transactions,
            suspicious_count=convert_numpy_types(int(ml_predictions.sum())),
            high_risk_count=risk_counts["High"],
            medium_risk_count=risk_counts["Medium"],
            low_risk_count=risk_counts["Low"],
            compliance_compliant_count=compliance_counts.get('compliant', 0),
            compliance_needs_review_count=compliance_counts.get('needs_review', 0),
            compliance_violation_count=compliance_counts.get('violation', 0),
            compliance_rate=compliance_rate,
            predictions=enhanced_results,
            compliance_summary=convert_numpy_types(compliance_summary),
            processing_metadata=convert_numpy_types(processing_metadata)
        )
        
        logger.info(f"🎉 Enhanced analysis complete!")
        logger.info(f"   ML: {int(ml_predictions.sum())} suspicious | Compliance: {compliance_counts}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Enhanced processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@compliance_router.get("/stats")
async def get_compliance_stats():
    """Get comprehensive compliance system statistics"""
    try:
        from main import get_ml_components
        
        vector_stats = vector_store.get_stats()
        processor_stats = document_processor.get_processing_stats()
        ml_components = get_ml_components()
        
        return {
            "system_status": "active",
            "vector_database": convert_numpy_types(vector_stats),
            "document_processor": convert_numpy_types(processor_stats),
            "ml_model": {
                "loaded": ml_components.get('loaded', False) if ml_components else False,
                "threshold": convert_numpy_types(ml_components.get('threshold', 'N/A')) if ml_components else 'N/A',
                "features": len(ml_components.get('feature_columns', [])) if ml_components and ml_components.get('loaded') else 0
            },
            "main_endpoint": "/compliance/enhanced-csv-upload",
            "capabilities": [
                "PDF document processing",
                "Local vector embeddings search",
                "ML risk assessment (XGBoost + 33 features)", 
                "RAG compliance analysis",
                "Intelligent mock compliance responses"
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting stats: {str(e)}")
        return {"status": "error", "error": str(e)}