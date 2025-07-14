"""
Simple API routes for compliance features
These are the endpoints our frontend will call
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List
import os

# Import our simple classes
from .document_processor import document_processor
from .vector_store import vector_store
from models.compliance_models import ComplianceQuery, ComplianceSearchResult

# Create a router for compliance endpoints
compliance_router = APIRouter(prefix="/compliance", tags=["compliance"])

@compliance_router.get("/health")
async def compliance_health():
    """
    Simple health check for compliance system
    """
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "vector_store": stats,
        "message": "Compliance system is running"
    }

@compliance_router.post("/upload-regulation")
async def upload_regulation(
    file: UploadFile = File(...),
    title: str = Form(...),
    jurisdiction: str = Form("Global"),
    regulation_type: str = Form("AML")
):
    """
    Upload a regulatory document (PDF)
    This processes the document and makes it searchable
    """
    try:
        # Check if file is PDF
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        print(f"📄 Processing document: {file.filename}")
        
        # Read file content
        file_content = await file.read()
        
        # Create metadata for this document
        metadata = {
            "title": title,
            "jurisdiction": jurisdiction,
            "regulation_type": regulation_type
        }
        
        # Process the document (extract text, split into chunks)
        processed_chunks = document_processor.process_document(
            file_content=file_content,
            filename=file.filename,
            metadata=metadata
        )
        
        # Add to vector database for searching
        success = vector_store.add_documents(processed_chunks)
        
        if success:
            return {
                "status": "success",
                "message": f"Successfully processed {file.filename}",
                "chunks_created": len(processed_chunks),
                "document_info": metadata
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to store document")
            
    except Exception as e:
        print(f"❌ Error uploading regulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@compliance_router.post("/search-regulations")
async def search_regulations(query_data: ComplianceQuery):
    """
    Search through regulatory documents
    Returns relevant regulation chunks
    """
    try:
        print(f"🔍 Searching for: {query_data.question}")
        
        # Search the vector database
        results = vector_store.search_regulations(
            query=query_data.question,
            max_results=5
        )
        
        # Convert to response format
        search_results = []
        for result in results:
            search_result = ComplianceSearchResult(
                regulation_text=result["text"],
                source_document=result["source_file"],
                relevance_score=result["relevance_score"]
            )
            search_results.append(search_result)
        
        return {
            "status": "success",
            "query": query_data.question,
            "results_found": len(search_results),
            "results": search_results
        }
        
    except Exception as e:
        print(f"❌ Error searching regulations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@compliance_router.get("/stats")
async def get_compliance_stats():
    """
    Get statistics about our regulatory database
    """
    try:
        # Get vector store statistics
        vector_stats = vector_store.get_stats()
        
        # Count uploaded documents
        docs_folder = "data/regulatory_docs"
        doc_count = 0
        if os.path.exists(docs_folder):
            doc_count = len([f for f in os.listdir(docs_folder) if f.endswith('.pdf')])
        
        return {
            "total_documents": doc_count,
            "total_chunks": vector_stats["total_chunks"],
            "database_status": vector_stats["status"],
            "storage_location": docs_folder
        }
        
    except Exception as e:
        print(f"❌ Error getting stats: {str(e)}")
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "database_status": "error",
            "error": str(e)
        }

# Simple test endpoint
@compliance_router.get("/test")
async def test_compliance():
    """
    Simple test endpoint to make sure everything is connected
    """
    return {
        "message": "Compliance system is working!",
        "endpoints": [
            "/compliance/upload-regulation",
            "/compliance/search-regulations", 
            "/compliance/stats",
            "/compliance/health"
        ]
    }

# Add this new endpoint to your existing routes.py

@compliance_router.post("/enhanced-csv-upload")
async def enhanced_csv_upload(file: UploadFile = File(...)):
    """
    Enhanced CSV upload with both ML screening AND compliance analysis
    This combines your existing ML model with compliance checking
    """
    try:
        # Import here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from main import MODEL_LOADED, model, scaler, feature_engineer, feature_columns, validate_csv_data
        from .compliance_engine import compliance_engine
        from models.compliance_models import EnhancedTransactionResult, EnhancedBatchResult
        import pandas as pd
        import numpy as np
        import io
        
        if not MODEL_LOADED:
            raise HTTPException(status_code=503, detail="ML model not available")
        
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files allowed")
        
        print(f"🚀 Processing enhanced CSV: {file.filename}")
        
        # Step 1: Read and validate CSV (same as existing)
        content = await file.read()
        csv_text = content.decode('utf-8-sig')
        
        try:
            df = pd.read_csv(io.StringIO(csv_text), sep=',')
            if len(df.columns) == 1:
                df = pd.read_csv(io.StringIO(csv_text), sep=';')
        except:
            df = pd.read_csv(io.StringIO(csv_text), sep=';')
        
        # Validate data
        validation_result = validate_csv_data(df)
        if not validation_result.is_valid:
            error_details = []
            for error in validation_result.errors[:5]:
                error_details.append(f"Row {error.row}: {error.error} in {error.column}")
            raise HTTPException(status_code=400, detail=f"CSV validation failed. Errors: {'; '.join(error_details)}")
        
        print(f"✅ CSV validated. Processing {len(df)} transactions with ML + Compliance...")
        
        # Step 2: ML Analysis (same as existing)
        df_features = feature_engineer.create_features(df)
        X = df_features[feature_columns].fillna(0)
        X_scaled = scaler.transform(X)
        
        ml_predictions = model.predict(X_scaled)
        ml_probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Step 3: Prepare ML results for compliance analysis
        ml_results_list = []
        for i in range(len(df)):
            prob = float(ml_probabilities[i])
            is_suspicious = bool(ml_predictions[i])
            
            if prob >= 0.7:
                risk_level = "High"
            elif prob >= 0.3:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            ml_result = {
                'risk_score': prob,
                'is_suspicious': is_suspicious,
                'risk_level': risk_level
            }
            ml_results_list.append(ml_result)
        
        # Step 4: Compliance Analysis (NEW!)
        compliance_results = compliance_engine.check_batch_compliance(df, ml_results_list)
        
        # Step 5: Combine ML + Compliance Results
        enhanced_results = []
        compliance_counts = {"compliant": 0, "needs_review": 0, "violation": 0}
        
        for i, (_, row) in enumerate(df.iterrows()):
            ml_result = ml_results_list[i]
            compliance_result = compliance_results[i]
            
            # Count compliance statuses
            comp_status = compliance_result.get('compliance_status', 'needs_review')
            compliance_counts[comp_status] = compliance_counts.get(comp_status, 0) + 1
            
            # Create enhanced result
            enhanced_result = EnhancedTransactionResult(
                # Basic transaction info
                sender_account=str(row['Sender_account']),
                receiver_account=str(row['Receiver_account']),
                amount=float(row['Amount']),
                
                # ML results
                risk_score=ml_result['risk_score'],
                is_suspicious=ml_result['is_suspicious'],
                risk_level=ml_result['risk_level'],
                
                # Compliance results
                compliance_status=compliance_result.get('compliance_status', 'needs_review'),
                compliance_risk_level=compliance_result.get('risk_level', 'medium'),
                required_actions=compliance_result.get('required_actions', []),
                applicable_regulations=compliance_result.get('applicable_regulations', []),
                compliance_explanation=compliance_result.get('explanation', 'Analysis unavailable')
            )
            enhanced_results.append(enhanced_result)
        
        # Step 6: Generate compliance summary
        compliance_summary = compliance_engine.get_compliance_summary(compliance_results)
        
        # Step 7: Calculate compliance rate
        total_transactions = len(df)
        compliance_rate = (compliance_counts.get('compliant', 0) / total_transactions * 100) if total_transactions > 0 else 0
        
        # Step 8: Create enhanced response
        ml_risk_counts = {"High": 0, "Medium": 0, "Low": 0}
        for result in ml_results_list:
            ml_risk_counts[result['risk_level']] += 1
        
        enhanced_response = EnhancedBatchResult(
            # ML stats (existing)
            total_transactions=total_transactions,
            suspicious_count=int(ml_predictions.sum()),
            high_risk_count=ml_risk_counts["High"],
            medium_risk_count=ml_risk_counts["Medium"],
            low_risk_count=ml_risk_counts["Low"],
            
            # Compliance stats (new)
            compliance_compliant_count=compliance_counts.get('compliant', 0),
            compliance_needs_review_count=compliance_counts.get('needs_review', 0),
            compliance_violation_count=compliance_counts.get('violation', 0),
            compliance_rate=compliance_rate,
            
            # Combined results
            predictions=enhanced_results,
            compliance_summary=compliance_summary
        )
        
        print(f"🎉 Enhanced analysis complete!")
        print(f"   ML: {int(ml_predictions.sum())} suspicious transactions")
        print(f"   Compliance: {compliance_counts['needs_review']} need review, {compliance_counts['violation']} violations")
        
        return enhanced_response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Enhanced processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced processing failed: {str(e)}")