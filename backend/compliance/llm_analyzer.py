"""
Simplified LLM Analyzer - Mock Compliance Analysis
Removed OpenAI dependencies, focused on mock responses for demo
"""
import json
import logging
from typing import Dict, List
import pandas as pd

logger = logging.getLogger(__name__)

class SimplifiedLLMAnalyzer:
    """
    Simplified LLM analyzer using intelligent mock responses
    Perfect for demo purposes - shows compliance logic without API dependencies
    """
    
    def __init__(self):
        logger.info("✅ Simplified LLM Analyzer initialized (mock mode)")
    
    def analyze_compliance(self, transaction_data: Dict, characteristics: Dict,
                          search_queries: List[str], vector_store) -> Dict:
        """
        Intelligent mock compliance analysis based on transaction characteristics
        """
        try:
            logger.info("🏛️ Running mock compliance analysis...")
            
            # Search for relevant regulations (this part works with your vector store)
            relevant_docs = self._search_relevant_regulations(search_queries, vector_store)
            
            # Generate intelligent mock response based on characteristics
            compliance_result = self._create_intelligent_mock_response(
                transaction_data, characteristics, relevant_docs
            )
            
            logger.info(f"✅ Mock compliance analysis complete - Status: {compliance_result['compliance_status']}")
            return compliance_result
            
        except Exception as e:
            logger.error(f"❌ Error in compliance analysis: {str(e)}")
            return self._create_error_response(str(e))
    
    def _search_relevant_regulations(self, search_queries: List[str], vector_store) -> List[Dict]:
        """Search for relevant regulations using vector store"""
        try:
            all_relevant_docs = []
            
            for query in search_queries[:3]:  # Limit to top 3 queries
                results = vector_store.search_regulations(query, max_results=2)
                all_relevant_docs.extend(results)
            
            # Remove duplicates and return top 5
            unique_docs = []
            seen = set()
            for doc in all_relevant_docs:
                identifier = doc.get('text', '')[:100]
                if identifier not in seen:
                    unique_docs.append(doc)
                    seen.add(identifier)
            
            return unique_docs[:5]
            
        except Exception as e:
            logger.warning(f"Regulation search failed: {str(e)}")
            return []
    
    def _create_intelligent_mock_response(self, transaction_data: Dict, 
                                        characteristics: Dict, relevant_docs: List[Dict]) -> Dict:
        """
        Create intelligent mock response based on transaction patterns
        This demonstrates real compliance logic
        """
        amount = transaction_data.get('Amount', 0)
        payment_type = transaction_data.get('Payment_type', 'Unknown')
        sender_country = transaction_data.get('Sender_bank_location', 'Unknown')
        receiver_country = transaction_data.get('Receiver_bank_location', 'Unknown')
        
        # Extract key characteristics
        is_large = characteristics.get('is_large_amount', False)
        is_cash = characteristics.get('is_cash_transaction', False)
        is_cross_border = characteristics.get('is_cross_border', False)
        is_high_risk = characteristics.get('involves_high_risk_country', False)
        is_structuring = characteristics.get('is_just_under_threshold', False)
        ml_risk_score = characteristics.get('ml_risk_score', 0)
        
        # Determine compliance status based on realistic AML rules
        if is_structuring:
            return {
                "compliance_status": "violation",
                "risk_level": "high",
                "confidence_score": 0.95,
                "applicable_regulations": [
                    "31 USC 5324 - Prohibition on Structuring", 
                    "BSA Structuring Rules"
                ],
                "required_actions": [
                    "File SAR immediately",
                    "Enhanced monitoring required",
                    "Consider criminal referral"
                ],
                "explanation": f"Transaction amount ${amount:,.2f} appears to be structured to avoid CTR reporting threshold. This violates federal anti-structuring laws.",
                "regulatory_citations": ["31 USC 5324", "31 CFR 103.18"],
                "risk_factors": ["Potential structuring pattern", "Amount just under $10,000"],
                "analysis_source": "intelligent_mock_analysis",
                "regulations_reviewed": len(relevant_docs)
            }
        
        elif is_large and is_cash:
            return {
                "compliance_status": "needs_review",
                "risk_level": "high",
                "confidence_score": 0.90,
                "applicable_regulations": [
                    "31 CFR 103.22 - Currency Transaction Reporting",
                    "BSA CTR Requirements"
                ],
                "required_actions": [
                    "File CTR within 15 days",
                    "Verify customer identity",
                    "Enhanced monitoring"
                ],
                "explanation": f"Large cash transaction of ${amount:,.2f} requires Currency Transaction Report (CTR) filing under Bank Secrecy Act regulations.",
                "regulatory_citations": ["31 CFR 103.22"],
                "risk_factors": ["Large cash amount", "CTR reporting threshold exceeded"],
                "analysis_source": "intelligent_mock_analysis",
                "regulations_reviewed": len(relevant_docs)
            }
        
        elif is_high_risk and amount > 5000:
            return {
                "compliance_status": "needs_review",
                "risk_level": "medium",
                "confidence_score": 0.75,
                "applicable_regulations": [
                    "31 CFR 103.18 - Enhanced Due Diligence",
                    "FATF High-Risk Country Guidelines"
                ],
                "required_actions": [
                    "Enhanced due diligence required",
                    "Additional documentation needed",
                    "Monitor for patterns"
                ],
                "explanation": f"Transaction involves high-risk jurisdiction ({sender_country} → {receiver_country}). Enhanced due diligence required per FATF recommendations.",
                "regulatory_citations": ["31 CFR 103.18"],
                "risk_factors": ["High-risk country involvement", "Significant amount"],
                "analysis_source": "intelligent_mock_analysis",
                "regulations_reviewed": len(relevant_docs)
            }
        
        elif is_cross_border and amount > 3000:
            return {
                "compliance_status": "compliant",
                "risk_level": "low",
                "confidence_score": 0.80,
                "applicable_regulations": [
                    "31 CFR 103.33 - Wire Transfer Recordkeeping",
                    "Travel Rule Requirements"
                ],
                "required_actions": [
                    "Standard wire transfer monitoring",
                    "Maintain proper records"
                ],
                "explanation": f"Cross-border wire transfer of ${amount:,.2f} appears compliant with standard monitoring requirements.",
                "regulatory_citations": ["31 CFR 103.33"],
                "risk_factors": ["Cross-border transaction"],
                "analysis_source": "intelligent_mock_analysis",
                "regulations_reviewed": len(relevant_docs)
            }
        
        elif ml_risk_score > 0.7:
            return {
                "compliance_status": "needs_review",
                "risk_level": "medium",
                "confidence_score": 0.85,
                "applicable_regulations": [
                    "31 CFR 103.18 - Suspicious Activity Reporting",
                    "ML Model Risk Assessment"
                ],
                "required_actions": [
                    "Review ML risk indicators",
                    "Consider SAR filing",
                    "Enhanced monitoring"
                ],
                "explanation": f"Machine learning model indicates elevated risk (score: {ml_risk_score:.3f}). Manual review recommended.",
                "regulatory_citations": ["31 CFR 103.18"],
                "risk_factors": ["High ML risk score", "Algorithmic detection"],
                "analysis_source": "intelligent_mock_analysis",
                "regulations_reviewed": len(relevant_docs)
            }
        
        else:
            return {
                "compliance_status": "compliant",
                "risk_level": "low",
                "confidence_score": 0.70,
                "applicable_regulations": [
                    "Standard AML Monitoring",
                    "Routine Transaction Processing"
                ],
                "required_actions": [
                    "Standard monitoring",
                    "Maintain transaction records"
                ],
                "explanation": f"Transaction appears to comply with standard AML requirements. No immediate action required.",
                "regulatory_citations": ["General AML Requirements"],
                "risk_factors": [],
                "analysis_source": "intelligent_mock_analysis",
                "regulations_reviewed": len(relevant_docs)
            }
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create error response for compliance failures"""
        return {
            "compliance_status": "error",
            "risk_level": "high",
            "confidence_score": 0.0,
            "applicable_regulations": [],
            "required_actions": ["System error - manual review required"],
            "explanation": f"Compliance analysis failed: {error_message}",
            "regulatory_citations": [],
            "analysis_source": "error_fallback"
        }

# Create global instance
llm_analyzer = SimplifiedLLMAnalyzer()