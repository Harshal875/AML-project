"""
Main Compliance Engine - Orchestrates the complete compliance checking process
Integrates transaction analysis + LLM analysis + vector search
"""
from typing import Dict, List, Optional
from .transaction_analyzer import transaction_analyzer
from .llm_analyzer import llm_analyzer
from .vector_store import vector_store

class ComplianceEngine:
    def __init__(self):
        print("✅ Compliance Engine initialized")
    
    def check_transaction_compliance(self, transaction_row, ml_results: Optional[Dict] = None) -> Dict:
        """
        Complete compliance check for a single transaction
        
        Args:
            transaction_row: pandas row or dict with transaction data
            ml_results: dict with ML risk assessment results
            
        Returns:
            dict with complete compliance assessment
        """
        try:
            # Step 1: Analyze transaction characteristics
            analysis = transaction_analyzer.analyze_transaction_for_compliance(
                transaction_row=transaction_row,
                ml_results=ml_results
            )
            
            # Step 2: Determine if compliance check is needed
            if not analysis['requires_compliance_check']:
                return self._create_low_priority_response(transaction_row, analysis)
            
            # Step 3: Perform LLM compliance analysis
            compliance_result = llm_analyzer.analyze_compliance(
                transaction_data=transaction_row,
                characteristics=analysis['characteristics'],
                search_queries=analysis['search_queries'],
                vector_store=vector_store
            )
            
            # Step 4: Add transaction analysis metadata
            compliance_result['transaction_analysis'] = {
                'compliance_priority': analysis['compliance_priority'],
                'search_queries_count': len(analysis['search_queries']),
                'characteristics_detected': len([k for k, v in analysis['characteristics'].items() if v])
            }
            
            return compliance_result
            
        except Exception as e:
            print(f"❌ Compliance check failed for transaction: {str(e)}")
            return self._create_error_response(str(e))
    
    def check_batch_compliance(self, transactions_df, ml_results_list: List[Dict]) -> List[Dict]:
        """
        Check compliance for a batch of transactions
        
        Args:
            transactions_df: pandas DataFrame with transactions
            ml_results_list: list of ML results for each transaction
            
        Returns:
            list of compliance results
        """
        compliance_results = []
        
        print(f"🔍 Running compliance checks on {len(transactions_df)} transactions...")
        
        for i, (_, transaction_row) in enumerate(transactions_df.iterrows()):
            # Get corresponding ML results
            ml_result = ml_results_list[i] if i < len(ml_results_list) else None
            
            # Check compliance for this transaction
            compliance_result = self.check_transaction_compliance(
                transaction_row=transaction_row,
                ml_results=ml_result
            )
            
            compliance_results.append(compliance_result)
        
        print(f"✅ Compliance checks complete")
        return compliance_results
    
    def _create_low_priority_response(self, transaction_row, analysis) -> Dict:
        """Create response for low-priority transactions"""
        return {
            "compliance_status": "compliant",
            "risk_level": "low",
            "applicable_regulations": ["Standard AML Monitoring"],
            "required_actions": ["Standard monitoring"],
            "explanation": f"Transaction priority score {analysis['compliance_priority']:.2f} below threshold. Standard monitoring applies.",
            "regulatory_citations": ["General AML Requirements"],
            "analysis_source": "low_priority_automatic",
            "transaction_analysis": {
                'compliance_priority': analysis['compliance_priority'],
                'search_queries_count': len(analysis['search_queries']),
                'characteristics_detected': len([k for k, v in analysis['characteristics'].items() if v])
            }
        }
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create error response"""
        return {
            "compliance_status": "error",
            "risk_level": "high",
            "applicable_regulations": [],
            "required_actions": ["System error - manual review required"],
            "explanation": f"Compliance analysis failed: {error_message}",
            "regulatory_citations": [],
            "analysis_source": "error_fallback"
        }
    
    def get_compliance_summary(self, compliance_results: List[Dict]) -> Dict:
        """
        Generate summary statistics for a batch of compliance results
        """
        total = len(compliance_results)
        if total == 0:
            return {"total": 0, "summary": "No transactions analyzed"}
        
        # Count statuses
        status_counts = {}
        risk_counts = {}
        
        for result in compliance_results:
            status = result.get('compliance_status', 'unknown')
            risk = result.get('risk_level', 'unknown')
            
            status_counts[status] = status_counts.get(status, 0) + 1
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        # Calculate percentages
        summary = {
            "total_transactions": total,
            "status_breakdown": {
                "compliant": status_counts.get('compliant', 0),
                "needs_review": status_counts.get('needs_review', 0),
                "violation": status_counts.get('violation', 0),
                "error": status_counts.get('error', 0)
            },
            "risk_breakdown": {
                "low": risk_counts.get('low', 0),
                "medium": risk_counts.get('medium', 0),
                "high": risk_counts.get('high', 0)
            },
            "compliance_rate": round((status_counts.get('compliant', 0) / total) * 100, 1),
            "high_risk_rate": round((risk_counts.get('high', 0) / total) * 100, 1)
        }
        
        return summary

# Create global instance
compliance_engine = ComplianceEngine()