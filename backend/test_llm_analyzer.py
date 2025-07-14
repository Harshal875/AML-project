"""
Test the LLM analyzer with sample transaction (using mock responses for now)
"""
from compliance.llm_analyzer import llm_analyzer
from compliance.transaction_analyzer import transaction_analyzer
from compliance.vector_store import vector_store

def test_llm_analyzer():
    """Test LLM analysis with mock data (no OpenAI API calls)"""
    
    print("🧪 Testing LLM Analyzer with Mock Data...")
    
    # Sample transaction
    transaction = {
        'Amount': 15000,
        'Payment_type': 'Cash Deposit',
        'Sender_bank_location': 'USA',
        'Receiver_bank_location': 'USA',
        'Payment_currency': 'USD',
        'Received_currency': 'USD',
        'Time': '14:30:00'
    }
    
    ml_results = {
        'risk_score': 0.85,
        'is_suspicious': True,
        'risk_level': 'High'
    }
    
    # Step 1: Analyze transaction characteristics
    analysis = transaction_analyzer.analyze_transaction_for_compliance(transaction, ml_results)
    
    print(f"📊 Transaction Analysis:")
    print(f"   Compliance Priority: {analysis['compliance_priority']:.2f}")
    print(f"   Search Queries: {len(analysis['search_queries'])}")
    print(f"   Requires Check: {analysis['requires_compliance_check']}")
    
    # Step 2: Mock compliance analysis (no API calls)
    try:
        compliance_result = llm_analyzer.analyze_compliance(
            transaction_data=transaction,
            characteristics=analysis['characteristics'],
            search_queries=analysis['search_queries'],
            vector_store=vector_store,
            use_mock=True  # This forces mock response
        )
        
        print(f"\n🤖 Mock Compliance Analysis:")
        print(f"   Status: {compliance_result['compliance_status']}")
        print(f"   Risk Level: {compliance_result['risk_level']}")
        print(f"   Applicable Regulations: {len(compliance_result['applicable_regulations'])}")
        for reg in compliance_result['applicable_regulations']:
            print(f"     - {reg}")
        print(f"   Required Actions: {len(compliance_result['required_actions'])}")
        for action in compliance_result['required_actions']:
            print(f"     - {action}")
        print(f"   Source: {compliance_result['analysis_source']}")
        
        print(f"\n📄 Explanation: {compliance_result['explanation']}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")

def test_cross_border_transaction():
    """Test cross-border transaction analysis"""
    
    print("\n🌍 Testing Cross-Border Transaction...")
    
    transaction = {
        'Amount': 8500,
        'Payment_type': 'Cross-border',
        'Sender_bank_location': 'USA',
        'Receiver_bank_location': 'Pakistan',
        'Payment_currency': 'USD',
        'Received_currency': 'PKR',
        'Time': '23:45:00'
    }
    
    ml_results = {
        'risk_score': 0.6,
        'is_suspicious': False,
        'risk_level': 'Medium'
    }
    
    analysis = transaction_analyzer.analyze_transaction_for_compliance(transaction, ml_results)
    
    compliance_result = llm_analyzer.analyze_compliance(
        transaction_data=transaction,
        characteristics=analysis['characteristics'],
        search_queries=analysis['search_queries'],
        vector_store=vector_store,
        use_mock=True
    )
    
    print(f"   Status: {compliance_result['compliance_status']}")
    print(f"   Risk Level: {compliance_result['risk_level']}")
    print(f"   Explanation: {compliance_result['explanation']}")

if __name__ == "__main__":
    test_llm_analyzer()
    test_cross_border_transaction()
    print("\n✅ All tests complete!")