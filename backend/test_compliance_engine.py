"""
Test the complete compliance engine with sample transactions
"""
import pandas as pd
from compliance.compliance_engine import compliance_engine

def test_compliance_engine():
    """Test the complete compliance workflow"""
    
    print("🧪 Testing Complete Compliance Engine...")
    
    # Sample transactions
    transactions_data = [
        {
            'Amount': 15000,
            'Payment_type': 'Cash Deposit',
            'Sender_bank_location': 'USA',
            'Receiver_bank_location': 'USA',
            'Payment_currency': 'USD',
            'Received_currency': 'USD',
            
            'Time': '14:30:00'
        },
        {
            'Amount': 8500,
            'Payment_type': 'Cross-border',
            'Sender_bank_location': 'USA',
            'Receiver_bank_location': 'Pakistan',
            'Payment_currency': 'USD',
            'Received_currency': 'PKR',
            'Time': '23:45:00'
        },
        {
            'Amount': 1500,
            'Payment_type': 'Credit card',
            'Sender_bank_location': 'USA',
            'Receiver_bank_location': 'USA',
            'Payment_currency': 'USD',
            'Received_currency': 'USD',
            'Time': '10:30:00'
        }
    ]
    
    # Corresponding ML results
    ml_results = [
        {'risk_score': 0.85, 'is_suspicious': True, 'risk_level': 'High'},
        {'risk_score': 0.6, 'is_suspicious': False, 'risk_level': 'Medium'},
        {'risk_score': 0.1, 'is_suspicious': False, 'risk_level': 'Low'}
    ]
    
    # Create DataFrame
    df = pd.DataFrame(transactions_data)
    
    # Test batch compliance checking
    compliance_results = compliance_engine.check_batch_compliance(df, ml_results)
    
    print(f"\n📊 Compliance Results:")
    for i, result in enumerate(compliance_results):
        print(f"\n   Transaction {i+1}:")
        print(f"   Amount: ${transactions_data[i]['Amount']:,}")
        print(f"   Type: {transactions_data[i]['Payment_type']}")
        print(f"   ML Risk: {ml_results[i]['risk_level']}")
        print(f"   Compliance Status: {result['compliance_status']}")
        print(f"   Compliance Risk: {result['risk_level']}")
        print(f"   Required Actions: {len(result['required_actions'])}")
    
    # Test summary generation
    summary = compliance_engine.get_compliance_summary(compliance_results)
    
    print(f"\n📈 Compliance Summary:")
    print(f"   Total Transactions: {summary['total_transactions']}")
    print(f"   Compliant: {summary['status_breakdown']['compliant']}")
    print(f"   Needs Review: {summary['status_breakdown']['needs_review']}")
    print(f"   Violations: {summary['status_breakdown']['violation']}")
    print(f"   Compliance Rate: {summary['compliance_rate']}%")
    print(f"   High Risk Rate: {summary['high_risk_rate']}%")
    
    print(f"\n✅ Compliance engine test complete!")

if __name__ == "__main__":
    test_compliance_engine()
