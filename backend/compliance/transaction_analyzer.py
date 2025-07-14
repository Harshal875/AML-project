"""
Transaction Analyzer - Extracts compliance-relevant characteristics from transactions
Maps transaction patterns to regulatory search queries
"""
from typing import Dict, List
import pandas as pd
from datetime import datetime

class TransactionAnalyzer:
    def __init__(self):
        # Define high-risk countries for AML (customize as needed)
        self.high_risk_countries = [
            'Afghanistan', 'Albania', 'Barbados', 'Botswana', 'Burkina Faso',
            'Cambodia', 'Cayman Islands', 'Haiti', 'Iran', 'Jamaica', 
            'Jordan', 'Mali', 'Morocco', 'Myanmar', 'Nicaragua',
            'Pakistan', 'Panama', 'Philippines', 'Senegal', 'South Sudan',
            'Syria', 'Turkey', 'Uganda', 'Yemen', 'Zimbabwe'
        ]
    
    def extract_transaction_characteristics(self, transaction_row, ml_results=None):
        """
        Extract compliance-relevant characteristics from a single transaction
        """
        characteristics = {}
        
        # Basic transaction info
        amount = float(transaction_row['Amount'])
        payment_type = str(transaction_row['Payment_type'])
        sender_country = str(transaction_row['Sender_bank_location'])
        receiver_country = str(transaction_row['Receiver_bank_location'])
        
        # 1. Amount-based characteristics
        characteristics['amount'] = amount
        characteristics['is_large_amount'] = amount > 10000  # Over $10k threshold
        characteristics['is_just_under_threshold'] = 9000 <= amount < 10000  # Potential structuring
        characteristics['is_round_amount'] = amount % 1000 == 0  # Round amounts suspicious
        characteristics['is_very_large'] = amount > 50000  # Very large transactions
        
        # 2. Payment type characteristics
        characteristics['payment_type'] = payment_type
        characteristics['is_cash_transaction'] = payment_type in ['Cash Deposit', 'Cash Withdrawal']
        characteristics['is_wire_transfer'] = payment_type in ['ACH', 'Cross-border']
        characteristics['is_card_payment'] = payment_type in ['Credit card', 'Debit card']
        
        # 3. Geographic characteristics
        characteristics['sender_country'] = sender_country
        characteristics['receiver_country'] = receiver_country
        characteristics['is_cross_border'] = sender_country != receiver_country
        characteristics['involves_high_risk_country'] = (
            sender_country in self.high_risk_countries or 
            receiver_country in self.high_risk_countries
        )
        
        # 4. Currency exchange
        payment_currency = str(transaction_row['Payment_currency'])
        received_currency = str(transaction_row['Received_currency'])
        characteristics['involves_currency_exchange'] = payment_currency != received_currency
        
        # 5. Time-based characteristics
        try:
            transaction_time = datetime.strptime(transaction_row['Time'], '%H:%M:%S').time()
            hour = transaction_time.hour
            characteristics['is_off_hours'] = hour < 8 or hour > 18  # Outside business hours
            characteristics['is_night_transaction'] = hour < 6 or hour > 22  # Late night
        except:
            characteristics['is_off_hours'] = False
            characteristics['is_night_transaction'] = False
        
        # 6. ML model results (if provided)
        if ml_results:
            characteristics['ml_risk_score'] = ml_results.get('risk_score', 0)
            characteristics['ml_is_suspicious'] = ml_results.get('is_suspicious', False)
            characteristics['ml_risk_level'] = ml_results.get('risk_level', 'Low')
        
        return characteristics
    
    def generate_regulatory_queries(self, characteristics):
        """
        Convert transaction characteristics into regulatory search queries
        """
        queries = []
        
        # Amount-based queries
        if characteristics.get('is_large_amount'):
            queries.append("Currency Transaction Report CTR filing requirements over 10000 dollars")
            queries.append("large cash transaction reporting BSA Bank Secrecy Act")
        
        if characteristics.get('is_just_under_threshold'):
            queries.append("structuring transactions avoid reporting requirements smurfing")
            queries.append("suspicious activity report SAR structuring patterns")
        
        if characteristics.get('is_cash_transaction') and characteristics.get('amount', 0) > 3000:
            queries.append("cash transaction monitoring AML suspicious activity")
            queries.append("cash intensive business reporting requirements")
        
        # Geographic queries
        if characteristics.get('is_cross_border'):
            queries.append("cross border wire transfer reporting requirements")
            queries.append("international fund transfer FATF recommendations")
        
        if characteristics.get('involves_high_risk_country'):
            queries.append("high risk countries enhanced due diligence requirements")
            queries.append("FATF high risk jurisdictions AML requirements")
        
        # Payment type queries
        if characteristics.get('is_wire_transfer'):
            queries.append("wire transfer reporting requirements BSA")
            queries.append("electronic fund transfer monitoring rules")
        
        # ML-based queries
        if characteristics.get('ml_is_suspicious'):
            queries.append("suspicious activity report SAR filing requirements")
            queries.append("suspicious transaction patterns AML compliance")
        
        # Time-based queries
        if characteristics.get('is_off_hours'):
            queries.append("unusual timing transactions suspicious activity monitoring")
        
        # Currency exchange queries
        if characteristics.get('involves_currency_exchange'):
            queries.append("currency exchange transactions reporting requirements")
            queries.append("foreign exchange AML monitoring obligations")
        
        # Always include general AML query
        queries.append("anti money laundering transaction monitoring requirements")
        
        # Remove duplicates and return
        return list(set(queries))
    
    def analyze_transaction_for_compliance(self, transaction_row, ml_results=None):
        """
        Complete analysis: extract characteristics and generate queries
        """
        # Extract characteristics
        characteristics = self.extract_transaction_characteristics(transaction_row, ml_results)
        
        # Generate search queries
        queries = self.generate_regulatory_queries(characteristics)
        
        # Create compliance priority score (higher = more compliance attention needed)
        priority_score = self.calculate_compliance_priority(characteristics)
        
        return {
            'characteristics': characteristics,
            'search_queries': queries,
            'compliance_priority': priority_score,
            'requires_compliance_check': priority_score > 0.3
        }
    
    def calculate_compliance_priority(self, characteristics):
        """
        Calculate how much compliance attention this transaction needs (0-1 scale)
        """
        priority = 0.0
        
        # Amount factors
        if characteristics.get('is_very_large'):
            priority += 0.4
        elif characteristics.get('is_large_amount'):
            priority += 0.3
        elif characteristics.get('is_just_under_threshold'):
            priority += 0.5  # Structuring is very suspicious
        
        # Geographic factors
        if characteristics.get('involves_high_risk_country'):
            priority += 0.3
        if characteristics.get('is_cross_border'):
            priority += 0.2
        
        # Payment type factors
        if characteristics.get('is_cash_transaction'):
            priority += 0.2
        
        # ML factors
        if characteristics.get('ml_is_suspicious'):
            priority += 0.4
        
        ml_risk = characteristics.get('ml_risk_score', 0)
        priority += ml_risk * 0.3  # Add ML risk component
        
        # Time factors
        if characteristics.get('is_off_hours'):
            priority += 0.1
        
        # Currency factors
        if characteristics.get('involves_currency_exchange'):
            priority += 0.1
        
        # Cap at 1.0
        return min(priority, 1.0)

# Create global instance
transaction_analyzer = TransactionAnalyzer()