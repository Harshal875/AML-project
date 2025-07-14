"""
LLM Analyzer - Uses AI to analyze transactions against regulatory requirements
Takes transaction data + relevant regulations → compliance assessment
"""
import json
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMAnalyzer:
    def __init__(self):
        # Don't initialize OpenAI client immediately - do it when needed
        self.client = None
        print("✅ LLM Analyzer configured (will initialize when needed)")
    
    def _initialize_openai_if_needed(self):
        """Initialize OpenAI client only when actually needed"""
        if self.client is None:
            try:
                import openai
                
                # Check if API key is available
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment variables")
                
                self.client = openai.OpenAI(api_key=api_key)
                print("✅ OpenAI client initialized")
                
            except Exception as e:
                print(f"❌ Error initializing OpenAI client: {str(e)}")
                raise e
    
    def search_relevant_regulations(self, search_queries: List[str], vector_store):
        """
        Search for relevant regulations using the provided queries
        """
        all_relevant_docs = []
        
        for query in search_queries:
            try:
                # Search vector database for each query
                results = vector_store.search_regulations(query, max_results=3)
                all_relevant_docs.extend(results)
            except Exception as e:
                print(f"⚠️ Error searching for query '{query}': {str(e)}")
                continue
        
        # Remove duplicates based on text content
        unique_docs = []
        seen_texts = set()
        
        for doc in all_relevant_docs:
            doc_text = doc.get('text', '')[:100]  # First 100 chars for deduplication
            if doc_text not in seen_texts:
                unique_docs.append(doc)
                seen_texts.add(doc_text)
        
        # Sort by relevance score and return top 10
        unique_docs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return unique_docs[:10]
    
    def format_regulatory_context(self, relevant_docs: List[Dict]) -> str:
        """
        Format regulatory documents for LLM prompt
        """
        if not relevant_docs:
            return "No specific regulations found."
        
        formatted_context = []
        
        for i, doc in enumerate(relevant_docs, 1):
            doc_text = doc.get('text', 'No content available')
            source = doc.get('source_file', 'Unknown source')
            title = doc.get('title', 'Unknown title')
            jurisdiction = doc.get('jurisdiction', 'Unknown')
            
            formatted_doc = f"""
REGULATION {i}:
Source: {source}
Title: {title}
Jurisdiction: {jurisdiction}
Content: {doc_text}
---
"""
            formatted_context.append(formatted_doc)
        
        return "\n".join(formatted_context)
    
    def analyze_compliance(self, transaction_data: Dict, characteristics: Dict, 
                          search_queries: List[str], vector_store, use_mock: bool = False) -> Dict:
        """
        Complete compliance analysis using LLM or mock data
        """
        try:
            # Step 1: Search for relevant regulations
            relevant_docs = self.search_relevant_regulations(search_queries, vector_store)
            
            if not relevant_docs and not use_mock:
                return self._create_no_regulations_response()
            
            # Step 2: If using mock or no real docs, create mock response
            if use_mock or not relevant_docs:
                return self._create_mock_compliance_response(transaction_data, characteristics)
            
            # Step 3: Format context for LLM
            regulatory_context = self.format_regulatory_context(relevant_docs)
            
            # Step 4: Create LLM prompt
            prompt = self.create_compliance_prompt(
                transaction_data, characteristics, regulatory_context
            )
            
            # Step 5: Get LLM analysis (this will initialize OpenAI when needed)
            llm_response = self.call_openai_for_compliance(prompt)
            
            # Step 6: Parse and validate response
            compliance_result = self.parse_llm_response(llm_response, relevant_docs)
            
            return compliance_result
            
        except Exception as e:
            print(f"❌ Error in compliance analysis: {str(e)}")
            return self._create_error_response(str(e))
    
    def _create_no_regulations_response(self) -> Dict:
        """Create response when no regulations are found"""
        return {
            "compliance_status": "needs_review",
            "risk_level": "medium",
            "applicable_regulations": [],
            "required_actions": ["Manual review required - no relevant regulations found"],
            "explanation": "No specific regulations found for this transaction pattern. Manual review recommended.",
            "regulatory_citations": [],
            "analysis_source": "no_regulations_found"
        }
    
    def _create_mock_compliance_response(self, transaction_data: Dict, characteristics: Dict) -> Dict:
        """Create mock compliance response for testing"""
        amount = transaction_data.get('Amount', 0)
        is_cash = characteristics.get('is_cash_transaction', False)
        is_large = characteristics.get('is_large_amount', False)
        is_cross_border = characteristics.get('is_cross_border', False)
        
        # Mock analysis based on characteristics
        if is_large and is_cash:
            return {
                "compliance_status": "needs_review",
                "risk_level": "high",
                "applicable_regulations": ["BSA Currency Transaction Reporting", "AML Cash Transaction Monitoring"],
                "required_actions": ["File CTR within 15 days", "Enhanced monitoring required"],
                "explanation": f"Large cash transaction of ${amount:,.2f} requires CTR filing and enhanced monitoring per BSA requirements.",
                "regulatory_citations": ["31 CFR 103.22", "BSA Section 5313"],
                "analysis_source": "mock_response"
            }
        elif is_cross_border:
            return {
                "compliance_status": "needs_review",
                "risk_level": "medium",
                "applicable_regulations": ["FATF International Transfer Guidelines", "Cross-Border Reporting Requirements"],
                "required_actions": ["Enhanced due diligence", "Document transfer purpose"],
                "explanation": f"Cross-border transaction requires enhanced due diligence per FATF recommendations.",
                "regulatory_citations": ["FATF Recommendation 16"],
                "analysis_source": "mock_response"
            }
        else:
            return {
                "compliance_status": "compliant",
                "risk_level": "low",
                "applicable_regulations": ["Standard AML Monitoring"],
                "required_actions": ["Standard monitoring"],
                "explanation": f"Transaction appears compliant with standard AML requirements.",
                "regulatory_citations": ["General AML Requirements"],
                "analysis_source": "mock_response"
            }
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create error response"""
        return {
            "compliance_status": "error",
            "risk_level": "high",
            "applicable_regulations": [],
            "required_actions": ["System error - manual review required"],
            "explanation": f"Analysis failed due to system error: {error_message}",
            "regulatory_citations": [],
            "analysis_source": "error_fallback"
        }
    
    def create_compliance_prompt(self, transaction_data: Dict, characteristics: Dict, 
                               regulatory_context: str) -> str:
        """Create structured prompt for LLM analysis"""
        prompt = f"""
You are an expert AML compliance analyst. Analyze this financial transaction for regulatory compliance.

TRANSACTION DETAILS:
- Amount: ${transaction_data.get('Amount', 0):,.2f}
- Payment Type: {transaction_data.get('Payment_type', 'Unknown')}
- From: {transaction_data.get('Sender_bank_location', 'Unknown')}
- To: {transaction_data.get('Receiver_bank_location', 'Unknown')}
- Payment Currency: {transaction_data.get('Payment_currency', 'Unknown')}
- Received Currency: {transaction_data.get('Received_currency', 'Unknown')}
- Time: {transaction_data.get('Time', 'Unknown')}

TRANSACTION CHARACTERISTICS:
- Is Large Amount (>$10K): {characteristics.get('is_large_amount', False)}
- Is Cash Transaction: {characteristics.get('is_cash_transaction', False)}
- Is Cross-Border: {characteristics.get('is_cross_border', False)}
- Involves High-Risk Country: {characteristics.get('involves_high_risk_country', False)}
- Just Under Threshold: {characteristics.get('is_just_under_threshold', False)}
- ML Risk Score: {characteristics.get('ml_risk_score', 0):.2f}
- ML Assessment: {characteristics.get('ml_risk_level', 'Unknown')}

RELEVANT REGULATIONS:
{regulatory_context}

RESPOND IN VALID JSON FORMAT:
{{
  "compliance_status": "compliant|needs_review|violation",
  "risk_level": "low|medium|high",
  "applicable_regulations": ["regulation1", "regulation2"],
  "required_actions": ["action1", "action2"],
  "explanation": "detailed explanation",
  "regulatory_citations": ["citation1", "citation2"]
}}
"""
        return prompt
    
    def call_openai_for_compliance(self, prompt: str) -> str:
        """Call OpenAI API for compliance analysis"""
        try:
            # This will initialize OpenAI client if needed
            self._initialize_openai_if_needed()
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert AML compliance analyst. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ OpenAI API error: {str(e)}")
            raise e
    
    def parse_llm_response(self, llm_response: str, relevant_docs: List[Dict]) -> Dict:
        """Parse and validate LLM response"""
        try:
            result = json.loads(llm_response)
            
            # Validate required fields
            required_fields = ["compliance_status", "risk_level", "applicable_regulations", 
                             "required_actions", "explanation", "regulatory_citations"]
            
            for field in required_fields:
                if field not in result:
                    result[field] = []
            
            # Add metadata
            result["analysis_source"] = "ai_automated"
            result["regulations_reviewed"] = len(relevant_docs)
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse LLM response: {str(e)}")
            return self._create_error_response("JSON parsing failed")

# Create global instance
llm_analyzer = LLMAnalyzer()