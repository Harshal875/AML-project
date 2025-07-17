"""
Quick OpenAI API Test for Windows
Save this as test_openai.py and run it
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai_connection():
    """Test basic OpenAI connection"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found in .env file")
            return False
        
        api_key = api_key.strip()
        print(f"✅ API Key found (starts with: {api_key[:10]}...)")
        
        # Test basic OpenAI import
        print("🔄 Testing OpenAI import...")
        import openai
        print("✅ OpenAI library imported successfully")
        
        # Test simple API call
        print("🔄 Testing OpenAI API connection...")
        client = openai.OpenAI(api_key=api_key)
        
        # Simple completion test
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'hello'"}],
            max_tokens=5
        )
        
        if response.choices[0].message.content:
            print("✅ OpenAI API connection successful!")
            print(f"Response: {response.choices[0].message.content}")
            return True
        else:
            print("❌ OpenAI API call failed - empty response")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI test failed: {str(e)}")
        print(f"Error type: {type(e)}")
        
        if "Invalid argument" in str(e):
            print("\n💡 Windows-specific suggestions:")
            print("1. Try running terminal as Administrator")
            print("2. Check Windows Firewall settings")
            print("3. Temporarily disable antivirus")
            print("4. Check if you're behind a corporate proxy")
        
        return False

if __name__ == "__main__":
    print("🧪 Testing OpenAI API Connection on Windows...")
    print("=" * 50)
    test_openai_connection()