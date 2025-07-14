"""
Simple test without OpenAI to check basic setup
"""
import os

def test_basic_setup():
    """Test basic setup without API calls"""
    
    print("🧪 Testing basic compliance system setup...")
    
    # Test 1: Check if folders exist
    folders_to_check = ["data/regulatory_docs", "data/chroma_db"]
    for folder in folders_to_check:
        if os.path.exists(folder):
            print(f"✅ Folder exists: {folder}")
        else:
            os.makedirs(folder, exist_ok=True)
            print(f"✅ Created folder: {folder}")
    
    # Test 2: Check imports (without initializing OpenAI)
    try:
        import langchain
        import chromadb
        import openai
        print("✅ All AI packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
    
    # Test 3: Check .env file
    if os.path.exists(".env"):
        print("✅ .env file exists")
        with open(".env", "r") as f:
            content = f.read()
            if "OPENAI_API_KEY" in content:
                print("✅ OpenAI API key found in .env")
            else:
                print("❌ OpenAI API key not found in .env")
    else:
        print("❌ .env file not found")
    
    print("🎉 Basic setup test complete!")

if __name__ == "__main__":
    test_basic_setup()