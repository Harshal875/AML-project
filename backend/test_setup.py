"""
Simple test to make sure our compliance system is working
"""
import os
from compliance.document_processor import document_processor
from compliance.vector_store import vector_store

def test_basic_setup():
    """Test if our basic setup is working"""
    
    print("🧪 Testing compliance system setup...")
    
    # Test 1: Check if folders exist
    folders_to_check = ["data/regulatory_docs", "data/chroma_db"]
    for folder in folders_to_check:
        if os.path.exists(folder):
            print(f"✅ Folder exists: {folder}")
        else:
            print(f"❌ Folder missing: {folder}")
    
    # Test 2: Check vector store
    stats = vector_store.get_stats()
    print(f"✅ Vector store status: {stats}")
    
    # Test 3: Test imports
    try:
        import langchain
        import chromadb
        import openai
        print("✅ All AI packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
    
    print("🎉 Basic setup test complete!")

if __name__ == "__main__":
    test_basic_setup()