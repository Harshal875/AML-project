"""
Simple vector store - lazy initialization to avoid SSL issues during testing
"""
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleVectorStore:
    def __init__(self):
        # Don't initialize OpenAI immediately - do it when needed
        self.embeddings = None
        self.vectorstore = None
        self.db_folder = "data/chroma_db"
        os.makedirs(self.db_folder, exist_ok=True)
        print("✅ Vector store configured (will initialize when needed)")
    
    def _initialize_if_needed(self):
        """Initialize OpenAI components only when actually needed"""
        if self.embeddings is None:
            try:
                from langchain_openai import OpenAIEmbeddings
                from langchain_community.vectorstores import Chroma
                
                # Check if API key is available
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment variables")
                
                self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
                self.vectorstore = Chroma(
                    collection_name="aml_regulations",
                    embedding_function=self.embeddings,
                    persist_directory=self.db_folder
                )
                print("✅ Vector store initialized with OpenAI")
                
            except Exception as e:
                print(f"❌ Error initializing vector store: {str(e)}")
                raise e
    
    def add_documents(self, processed_chunks: List[Dict]) -> bool:
        """Add processed document chunks to the searchable database"""
        try:
            self._initialize_if_needed()
            
            texts = [chunk["text"] for chunk in processed_chunks]
            metadatas = [
                {
                    "source_file": chunk["source_file"],
                    "chunk_number": chunk["chunk_number"],
                    "jurisdiction": chunk["jurisdiction"],
                    "regulation_type": chunk["regulation_type"],
                    "title": chunk["title"]
                }
                for chunk in processed_chunks
            ]
            
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            self.vectorstore.persist()
            
            print(f"✅ Added {len(texts)} chunks to vector database")
            return True
            
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            return False
    
    def search_regulations(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search for regulations related to the query"""
        try:
            self._initialize_if_needed()
            
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=max_results
            )
            
            formatted_results = []
            for document, score in results:
                result = {
                    "text": document.page_content,
                    "source_file": document.metadata.get("source_file", "Unknown"),
                    "title": document.metadata.get("title", "Unknown"),
                    "jurisdiction": document.metadata.get("jurisdiction", "Unknown"),
                    "regulation_type": document.metadata.get("regulation_type", "Unknown"),
                    "relevance_score": float(1 - score)
                }
                formatted_results.append(result)
            
            print(f"✅ Found {len(formatted_results)} relevant regulations")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching regulations: {str(e)}")
            return []
    
    def get_stats(self) -> Dict:
        """Get simple statistics about our regulation database"""
        try:
            if self.vectorstore is None:
                return {
                    "total_chunks": 0,
                    "status": "not_initialized"
                }
            
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "total_chunks": count,
                "status": "active" if count > 0 else "empty"
            }
        except:
            return {
                "total_chunks": 0,
                "status": "error"
            }

# Create global instance
vector_store = SimpleVectorStore()