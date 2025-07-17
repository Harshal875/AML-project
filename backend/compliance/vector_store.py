"""
WINDOWS WORKAROUND: Vector Store with Local Embeddings
This bypasses the OpenAI networking issue while still providing semantic search
Replace your compliance/vector_store.py with this
"""
import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import sentence transformers for better embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("✅ SentenceTransformers available for local embeddings")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.info("ℹ️ SentenceTransformers not available, using TF-IDF")

from dotenv import load_dotenv
load_dotenv()

class LocalEmbeddingsVectorStore:
    """
    Windows-compatible Vector Store using local embeddings
    - Uses SentenceTransformers if available (better semantic search)
    - Falls back to TF-IDF (still good for RAG)
    - No OpenAI API calls required
    """
    
    def __init__(self, persist_directory: str = "data/local_vectors"):
        self.persist_directory = persist_directory
        self.documents_storage = []
        self.embeddings_matrix = None
        self.vectorizer = None
        self.sentence_model = None
        self.embedding_type = "none"
        
        self._ensure_directory_exists()
        self._initialize_embedding_model()
        self._load_existing_data()
        
        logger.info(f"✅ Local Embeddings Vector Store initialized (using {self.embedding_type})")
    
    def _ensure_directory_exists(self):
        """Create storage directory"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            test_file = Path(self.persist_directory) / "test_write.txt"
            test_file.write_text("test")
            test_file.unlink()
            logger.info(f"✅ Directory {self.persist_directory} is writable")
        except Exception as e:
            logger.error(f"❌ Directory issue: {e}")
            raise
    
    def _initialize_embedding_model(self):
        """Initialize local embedding model"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info("🔄 Loading SentenceTransformer model...")
                # Use a small, fast model that works well offline
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_type = "sentence_transformer"
                logger.info("✅ SentenceTransformer model loaded (384 dimensions)")
            else:
                logger.info("🔄 Initializing TF-IDF vectorizer...")
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    lowercase=True,
                    strip_accents='unicode'
                )
                self.embedding_type = "tfidf"
                logger.info("✅ TF-IDF vectorizer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding model: {e}")
            self.embedding_type = "text_only"
    
    def _load_existing_data(self):
        """Load existing documents and embeddings"""
        try:
            docs_file = Path(self.persist_directory) / "documents.json"
            embeddings_file = Path(self.persist_directory) / "embeddings.pkl"
            vectorizer_file = Path(self.persist_directory) / "vectorizer.pkl"
            
            if docs_file.exists():
                with open(docs_file, 'r', encoding='utf-8') as f:
                    self.documents_storage = json.load(f)
                logger.info(f"✅ Loaded {len(self.documents_storage)} existing documents")
            
            if embeddings_file.exists() and self.documents_storage:
                with open(embeddings_file, 'rb') as f:
                    self.embeddings_matrix = pickle.load(f)
                logger.info(f"✅ Loaded embeddings matrix: {self.embeddings_matrix.shape}")
            
            if vectorizer_file.exists() and self.embedding_type == "tfidf":
                with open(vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("✅ Loaded TF-IDF vectorizer")
                
        except Exception as e:
            logger.error(f"❌ Error loading existing data: {e}")
            # Reset if loading fails
            self.documents_storage = []
            self.embeddings_matrix = None
    
    def _save_data(self):
        """Save documents and embeddings to disk"""
        try:
            docs_file = Path(self.persist_directory) / "documents.json"
            embeddings_file = Path(self.persist_directory) / "embeddings.pkl"
            vectorizer_file = Path(self.persist_directory) / "vectorizer.pkl"
            
            # Save documents
            with open(docs_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents_storage, f, ensure_ascii=False, indent=2)
            
            # Save embeddings
            if self.embeddings_matrix is not None:
                with open(embeddings_file, 'wb') as f:
                    pickle.dump(self.embeddings_matrix, f)
            
            # Save vectorizer
            if self.vectorizer is not None and self.embedding_type == "tfidf":
                with open(vectorizer_file, 'wb') as f:
                    pickle.dump(self.vectorizer, f)
            
            logger.info("✅ Data saved to disk")
            
        except Exception as e:
            logger.error(f"❌ Error saving data: {e}")
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        try:
            if self.embedding_type == "sentence_transformer":
                embeddings = self.sentence_model.encode(texts, show_progress_bar=False)
                return embeddings
            
            elif self.embedding_type == "tfidf":
                if self.embeddings_matrix is None:
                    # First time - fit the vectorizer
                    all_texts = [doc["text"] for doc in self.documents_storage] + texts
                    embeddings_matrix = self.vectorizer.fit_transform(all_texts)
                    # Return only the new embeddings
                    return embeddings_matrix[-len(texts):].toarray()
                else:
                    # Transform new texts using existing vectorizer
                    return self.vectorizer.transform(texts).toarray()
            
            else:
                # Fallback: return random embeddings for consistency
                return np.random.rand(len(texts), 100)
                
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            # Return random embeddings as fallback
            return np.random.rand(len(texts), 100)
    
    def add_documents(self, processed_chunks: List[Dict]) -> bool:
        """Add documents with local embeddings"""
        try:
            logger.info(f"🔄 Adding {len(processed_chunks)} chunks with {self.embedding_type} embeddings...")
            
            if not processed_chunks:
                logger.warning("⚠️ No chunks provided")
                return False
            
            # Extract texts for embedding
            texts = [chunk["text"] for chunk in processed_chunks]
            
            # Generate embeddings
            logger.info(f"🔄 Generating {self.embedding_type} embeddings...")
            new_embeddings = self._generate_embeddings(texts)
            
            # Add to storage
            start_idx = len(self.documents_storage)
            self.documents_storage.extend(processed_chunks)
            
            # Update embeddings matrix
            if self.embeddings_matrix is None:
                self.embeddings_matrix = new_embeddings
            else:
                self.embeddings_matrix = np.vstack([self.embeddings_matrix, new_embeddings])
            
            # Save to disk
            self._save_data()
            
            logger.info(f"✅ SEMANTIC STORAGE COMPLETE! Added {len(processed_chunks)} documents")
            logger.info(f"   Total documents: {len(self.documents_storage)}")
            logger.info(f"   Embeddings shape: {self.embeddings_matrix.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {str(e)}")
            return False
    
    def search_regulations(self, query: str, max_results: int = 5, 
                          filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Semantic search using local embeddings"""
        try:
            logger.info(f"🔍 LOCAL SEMANTIC SEARCH for: '{query}' (using {self.embedding_type})")
            
            if not self.documents_storage:
                logger.info("ℹ️ No documents in storage")
                return []
            
            if self.embeddings_matrix is None:
                logger.warning("⚠️ No embeddings available, using text search")
                return self._text_search_fallback(query, max_results)
            
            # Generate embedding for query
            query_embedding = self._generate_embeddings([query])
            
            if query_embedding.shape[0] == 0:
                logger.warning("⚠️ Failed to generate query embedding")
                return self._text_search_fallback(query, max_results)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:max_results]
            
            # Format results
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    doc = self.documents_storage[idx]
                    
                    # Apply metadata filter if provided
                    if filter_metadata:
                        match = True
                        for key, value in filter_metadata.items():
                            if doc.get(key) != value:
                                match = False
                                break
                        if not match:
                            continue
                    
                    result = {
                        "text": doc["text"],
                        "source_file": doc.get("source_file", "Unknown"),
                        "title": doc.get("title", "Unknown"),
                        "jurisdiction": doc.get("jurisdiction", "Unknown"),
                        "regulation_type": doc.get("regulation_type", "Unknown"),
                        "chunk_number": doc.get("chunk_number", 0),
                        "relevance_score": float(similarities[idx]),
                        "confidence": "high" if similarities[idx] > 0.7 else "medium" if similarities[idx] > 0.4 else "low",
                        "search_type": f"local_{self.embedding_type}"
                    }
                    results.append(result)
            
            logger.info(f"✅ Found {len(results)} semantic results (top score: {similarities[top_indices[0]]:.3f})")
            
            # Log top result for verification
            if results:
                top_result = results[0]
                logger.info(f"🎯 Top match: {top_result['text'][:100]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {str(e)}")
            return self._text_search_fallback(query, max_results)
    
    def _text_search_fallback(self, query: str, max_results: int) -> List[Dict]:
        """Fallback text search"""
        logger.info("🔄 Using fallback text search...")
        
        query_lower = query.lower()
        results = []
        
        for doc in self.documents_storage:
            text_lower = doc["text"].lower()
            score = 0
            for word in query_lower.split():
                if word in text_lower:
                    score += text_lower.count(word)
            
            if score > 0:
                results.append({
                    "text": doc["text"],
                    "source_file": doc.get("source_file", "Unknown"),
                    "title": doc.get("title", "Unknown"),
                    "jurisdiction": doc.get("jurisdiction", "Unknown"),
                    "regulation_type": doc.get("regulation_type", "Unknown"),
                    "chunk_number": doc.get("chunk_number", 0),
                    "relevance_score": min(score / 10, 1.0),
                    "confidence": "medium",
                    "search_type": "text_fallback"
                })
        
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:max_results]
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        try:
            total_chunks = len(self.documents_storage)
            
            stats = {
                "total_chunks": total_chunks,
                "status": "active" if total_chunks > 0 else "empty",
                "storage_path": self.persist_directory,
                "collection_name": "local_embeddings",
                "embedding_model": self.embedding_type,
                "search_type": f"local_{self.embedding_type}",
                "embeddings_shape": list(self.embeddings_matrix.shape) if self.embeddings_matrix is not None else None,
                "local_storage": True,
                "windows_compatible": True
            }
            
            if self.embedding_type == "sentence_transformer":
                stats["model_details"] = "all-MiniLM-L6-v2 (384 dimensions)"
            elif self.embedding_type == "tfidf":
                stats["model_details"] = "TF-IDF with 1000 features"
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting stats: {str(e)}")
            return {
                "total_chunks": len(self.documents_storage),
                "status": "error",
                "error": str(e),
                "storage_path": self.persist_directory
            }
    
    def search_by_metadata(self, metadata_filter: Dict, max_results: int = 10) -> List[Dict]:
        """Search by metadata"""
        try:
            results = []
            for doc in self.documents_storage:
                match = True
                for key, value in metadata_filter.items():
                    if doc.get(key) != value:
                        match = False
                        break
                
                if match:
                    results.append({
                        "text": doc["text"],
                        "source_file": doc.get("source_file", "Unknown"),
                        "title": doc.get("title", "Unknown"),
                        "jurisdiction": doc.get("jurisdiction", "Unknown"),
                        "regulation_type": doc.get("regulation_type", "Unknown"),
                        "chunk_number": doc.get("chunk_number", 0),
                        "relevance_score": 1.0,
                        "search_type": "metadata_filter"
                    })
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"❌ Metadata search failed: {str(e)}")
            return []
    
    def reset_database(self) -> bool:
        """Reset the vector database"""
        try:
            import shutil
            
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                self._ensure_directory_exists()
            
            # Reset instance variables
            self.documents_storage = []
            self.embeddings_matrix = None
            
            logger.info("✅ Vector database reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error resetting database: {str(e)}")
            return False

# Create global instance
vector_store = LocalEmbeddingsVectorStore()