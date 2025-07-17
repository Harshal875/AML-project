"""
Professional Document Processor with Advanced Text Splitting
Replace your compliance/document_processor.py with this
"""
import os
import logging
from typing import List, Dict, Optional
from pathlib import Path

# LangChain imports for professional document processing
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfessionalDocumentProcessor:
    """
    Professional document processor for AML regulatory documents
    Features:
    - Multiple text splitting strategies
    - PDF and text file support
    - Metadata enrichment
    - Chunk optimization for embeddings
    - Quality validation
    """
    
    def __init__(self, storage_directory: str = "data/regulatory_docs"):
        self.storage_directory = storage_directory
        self._ensure_storage_exists()
        
        # Initialize multiple text splitters for different use cases
        self.splitters = self._initialize_splitters()
        
        logger.info("✅ Professional Document Processor initialized")
    
    def _ensure_storage_exists(self):
        """Create storage directory if it doesn't exist"""
        Path(self.storage_directory).mkdir(parents=True, exist_ok=True)
    
    def _initialize_splitters(self) -> Dict:
        """Initialize different text splitting strategies"""
        splitters = {
            # Main splitter: Optimized for regulatory documents
            "recursive": RecursiveCharacterTextSplitter(
                chunk_size=1000,      # Optimal for embeddings
                chunk_overlap=200,    # Maintain context
                length_function=len,
                separators=[
                    "\n\n",           # Paragraphs first
                    "\n",             # Lines
                    ". ",             # Sentences
                    "! ",             # Exclamations
                    "? ",             # Questions
                    ";",              # Semicolons
                    ",",              # Commas
                    " ",              # Words
                    ""                # Characters
                ]
            ),
            
            # Token-based splitter: For precise token control
            "token": TokenTextSplitter(
                chunk_size=800,       # Conservative token count
                chunk_overlap=100,
                model_name="gpt-3.5-turbo"  # For accurate token counting
            ),
            
            # Large chunk splitter: For comprehensive sections
            "large": RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=400,
                length_function=len,
                separators=["\n\n", "\n", ". "]
            ),
            
            # Small chunk splitter: For precise matching
            "small": RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ". ", " "]
            )
        }
        
        logger.info("✅ Text splitters initialized")
        return splitters
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """
        Save uploaded file with proper error handling
        
        Args:
            file_content: Raw file content
            filename: Name of the file
            
        Returns:
            str: Path to saved file
        """
        try:
            file_path = Path(self.storage_directory) / filename
            
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            logger.info(f"✅ Saved document: {filename} ({len(file_content)} bytes)")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Error saving file {filename}: {str(e)}")
            raise e
    
    def extract_text_from_pdf(self, file_path: str) -> List[Document]:
        """
        Extract text from PDF with proper document structure preservation
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of LangChain Documents with metadata
        """
        try:
            logger.info(f"Extracting text from PDF: {file_path}")
            
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Enrich metadata
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "source_type": "pdf",
                    "page_number": i + 1,
                    "file_path": file_path,
                    "extraction_method": "PyPDFLoader"
                })
            
            logger.info(f"✅ Extracted {len(documents)} pages from PDF")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error extracting from PDF {file_path}: {str(e)}")
            return []
    
    def extract_text_from_txt(self, file_path: str) -> List[Document]:
        """
        Extract text from text file with proper encoding handling
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of LangChain Documents
        """
        try:
            logger.info(f"Extracting text from TXT: {file_path}")
            
            # Try multiple encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.info(f"Successfully read file with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise ValueError("Could not decode file with any supported encoding")
            
            # Create document with metadata
            document = Document(
                page_content=content,
                metadata={
                    "source_type": "text",
                    "file_path": file_path,
                    "character_count": len(content),
                    "extraction_method": "TextLoader"
                }
            )
            
            logger.info(f"✅ Extracted text file ({len(content)} characters)")
            return [document]
            
        except Exception as e:
            logger.error(f"❌ Error extracting from TXT {file_path}: {str(e)}")
            return []
    
    def split_documents(self, documents: List[Document], 
                       strategy: str = "recursive") -> List[Document]:
        """
        Split documents using specified strategy
        
        Args:
            documents: List of documents to split
            strategy: Splitting strategy ('recursive', 'token', 'large', 'small')
            
        Returns:
            List of split document chunks
        """
        try:
            if strategy not in self.splitters:
                logger.warning(f"Unknown strategy '{strategy}', using 'recursive'")
                strategy = "recursive"
            
            splitter = self.splitters[strategy]
            logger.info(f"Splitting documents using '{strategy}' strategy")
            
            split_docs = splitter.split_documents(documents)
            
            # Enrich chunk metadata
            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    "chunk_id": i,
                    "splitting_strategy": strategy,
                    "chunk_length": len(doc.page_content),
                    "chunk_token_estimate": len(doc.page_content) // 4  # Rough estimate
                })
            
            logger.info(f"✅ Split into {len(split_docs)} chunks using {strategy} strategy")
            return split_docs
            
        except Exception as e:
            logger.error(f"❌ Error splitting documents: {str(e)}")
            return documents  # Return original if splitting fails
    
    def validate_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Validate and filter document chunks for quality
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of validated chunks
        """
        validated_chunks = []
        
        for chunk in chunks:
            content = chunk.page_content.strip()
            
            # Quality checks
            if len(content) < 50:  # Too short
                logger.debug(f"Skipping short chunk: {len(content)} characters")
                continue
            
            if len(content) > 5000:  # Too long for embeddings
                logger.debug(f"Chunk too long ({len(content)} chars), will be re-split")
                # Re-split long chunks
                small_splitter = self.splitters["small"]
                sub_chunks = small_splitter.split_documents([chunk])
                validated_chunks.extend(sub_chunks)
                continue
            
            # Check for meaningful content (not just whitespace/symbols)
            word_count = len(content.split())
            if word_count < 10:
                logger.debug(f"Skipping chunk with too few words: {word_count}")
                continue
            
            validated_chunks.append(chunk)
        
        logger.info(f"✅ Validated {len(validated_chunks)} chunks (filtered from {len(chunks)})")
        return validated_chunks
    
    def process_document(self, file_content: bytes, filename: str, 
                        metadata: Dict, splitting_strategy: str = "recursive") -> List[Dict]:
        """
        Complete document processing pipeline
        
        Args:
            file_content: Raw file content
            filename: Name of the file
            metadata: Document metadata
            splitting_strategy: Text splitting strategy to use
            
        Returns:
            List of processed chunks with enriched metadata
        """
        try:
            logger.info(f"Processing document: {filename}")
            
            # Step 1: Save file
            file_path = self.save_uploaded_file(file_content, filename)
            
            # Step 2: Extract text based on file type
            if filename.lower().endswith('.pdf'):
                documents = self.extract_text_from_pdf(file_path)
            elif filename.lower().endswith(('.txt', '.md')):
                documents = self.extract_text_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file type: {filename}")
            
            if not documents:
                raise ValueError("No content extracted from document")
            
            # Step 3: Split documents into chunks
            chunks = self.split_documents(documents, strategy=splitting_strategy)
            
            # Step 4: Validate chunks
            validated_chunks = self.validate_chunks(chunks)
            
            # Step 5: Convert to output format with enriched metadata
            processed_chunks = []
            for i, chunk in enumerate(validated_chunks):
                chunk_metadata = {
                    "text": chunk.page_content,
                    "source_file": filename,
                    "chunk_number": i,
                    "title": metadata.get("title", filename),
                    "jurisdiction": metadata.get("jurisdiction", "Unknown"),
                    "regulation_type": metadata.get("regulation_type", "Unknown"),
                    
                    # Enhanced metadata
                    "file_path": file_path,
                    "chunk_length": len(chunk.page_content),
                    "word_count": len(chunk.page_content.split()),
                    "splitting_strategy": splitting_strategy,
                    "processing_timestamp": str(pd.Timestamp.now()),
                    
                    # Merge original chunk metadata
                    **chunk.metadata
                }
                processed_chunks.append(chunk_metadata)
            
            logger.info(f"✅ Successfully processed {filename} into {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"❌ Error processing document {filename}: {str(e)}")
            raise e
    
    def get_processing_stats(self) -> Dict:
        """
        Get statistics about processed documents
        
        Returns:
            Dictionary with processing statistics
        """
        try:
            storage_path = Path(self.storage_directory)
            
            if not storage_path.exists():
                return {"status": "no_storage", "processed_files": 0}
            
            files = list(storage_path.glob("*"))
            pdf_files = list(storage_path.glob("*.pdf"))
            txt_files = list(storage_path.glob("*.txt"))
            
            stats = {
                "status": "active",
                "storage_directory": str(storage_path),
                "total_files": len(files),
                "pdf_files": len(pdf_files),
                "txt_files": len(txt_files),
                "supported_formats": [".pdf", ".txt", ".md"],
                "splitting_strategies": list(self.splitters.keys())
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting processing stats: {str(e)}")
            return {"status": "error", "error": str(e)}

# Add pandas import for timestamp
import pandas as pd

# Create global instance
document_processor = ProfessionalDocumentProcessor()