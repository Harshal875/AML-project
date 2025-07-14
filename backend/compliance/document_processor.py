"""
Simple document processor - handles uploading and chunking regulatory documents
This is like a librarian that organizes documents for easy searching
"""
import os
from typing import List
# UPDATED: Use new import path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SimpleDocumentProcessor:
    def __init__(self):
        # This splits long documents into smaller chunks
        # Think of it like breaking a book into chapters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Each chunk is about 1000 characters
            chunk_overlap=200,    # Overlap a bit to keep context
            separators=["\n\n", "\n", ".", " "]  # Split on paragraphs, sentences, etc.
        )
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """
        Save the uploaded file to our storage folder
        Returns the file path where it was saved
        """
        # Create the folder if it doesn't exist
        upload_folder = "data/regulatory_docs"
        os.makedirs(upload_folder, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(upload_folder, filename)
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        print(f"✅ Saved document: {filename}")
        return file_path
    
    def extract_text_from_pdf(self, file_path: str) -> List[str]:
        """
        Extract text from PDF and split into smaller chunks
        Returns list of text chunks
        """
        try:
            # Load the PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Extract just the text from each chunk
            text_chunks = [chunk.page_content for chunk in chunks]
            
            print(f"✅ Extracted {len(text_chunks)} chunks from {file_path}")
            return text_chunks
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")
            return []
    
    def process_document(self, file_content: bytes, filename: str, metadata: dict) -> List[str]:
        """
        Complete document processing pipeline:
        1. Save file
        2. Extract text
        3. Split into chunks
        4. Return chunks with metadata
        """
        # Step 1: Save the file
        file_path = self.save_uploaded_file(file_content, filename)
        
        # Step 2: Extract and chunk text
        text_chunks = self.extract_text_from_pdf(file_path)
        
        # Step 3: Add metadata to each chunk
        processed_chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_with_metadata = {
                "text": chunk_text,
                "source_file": filename,
                "chunk_number": i,
                "jurisdiction": metadata.get("jurisdiction", "Unknown"),
                "regulation_type": metadata.get("regulation_type", "Unknown"),
                "title": metadata.get("title", filename)
            }
            processed_chunks.append(chunk_with_metadata)
        
        print(f"✅ Processed {len(processed_chunks)} chunks with metadata")
        return processed_chunks

# Create a global instance
document_processor = SimpleDocumentProcessor()