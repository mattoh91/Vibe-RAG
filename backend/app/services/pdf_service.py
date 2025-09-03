import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
import logging
import uuid
from datetime import datetime
import os

from app.core.config import settings
from app.models.document import Document, DocumentChunk, DocumentStatus

logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        
    async def process_pdf(self, file_path: str, original_filename: str) -> Optional[Document]:
        """Process a PDF file and extract text"""
        try:
            document_id = str(uuid.uuid4())
            
            # Create document record
            document = Document(
                id=document_id,
                filename=f"{document_id}.pdf",
                original_filename=original_filename,
                file_path=file_path,
                upload_time=datetime.utcnow(),
                status=DocumentStatus.PROCESSING
            )
            
            # Extract text from PDF
            chunks = await self._extract_text_chunks(file_path, document_id)
            
            # Update document with processing results
            document.page_count = len(set(chunk.page_number for chunk in chunks if chunk.page_number))
            document.chunk_count = len(chunks)
            document.status = DocumentStatus.COMPLETED
            
            logger.info(f"Successfully processed PDF: {original_filename} -> {len(chunks)} chunks")
            return document
            
        except Exception as e:
            logger.error(f"Failed to process PDF {original_filename}: {e}")
            return None
    
    async def _extract_text_chunks(self, file_path: str, document_id: str) -> List[DocumentChunk]:
        """Extract text chunks from PDF using hybrid recursive chunking"""
        chunks = []
        
        try:
            # Open PDF
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():
                    # Apply hybrid recursive chunking
                    page_chunks = self._hybrid_recursive_chunk(
                        text, 
                        page_num + 1,  # 1-based page numbering
                        document_id
                    )
                    chunks.extend(page_chunks)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise
        
        return chunks
    
    def _hybrid_recursive_chunk(
        self, 
        text: str, 
        page_number: int, 
        document_id: str
    ) -> List[DocumentChunk]:
        """Implement hybrid recursive chunking strategy"""
        chunks = []
        chunk_index = 0
        
        # Define separators in order of preference
        separators = [
            "\n\n\n",  # Multiple line breaks (section separators)
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentence endings
            ", ",      # Clause separators
            " ",       # Word boundaries
            ""         # Character level (fallback)
        ]
        
        def recursive_split(text: str, separators: List[str], start_idx: int = 0) -> List[str]:
            """Recursively split text using separators"""
            if len(text) <= settings.CHUNK_SIZE:
                return [text] if text.strip() else []
            
            if start_idx >= len(separators):
                # Fallback to character splitting
                return [text[i:i + settings.CHUNK_SIZE] 
                       for i in range(0, len(text), settings.CHUNK_SIZE)]
            
            separator = separators[start_idx]
            if separator == "":
                # Character level splitting
                return [text[i:i + settings.CHUNK_SIZE] 
                       for i in range(0, len(text), settings.CHUNK_SIZE)]
            
            parts = text.split(separator)
            result = []
            current_chunk = ""
            
            for part in parts:
                # Check if adding this part would exceed chunk size
                potential_chunk = current_chunk + separator + part if current_chunk else part
                
                if len(potential_chunk) <= settings.CHUNK_SIZE:
                    current_chunk = potential_chunk
                else:
                    # Save current chunk if it exists
                    if current_chunk:
                        result.append(current_chunk)
                    
                    # If this part is still too large, split it further
                    if len(part) > settings.CHUNK_SIZE:
                        result.extend(recursive_split(part, separators, start_idx + 1))
                        current_chunk = ""
                    else:
                        current_chunk = part
            
            # Add remaining chunk
            if current_chunk:
                result.append(current_chunk)
            
            return result
        
        # Split text into chunks
        text_chunks = recursive_split(text, separators)
        
        # Create DocumentChunk objects with overlap
        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue
                
            # Add overlap from previous chunk
            if i > 0 and settings.CHUNK_OVERLAP > 0:
                prev_chunk = text_chunks[i - 1]
                overlap_text = prev_chunk[-settings.CHUNK_OVERLAP:]
                chunk_text = overlap_text + " " + chunk_text
            
            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=chunk_index,
                text=chunk_text.strip(),
                page_number=page_number,
                metadata={
                    "chunk_size": len(chunk_text),
                    "has_overlap": i > 0 and settings.CHUNK_OVERLAP > 0
                }
            )
            
            chunks.append(chunk)
            chunk_index += 1
        
        return chunks
    
    async def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """Save uploaded file to disk"""
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = os.path.splitext(filename)[1]
            saved_filename = f"{file_id}{file_extension}"
            file_path = os.path.join(self.upload_dir, saved_filename)
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            logger.info(f"Saved uploaded file: {filename} -> {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file {filename}: {e}")
            raise
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete a file from disk"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False