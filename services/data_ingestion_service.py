"""
Data ingestion service for PDF text extraction and chunking
"""
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from io import BytesIO

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

from core.config import settings
from services.embeddings_service import EmbeddingsService
from services.vectordb_service import VectorDBService
from utils.logger import get_logger

logger = get_logger(__name__)

class DataIngestionService:
    def __init__(self):
        self.text_splitter = self._initialize_text_splitter()
        self.embeddings_service = EmbeddingsService()
        self.vectordb_service = VectorDBService()

    def _initialize_text_splitter(self):
        """Initialize the text splitter for chunking documents"""
        if RecursiveCharacterTextSplitter is None:
            logger.warning("LangChain not available, using simple text splitter")
            return None

        return RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def _simple_text_splitter(self, text: str) -> List[str]:
        """Simple text splitter fallback if LangChain is not available"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > settings.CHUNK_SIZE:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Keep overlap
                    overlap_words = current_chunk[-settings.CHUNK_OVERLAP//10:]
                    current_chunk = overlap_words + [word]
                    current_length = sum(len(w) for w in current_chunk) + len(current_chunk)
                else:
                    current_chunk = [word]
                    current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    async def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file content"""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")

        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                text += page_text + "\n"

            return text.strip()

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    async def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        try:
            if self.text_splitter:
                chunks = self.text_splitter.split_text(text)
            else:
                chunks = self._simple_text_splitter(text)

            # Filter out very small chunks
            filtered_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]

            logger.info(f"Created {len(filtered_chunks)} chunks from text")
            return filtered_chunks

        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise ValueError(f"Failed to chunk text: {str(e)}")

    async def process_file(
        self,
        file_content: bytes,
        filename: str,
        metadata: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a file: extract text, chunk it, generate embeddings, and store in vector DB
        """
        try:
            # Generate file ID if not provided
            if not file_id:
                file_id = str(uuid.uuid4())

            logger.info(f"Processing file: {filename} (ID: {file_id})")

            # Extract text from PDF
            text = await self.extract_text_from_pdf(file_content)

            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")

            # Chunk the text
            chunks = await self.chunk_text(text)

            if not chunks:
                raise ValueError("No valid chunks created from the text")

            # Generate embeddings for each chunk
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            chunk_embeddings = []

            for i, chunk in enumerate(chunks):
                try:
                    embedding = await self.embeddings_service.generate_embedding(chunk)
                    chunk_embeddings.append({
                        "id": f"{file_id}_chunk_{i}",
                        "text": chunk,
                        "embedding": embedding,
                        "metadata": {
                            "file_id": file_id,
                            "filename": filename,
                            "chunk_index": i,
                            "upload_date": datetime.utcnow().isoformat(),
                            "custom_metadata": metadata
                        }
                    })
                except Exception as e:
                    logger.error(f"Error generating embedding for chunk {i}: {str(e)}")
                    continue

            if not chunk_embeddings:
                raise ValueError("Failed to generate any embeddings")

            # Store in vector database
            await self.vectordb_service.store_embeddings(chunk_embeddings)

            logger.info(f"Successfully processed file {filename}: {len(chunk_embeddings)} chunks stored")

            return {
                "file_id": file_id,
                "chunks_processed": len(chunk_embeddings),
                "total_chunks_created": len(chunks),
                "text_length": len(text)
            }

        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            raise

    async def reprocess_file(self, file_id: str) -> Dict[str, Any]:
        """
        Reprocess an existing file (useful for updating chunk strategy or embeddings)
        """
        try:
            # This would require storing original file content
            # For now, we'll raise an error suggesting to re-upload
            raise NotImplementedError(
                "File reprocessing not implemented. Please re-upload the file to update it."
            )

        except Exception as e:
            logger.error(f"Error reprocessing file {file_id}: {str(e)}")
            raise