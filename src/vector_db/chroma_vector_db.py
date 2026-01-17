import logging
from typing import List, Any, Dict, Optional
import json
from pathlib import Path
import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaVectorDB:
    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "notebook_lm"
    ):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        self._initialize_client()
        self._setup_collection()
    
    def _initialize_client(self):
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB client initialized with database: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise
    
    def _setup_collection(self):
        try:
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "NotebookLM document chunks"}
            )
            logger.info(f"Collection '{self.collection_name}' ready")
            
        except Exception as e:
            logger.error(f"Error setting up collection: {str(e)}")
            raise
    
    def insert_embeddings(self, embedded_chunks) -> List[str]:
        """Insert embedded chunks into ChromaDB"""
        if not embedded_chunks:
            return []
        
        try:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            for embedded_chunk in embedded_chunks:
                chunk_data = embedded_chunk.to_vector_db_format()
                
                ids.append(chunk_data['id'])
                documents.append(chunk_data['content'])
                embeddings.append(chunk_data['vector'])
                
                # Prepare metadata (ChromaDB doesn't support nested objects)
                metadata = {
                    'source_file': chunk_data['source_file'],
                    'source_type': chunk_data['source_type'],
                    'chunk_index': chunk_data['chunk_index'],
                    'embedding_model': chunk_data['embedding_model']
                }
                
                # Add optional fields if they exist
                if chunk_data.get('page_number') is not None:
                    metadata['page_number'] = chunk_data['page_number']
                if chunk_data.get('start_char') is not None:
                    metadata['start_char'] = chunk_data['start_char']
                if chunk_data.get('end_char') is not None:
                    metadata['end_char'] = chunk_data['end_char']
                
                # Store additional metadata as JSON string
                if chunk_data.get('metadata'):
                    metadata['additional_metadata'] = json.dumps(chunk_data['metadata'])
                
                metadatas.append(metadata)
            
            # Insert into ChromaDB
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Inserted {len(ids)} embeddings into ChromaDB")
            return ids
            
        except Exception as e:
            logger.error(f"Error inserting embeddings: {str(e)}")
            raise
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        try:
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where=filter_dict,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    metadata = results['metadatas'][0][i]
                    
                    # Parse additional metadata if it exists
                    additional_metadata = {}
                    if 'additional_metadata' in metadata:
                        try:
                            additional_metadata = json.loads(metadata['additional_metadata'])
                        except:
                            pass
                    
                    formatted_result = {
                        'id': results['ids'][0][i],
                        'score': results['distances'][0][i],
                        'content': results['documents'][0][i],
                        'citation': {
                            'source_file': metadata.get('source_file'),
                            'source_type': metadata.get('source_type'),
                            'page_number': metadata.get('page_number'),
                            'chunk_index': metadata.get('chunk_index'),
                            'start_char': metadata.get('start_char'),
                            'end_char': metadata.get('end_char'),
                        },
                        'metadata': additional_metadata,
                        'embedding_model': metadata.get('embedding_model')
                    }
                    formatted_results.append(formatted_result)
            
            logger.info(f"Search completed: {len(formatted_results)} results found")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chunk by ID"""
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=['documents', 'metadatas']
            )
            
            if results['ids'] and len(results['ids']) > 0:
                metadata = results['metadatas'][0]
                
                # Parse additional metadata
                additional_metadata = {}
                if 'additional_metadata' in metadata:
                    try:
                        additional_metadata = json.loads(metadata['additional_metadata'])
                    except:
                        pass
                
                return {
                    'id': results['ids'][0],
                    'content': results['documents'][0],
                    'metadata': additional_metadata,
                    'source_file': metadata.get('source_file'),
                    'source_type': metadata.get('source_type'),
                    'page_number': metadata.get('page_number'),
                    'chunk_index': metadata.get('chunk_index')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving chunk by ID {chunk_id}: {str(e)}")
            return None
    
    def delete_collection(self):
        """Delete the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted")
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}


if __name__ == "__main__":
    from src.doc.doc_processor import DocumentProcessor
    from src.embeddings.embedding_generator import EmbeddingGenerator

    # Test the ChromaDB implementation
    doc_processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator()
    vector_db = ChromaVectorDB()

    try:
        # Process document
        chunks = doc_processor.process_document("data/attn_is_all_you_need.pdf")
        print(f"Processed {len(chunks)} chunks")
        
        # Generate embeddings
        embedded_chunks = embedding_generator.generate_embeddings(chunks)
        print(f"Generated embeddings for {len(embedded_chunks)} chunks")
        
        # Insert into vector database
        inserted_ids = vector_db.insert_embeddings(embedded_chunks)
        print(f"Inserted {len(inserted_ids)} embeddings")
        
        # Test search
        query_text = "What is attention mechanism?"
        query_vector = embedding_generator.generate_query_embedding(query_text)
        
        search_results = vector_db.search(query_vector.tolist(), limit=3)
        
        print(f"\nSearch results for: '{query_text}'")
        for i, result in enumerate(search_results):
            print(f"\nResult {i+1}:")
            print(f"Score: {result['score']:.4f}")
            print(f"Content: {result['content'][:200]}...")
            print(f"Source: {result['citation']['source_file']} (page {result['citation']['page_number']})")

    except Exception as e:
        print(f"Error in example: {e}")
        import traceback
        traceback.print_exc()