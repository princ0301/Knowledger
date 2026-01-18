import os
import logging
from typing import List, Any, Dict, Optional
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
try:
    from qdrant_client.models import PayloadSchemaType
except ImportError:
    # Fallback for older versions
    PayloadSchemaType = None
from qdrant_client.http.exceptions import UnexpectedResponse
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantVectorDB:
    def __init__(
        self,
        collection_name: str = "notebook_lm",
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        vector_size: int = 384  # Default for BAAI/bge-small-en-v1.5
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Get credentials from parameters or environment
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.url = url or os.getenv("QDRANT_CLUSTER")
        
        if not self.api_key or not self.url:
            raise ValueError("QDRANT_API_KEY and QDRANT_CLUSTER must be provided")
        
        self.client = None
        self._initialize_client()
        self._setup_collection()
    
    def _initialize_client(self):
        """Initialize Qdrant client"""
        try:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=60
            )
            logger.info(f"Qdrant client initialized successfully")
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant cluster with {len(collections.collections)} collections")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise
    
    def _setup_collection(self):
        """Create or get collection"""
        try:
            # Check if collection exists
            collection_exists = False
            try:
                collection_info = self.client.get_collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' already exists")
                collection_exists = True
            except UnexpectedResponse as e:
                if e.status_code == 404:
                    # Collection doesn't exist, create it
                    collection_exists = False
                else:
                    raise
            
            if not collection_exists:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
            
            # Ensure payload indexes exist (for both new and existing collections)
            self._ensure_indexes()
            
        except Exception as e:
            logger.error(f"Error setting up collection: {str(e)}")
            raise
    
    def _ensure_indexes(self):
        """Ensure required payload indexes exist"""
        try:
            # Create payload indexes for filtering
            try:
                if PayloadSchemaType:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="source_file",
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                else:
                    # Fallback for older versions
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="source_file",
                        field_schema="keyword"
                    )
                logger.info("Created index for 'source_file' field")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.debug("Index for 'source_file' already exists")
                else:
                    logger.warning(f"Could not create index for 'source_file': {e}")
            
            try:
                if PayloadSchemaType:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="source_type",
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                else:
                    # Fallback for older versions
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="source_type",
                        field_schema="keyword"
                    )
                logger.info("Created index for 'source_type' field")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.debug("Index for 'source_type' already exists")
                else:
                    logger.warning(f"Could not create index for 'source_type': {e}")
                    
        except Exception as e:
            logger.warning(f"Error ensuring indexes: {str(e)}")
            # Don't raise here as the collection might still work without indexes for basic operations
    
    def insert_embeddings(self, embedded_chunks) -> List[str]:
        """Insert embedded chunks into Qdrant"""
        if not embedded_chunks:
            return []
        
        try:
            points = []
            
            for embedded_chunk in embedded_chunks:
                chunk_data = embedded_chunk.to_vector_db_format()
                
                # Create payload with all metadata
                payload = {
                    'content': chunk_data['content'],
                    'source_file': chunk_data['source_file'],
                    'source_type': chunk_data['source_type'],
                    'chunk_index': chunk_data['chunk_index'],
                    'embedding_model': chunk_data['embedding_model']
                }
                
                # Add optional fields if they exist
                if chunk_data.get('page_number') is not None:
                    payload['page_number'] = chunk_data['page_number']
                if chunk_data.get('start_char') is not None:
                    payload['start_char'] = chunk_data['start_char']
                if chunk_data.get('end_char') is not None:
                    payload['end_char'] = chunk_data['end_char']
                
                # Add additional metadata
                if chunk_data.get('metadata'):
                    payload['additional_metadata'] = chunk_data['metadata']
                
                # Create point
                point = PointStruct(
                    id=chunk_data['id'],
                    vector=chunk_data['vector'],
                    payload=payload
                )
                points.append(point)
            
            # Insert points in batches
            batch_size = 100
            inserted_ids = []
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                inserted_ids.extend([point.id for point in batch])
            
            logger.info(f"Inserted {len(inserted_ids)} embeddings into Qdrant")
            return inserted_ids
            
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
            # Convert filter_dict to Qdrant filter if provided
            qdrant_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    qdrant_filter = Filter(must=conditions)
            
            # Perform search using the correct method
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False
            ).points
            
            formatted_results = []
            for result in search_results:
                payload = result.payload
                
                formatted_result = {
                    'id': result.id,
                    'score': result.score,
                    'content': payload.get('content', ''),
                    'citation': {
                        'source_file': payload.get('source_file'),
                        'source_type': payload.get('source_type'),
                        'page_number': payload.get('page_number'),
                        'chunk_index': payload.get('chunk_index'),
                        'start_char': payload.get('start_char'),
                        'end_char': payload.get('end_char'),
                    },
                    'metadata': payload.get('additional_metadata', {}),
                    'embedding_model': payload.get('embedding_model')
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
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id],
                with_payload=True,
                with_vectors=False
            )
            
            if points:
                point = points[0]
                payload = point.payload
                
                return {
                    'id': point.id,
                    'content': payload.get('content', ''),
                    'metadata': payload.get('additional_metadata', {}),
                    'source_file': payload.get('source_file'),
                    'source_type': payload.get('source_type'),
                    'page_number': payload.get('page_number'),
                    'chunk_index': payload.get('chunk_index')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving chunk by ID {chunk_id}: {str(e)}")
            return None
    
    def delete_collection(self):
        """Delete the collection"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted")
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'total_chunks': collection_info.points_count,
                'collection_name': self.collection_name,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.value
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def close(self):
        """Close the Qdrant client connection"""
        try:
            if self.client:
                # Qdrant client doesn't need explicit closing
                # Just reset the reference
                self.client = None
                logger.info("Qdrant client closed")
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {str(e)}")


if __name__ == "__main__":
    from src.doc.doc_processor import DocumentProcessor
    from src.embeddings.embedding_generator import EmbeddingGenerator

    # Test the Qdrant implementation
    try:
        doc_processor = DocumentProcessor()
        embedding_generator = EmbeddingGenerator()
        vector_db = QdrantVectorDB()

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

        # Get collection stats
        stats = vector_db.get_collection_stats()
        print(f"\nCollection Stats: {stats}")

    except Exception as e:
        print(f"Error in example: {e}")
        import traceback
        traceback.print_exc()