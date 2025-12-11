import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import uuid
import json


class ChromaDBManager:
    def __init__(self, host: str = "localhost", port: int = 8000):
        """Initialize ChromaDB connection"""
        print(f"Connecting to ChromaDB at {host}:{port}")
        
        try:
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
            print("Connected to ChromaDB successfully!")
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            try:
                self.client = chromadb.PersistentClient(
                    path="./chroma_db",
                    settings=Settings(anonymized_telemetry=False)
                )
                print("Connected to local ChromaDB successfully!")
            except Exception as e2:
                print(f"Failed to connect to ChromaDB: {e2}")
                raise
    
    
    def create_or_get_collection(self, name: str = "entity_data") -> chromadb.Collection:
        """Create or get a collection for entity storage"""
        try:
            # Try to get the collection first
            collection = self.client.get_collection(name)
            print(f"Using existing collection: {name}")
        except Exception as e:
            # If it doesn't exist, create it
            print(f"Creating new collection: {name}")
            collection = self.client.create_collection(
                name=name,
                metadata={"description": "Entity extraction data"}
            )
        
        return collection
    
    def store_entity_results(self, collection: chromadb.Collection, 
                           text: str, 
                           entities: List[Dict[str, Any]],
                           embedding: List[float]) -> None:
        """Store entity extraction results in ChromaDB"""
        if not text or not embedding:
            print("No data to store")
            return
        
        # Prepare metadata with entities
        metadata = {
            "text": text[:500],  # Truncate if too long
            "entities": json.dumps(entities),
            "entity_count": len(entities),
            "entity_types": ",".join(list(set([e['type'] for e in entities]))),
            "source": "entity_extraction"
        }
        
        # Add to collection
        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[f"entity_{uuid.uuid4().hex[:8]}"]
        )
        
        print(f"Stored text with {len(entities)} entities in ChromaDB")
    
    def semantic_search(self, collection: chromadb.Collection, 
                       query: str, 
                       entity_type: str = "ALL",
                       n_results: int = 5) -> Dict[str, Any]:
        """Perform semantic search on stored texts"""
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    
                    # Filter by entity type if specified
                    if entity_type != "ALL" and entity_type not in metadata.get('entity_types', ''):
                        continue
                    
                    # Parse entities from JSON string
                    entities = []
                    if 'entities' in metadata:
                        try:
                            entities = json.loads(metadata['entities'])
                        except:
                            entities = []
                    
                    formatted_results.append({
                        "document": results['documents'][0][i],
                        "entities": entities,
                        "metadata": metadata,
                        "distance": results['distances'][0][i] if results['distances'] else 0,
                        "similarity": 1 - (results['distances'][0][i] if results['distances'] else 0)
                    })
            
            return {
                "query": query,
                "entity_type_filter": entity_type,
                "results": formatted_results,
                "total_found": len(formatted_results)
            }
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return {"query": query, "results": [], "total_found": 0}
    
    def get_entity_stats(self, collection: chromadb.Collection) -> Dict[str, Any]:
        """Get statistics about stored entities"""
        try:
            count = collection.count()
            
            # Get all metadata to analyze entity types
            try:
                all_data = collection.get(include=["metadatas"])
                entity_types = {}
                
                if all_data['metadatas']:
                    for metadata in all_data['metadatas']:
                        if 'entity_types' in metadata:
                            types_str = metadata['entity_types']
                            if types_str:
                                for entity_type in types_str.split(','):
                                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                
                return {
                    "collection_name": collection.name,
                    "total_documents": count,
                    "entity_type_distribution": entity_types,
                    "unique_entity_types": list(entity_types.keys()),
                    "total_entities": sum(entity_types.values())
                }
            except:
                return {
                    "collection_name": collection.name,
                    "total_documents": count,
                    "entity_type_distribution": {},
                    "unique_entity_types": [],
                    "total_entities": 0
                }
                
        except Exception as e:
            print(f"Error getting entity stats: {e}")
            return {
                "collection_name": collection.name if collection else "Unknown",
                "total_documents": 0,
                "entity_type_distribution": {},
                "status": "error"
            }