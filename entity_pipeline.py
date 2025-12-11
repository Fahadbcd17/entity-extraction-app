import re
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
from collections import defaultdict

class EntityExtractor:
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """Initialize BERT NER model"""
        print(f"Loading NER model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load BERT NER model and tokenizer
        print("Loading BERT tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Create NER pipeline
        print("Creating NER pipeline...")
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Try to load spaCy for additional processing (optional)
        self.nlp = None
        try:
            import spacy
            print("Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_sm")
            print("spaCy model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load spaCy model: {e}")
            print("Note: Using BERT NER only (spaCy not available). To install spaCy model, run:")
            print("python -m spacy download en_core_web_sm")
        
        # Initialize sentence transformer for embeddings
        print("Loading sentence transformer for embeddings...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Entity type mapping for BERT NER
        self.entity_labels = {
            'O': 'Outside',
            'B-PER': 'Person', 'I-PER': 'Person',
            'B-ORG': 'Organization', 'I-ORG': 'Organization',
            'B-LOC': 'Location', 'I-LOC': 'Location',
            'B-MISC': 'Miscellaneous', 'I-MISC': 'Miscellaneous',
            'B-DATE': 'Date', 'I-DATE': 'Date',
            'B-GPE': 'Geo-Political', 'I-GPE': 'Geo-Political'
        }
        
        print("NER model loaded successfully!")
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        if not text.strip():
            return []
        
        cleaned_text = self.clean_text(text)
        
        try:
            # Get NER predictions from BERT
            ner_results = self.ner_pipeline(cleaned_text)
            
            # Process results
            entities = []
            for entity in ner_results:
                entities.append({
                    'text': entity['word'],
                    'type': entity['entity_group'],
                    'score': float(entity['score']),
                    'start': entity['start'],
                    'end': entity['end']
                })
            
            # Additional processing with spaCy if available
            if self.nlp:
                try:
                    doc = self.nlp(cleaned_text)
                    for ent in doc.ents:
                        # Map spaCy labels to consistent format
                        spacy_to_bert = {
                            'PERSON': 'PER',
                            'ORG': 'ORG', 
                            'GPE': 'LOC',
                            'LOC': 'LOC',
                            'DATE': 'DATE',
                            'TIME': 'DATE',
                            'MONEY': 'MISC',
                            'PERCENT': 'MISC',
                            'FAC': 'LOC',
                            'PRODUCT': 'MISC',
                            'EVENT': 'MISC',
                            'LAW': 'MISC',
                            'LANGUAGE': 'MISC',
                            'NORP': 'MISC',
                            'WORK_OF_ART': 'MISC',
                            'QUANTITY': 'MISC',
                            'ORDINAL': 'MISC',
                            'CARDINAL': 'MISC'
                        }
                        
                        entity_type = spacy_to_bert.get(ent.label_, 'MISC')
                        
                        # Check if entity is already found by BERT
                        is_duplicate = False
                        for bert_entity in entities:
                            bert_text = bert_entity['text'].lower()
                            spacy_text = ent.text.lower()
                            
                            # Check for overlap
                            if (bert_text in spacy_text or spacy_text in bert_text or
                                bert_text == spacy_text):
                                is_duplicate = True
                                # Keep the one with higher confidence
                                if bert_entity['score'] < 0.85:  # spaCy's approximate confidence
                                    bert_entity['text'] = ent.text
                                    bert_entity['type'] = entity_type
                                    bert_entity['score'] = 0.85
                                break
                        
                        if not is_duplicate:
                            entities.append({
                                'text': ent.text,
                                'type': entity_type,
                                'score': 0.85,  # Approximate confidence for spaCy
                                'start': ent.start_char,
                                'end': ent.end_char
                            })
                except Exception as e:
                    print(f"Warning: Error in spaCy processing: {e}")
                    print("Continuing with BERT NER results only...")
            
            # Remove exact duplicates and keep highest confidence
            unique_entities = self._deduplicate_entities(entities)
            
            return unique_entities
            
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return []
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities, keeping highest confidence"""
        entity_map = {}
        
        for entity in entities:
            key = (entity['text'].lower(), entity['type'])
            if key not in entity_map or entity['score'] > entity_map[key]['score']:
                entity_map[key] = entity
        
        return list(entity_map.values())
    
    def batch_extract(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract entities from multiple texts"""
        results = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"Processing text {i+1}/{len(texts)}...")
            
            entities = self.extract_entities(text)
            results.append({
                'text': text,
                'entities': entities,
                'entity_count': len(entities),
                'entity_types': list(set([e['type'] for e in entities]))
            })
        return results
    
    def get_entity_embeddings(self, entities: List[str]) -> List[List[float]]:
        """Get embeddings for extracted entities"""
        try:
            if not entities:
                return []
            
            # Filter out empty strings
            valid_entities = [e for e in entities if e.strip()]
            if not valid_entities:
                return []
            
            embeddings = self.embedding_model.encode(valid_entities)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            # Return dummy embeddings if model fails
            return [[0.0] * 384 for _ in entities]
    
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for entire texts"""
        try:
            cleaned_texts = [self.clean_text(text) for text in texts]
            embeddings = self.embedding_model.encode(cleaned_texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error getting text embeddings: {e}")
            return [[0.0] * 384 for _ in texts]
    
    def get_entity_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Get statistics about entities in texts"""
        all_entities = []
        for text in texts:
            entities = self.extract_entities(text)
            all_entities.extend(entities)
        
        entity_types = {}
        for entity in all_entities:
            entity_type = entity['type']
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        avg_confidence = np.mean([e['score'] for e in all_entities]) if all_entities else 0
        
        return {
            "total_entities": len(all_entities),
            "entity_type_distribution": entity_types,
            "unique_entity_types": list(entity_types.keys()),
            "average_confidence": avg_confidence,
            "total_texts": len(texts)
        }

class DataPreprocessor:
    """Handle data preprocessing for entity extraction"""
    
    @staticmethod
    def load_and_prepare_data(file_path: str) -> List[Dict[str, Any]]:
        """Load dataset from CSV or JSON file"""
        import pandas as pd
        import json
        
        try:
            data = []
            
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
                
                # Use first column as text, second as labels if exists
                for idx, row in df.iterrows():
                    text = str(row.iloc[0]) if len(row) > 0 else ""
                    
                    # Try to parse entities from second column if exists
                    entities = []
                    if len(row) > 1:
                        label_data = row.iloc[1]
                        if isinstance(label_data, str) and label_data.strip():
                            try:
                                # Try to parse as JSON
                                entities = json.loads(label_data)
                            except:
                                # If not JSON, treat as comma-separated entity types
                                entity_types = [t.strip() for t in label_data.split(',')]
                                entities = [{"text": text, "type": et} for et in entity_types if et]
                    
                    item = {
                        "id": idx,
                        "text": text,
                        "entities": entities
                    }
                    data.append(item)
                    
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        if line.strip():
                            data.append({
                                "id": idx,
                                "text": line.strip(),
                                "entities": []
                            })
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data = [data]  # Convert single dict to list
            else:
                print(f"Unsupported file format: {file_path}")
                return []
            
            print(f"Successfully loaded {len(data)} records")
            return data
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
    
    @staticmethod
    def normalize_texts(texts: List[str]) -> List[str]:
        """Normalize texts by cleaning and standardizing"""
        normalized = []
        for text in texts:
            if isinstance(text, str):
                # Basic cleaning
                text = re.sub(r'\s+', ' ', text).strip()
                normalized.append(text)
        return normalized
    
    @staticmethod
    def prepare_training_data(data: List[Dict]) -> Tuple[List[str], List[List[Dict]]]:
        """Prepare data for NER model training"""
        texts = []
        entities_list = []
        
        for item in data:
            if "text" in item and item["text"]:
                texts.append(item["text"])
                
                if "entities" in item:
                    entities_list.append(item["entities"])
                else:
                    entities_list.append([])
        
        print(f"Prepared {len(texts)} texts for training")
        return texts, entities_list
    
    @staticmethod
    def normalize_entities(entities: List[Dict]) -> List[Dict]:
        """Normalize entity formats"""
        normalized = []
        for entity in entities:
            if isinstance(entity, dict):
                norm_entity = {
                    'text': entity.get('text', '').strip(),
                    'type': entity.get('type', 'UNKNOWN').upper().replace(' ', '_'),
                    'score': float(entity.get('score', 0.0))
                }
                if 'start' in entity:
                    norm_entity['start'] = int(entity['start'])
                if 'end' in entity:
                    norm_entity['end'] = int(entity['end'])
                normalized.append(norm_entity)
        return normalized
    
    @staticmethod
    def create_sample_dataset(output_path: str = "data/sample_entities.csv"):
        """Create a sample dataset for demonstration"""
        import pandas as pd
        
        sample_data = [
            {
                "text": "Apple CEO Tim Cook announced new products at the WWDC event in San Jose, California.",
                "entities": [
                    {"text": "Apple", "type": "ORG", "start": 0, "end": 5},
                    {"text": "Tim Cook", "type": "PER", "start": 9, "end": 17},
                    {"text": "WWDC", "type": "ORG", "start": 44, "end": 48},
                    {"text": "San Jose", "type": "LOC", "start": 56, "end": 64},
                    {"text": "California", "type": "LOC", "start": 66, "end": 76}
                ]
            },
            {
                "text": "Elon Musk's SpaceX launched a Falcon 9 rocket from Kennedy Space Center in Florida yesterday.",
                "entities": [
                    {"text": "Elon Musk", "type": "PER", "start": 0, "end": 9},
                    {"text": "SpaceX", "type": "ORG", "start": 12, "end": 18},
                    {"text": "Falcon 9", "type": "MISC", "start": 28, "end": 36},
                    {"text": "Kennedy Space Center", "type": "LOC", "start": 52, "end": 73},
                    {"text": "Florida", "type": "LOC", "start": 77, "end": 84}
                ]
            },
            {
                "text": "Microsoft reported quarterly earnings of $56.2 billion, beating analysts' expectations.",
                "entities": [
                    {"text": "Microsoft", "type": "ORG", "start": 0, "end": 9}
                ]
            }
        ]
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "text": item["text"],
                "entities_json": json.dumps(item["entities"])
            }
            for item in sample_data
        ])
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Sample dataset created at: {output_path}")
        return output_path


# Helper function to test the model
def test_entity_extraction():
    """Test the entity extraction functionality"""
    print("=" * 50)
    print("Testing Entity Extraction...")
    print("=" * 50)
    
    # Initialize extractor
    extractor = EntityExtractor()
    
    # Test sentences
    test_sentences = [
        "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
        "Microsoft CEO Satya Nadella announced new products in Seattle.",
        "The United Nations meeting will be held in New York on December 15, 2024."
    ]
    
    for i, sentence in enumerate(test_sentences):
        print(f"\nTest {i+1}: {sentence}")
        entities = extractor.extract_entities(sentence)
        
        if entities:
            print(f"Found {len(entities)} entities:")
            for entity in entities:
                print(f"  - '{entity['text']}' ({entity['type']}) - Confidence: {entity['score']:.2f}")
        else:
            print("No entities found.")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)


if __name__ == "__main__":
    test_entity_extraction()