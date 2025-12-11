# entity-extraction-app

# Smart Entity Extraction Platform 🚀
A tool for extracting, analyzing, and searching named entities (people, organizations, locations, dates, etc.) from text using AI! Perfect for NLP projects, data analysis, or just playing around with advanced language models.


## What It Does
This tool lets you:
- Extract entities from single text inputs (like sentences or paragraphs)
- Upload CSV/TXT files to process batches of text
- Search for similar texts using semantic search (powered by ChromaDB)
- View statistics and breakdowns of extracted entities
- See highlighted entities in your original text with confidence scores


## How It Works
Under the hood, it uses:
- **BERT NER Model**: For accurate entity extraction (dslim/bert-base-NER)
- **spaCy**: Extra NLP power to catch more entities (fallback if BERT misses something)
- **Sentence Transformers**: To create text embeddings for semantic search
- **ChromaDB**: A vector database to store and search through your text data
- **Gradio**: The web interface that makes it easy to use (no coding required!)


## Installation & Setup
### Prerequisites
- Docker and Docker Compose (easiest way to run everything)
- Or Python 3.9+ if you want to run it locally without Docker

### Option 1: Docker (Recommended, Less Headaches)
1. Clone this repo (or download all the files)
2. Make sure Docker is running on your computer
3. Open a terminal in the project folder
4. Run this command:
   docker compose up --build
5. Wait a few minutes 
6. Open your browser and go to `http://localhost:7860` – that’s the app!

### Option 2: Local Python Setup 
1. Create a virtual environment :
   python3 -m venv .venv
   # On Windows: .venv\Scripts\activate
   # On Mac/Linux: source .venv/bin/activate
   
2. Install all the requirements:
   pip3 install -r requirements.txt

3. Download the spaCy model:
   python -m spacy download en_core_web_sm

4. Run ChromaDB locally:
   - chroma run --host 0.0.0.0 --port 8000

5. Run the app:
   python3 app.py

6. Go to `http://localhost:7860` in your browser


## How to Use
### 1. Single Text Analysis
- Paste any text (sentence, paragraph, etc.) into the text box
- Click "🚀 Extract Entities"
- Check out the highlighted text (entities are color-coded!) and entity details (with confidence scores)

### 2. File Upload
- Upload a CSV or TXT file (see format notes below)
- Click "🚀 Process File"
- View sample results and statistics about the entities found

#### File Format Rules:
- **CSV**: First column must have the text (other columns are ignored)
- **TXT**: Each line is treated as a separate text entry
- Max 30 entries processed (to keep it fast!)

### 3. Semantic Search
- Type a query (e.g., "Apple", "climate change")
- Optional: Filter by entity type (Person, Organization, etc.)
- Choose how many results you want
- Click "🔍 Search" to find similar texts in the database

### 4. Dashboard
- Click "🔄 Refresh Dashboard" to see overall stats
- Check ChromaDB connection status
- View entity type distribution (how many people vs. organizations vs. locations)


## What’s Included in the Project?
Here’s all the files you get (and what they do):
- `app.py`: The main app with the Gradio interface
- `entity_pipeline.py`: Handles entity extraction with BERT and spaCy
- `chroma_manager.py`: Manages the ChromaDB database for storage/search
- `docker-compose.yml`: Sets up Docker containers for the app and ChromaDB
- `Dockerfile`: Builds the app container (for Docker setup)
- `requirements.txt`: All the Python packages you need
- `sample_data.csv`: Example data to test the file upload feature
- `data/`: Folder where sample data are stored


## Entity Types Recognized
The tool can identify these entity types (color-coded!):
- 👤 **PER**: People (e.g., Steve Jobs, Elon Musk)
- 🏢 **ORG**: Organizations (e.g., Apple, United Nations)
- 📍 **LOC**: Locations (e.g., California, Seattle)
- 📅 **DATE**: Dates (e.g., April 1, 1976, December 15, 2023)
- 🌍 **GPE**: Geo-Political Entities (e.g., USA, New York)
- 📦 **MISC**: Miscellaneous (e.g., Falcon 9, WWDC)


## Known Issues
- Sometimes the BERT model misses rare entities (spaCy helps, but not perfect)
- Semantic search only works if ChromaDB is running
- Large files might take a few seconds to process
- If the app crashes, just restart it (Docker: `docker-compose restart ner-app`)


## Future Improvements  
- Support for more languages (right now it’s English only)
- Export results to CSV/JSON
- Train the model on custom data
- Better error messages for bad files


