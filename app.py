
'''If you want to run it using venv uncomment from proxy settings line 3 to 9, and comment the port setting line 746 to 751'''
# import os
# os.environ['NO_PROXY'] = '*'
# os.environ['all_proxy'] = ''
# os.environ['ALL_PROXY'] = ''
# os.environ['http_proxy'] = ''
# os.environ['https_proxy'] = ''
# os.environ['socks_proxy'] = ''

import os
import gradio as gr
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import plotly.graph_objects as go
from datetime import datetime
import requests
import json

from entity_pipeline import EntityExtractor, DataPreprocessor
from chroma_manager import ChromaDBManager

# Initialize components
print("=" * 50)
print("Initializing Entity Extraction Application...")
print("=" * 50)

entity_extractor = EntityExtractor()

# Get ChromaDB host from environment variables
chroma_host = os.getenv("CHROMA_HOST", "localhost")
chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
api_url = os.getenv("API_URL", "http://localhost:8001")

# Try to connect to ChromaDB
try:
    chroma_manager = ChromaDBManager(host=chroma_host, port=chroma_port)
    collection = chroma_manager.create_or_get_collection("entity_data")
    print(f"Connected to ChromaDB at {chroma_host}:{chroma_port} successfully!")
except Exception as e:
    print(f"Warning: Could not connect to ChromaDB: {e}")
    print("Running in local mode without ChromaDB...")
    chroma_manager = None
    collection = None


# Sample data
SAMPLE_TEXTS = [
    "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976.",
    "Microsoft CEO Satya Nadella announced new products in Seattle yesterday.",
    "Elon Musk's Tesla and SpaceX are based in Texas, USA.",
    "The United Nations meeting in New York discussed climate change on December 15, 2023."
]

# Initialize with sample data
if chroma_manager and collection:
    print("Initializing with sample data...")
    try:
        for text in SAMPLE_TEXTS:
            entities = entity_extractor.extract_entities(text)
            text_embedding = entity_extractor.get_text_embeddings([text])[0]
            chroma_manager.store_entity_results(
                collection, 
                text, 
                entities, 
                text_embedding
            )
        print("Sample data initialized!")
    except Exception as e:
        print(f"Warning: Could not initialize sample data: {e}")

# Custom CSS for better UI
custom_css = """
    .entity-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        background: white;
    }
    .person { border-left: 4px solid #4CAF50; }
    .org { border-left: 4px solid #2196F3; }
    .loc { border-left: 4px solid #FF9800; }
    .date { border-left: 4px solid #9C27B0; }
    .misc { border-left: 4px solid #F44336; }
    .gpe { border-left: 4px solid #795548; }
    .entity-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 8px;
    }
    .confidence-bar {
        height: 6px;
        background: #e0e0e0;
        border-radius: 3px;
        margin-top: 4px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 3px;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .highlight-text {
        background-color: #fff3cd;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 500;
    }
"""

# Helper functions for formatting
def format_entity_card(entity: Dict) -> str:
    """Format entity as HTML card"""
    type_colors = {
        'PER': '#4CAF50',
        'ORG': '#2196F3',
        'LOC': '#FF9800',
        'DATE': '#9C27B0',
        'MISC': '#F44336',
        'GPE': '#795548'
    }
    
    type_labels = {
        'PER': '👤 Person',
        'ORG': '🏢 Organization',
        'LOC': '📍 Location',
        'DATE': '📅 Date',
        'MISC': '📦 Miscellaneous',
        'GPE': '🌍 Geo-Political'
    }
    
    color = type_colors.get(entity['type'], '#9E9E9E')
    label = type_labels.get(entity['type'], entity['type'])
    
    confidence_percent = int(entity['score'] * 100)
    confidence_color = '#4CAF50' if confidence_percent > 80 else '#FF9800' if confidence_percent > 60 else '#F44336'
    
    return f"""
    <div class="entity-card {entity['type'].lower()}">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <div style="font-size: 16px; font-weight: 600; margin-bottom: 4px;">
                    "{entity['text']}"
                </div>
                <span class="entity-badge" style="background: {color}20; color: {color}; border: 1px solid {color}40;">
                    {label}
                </span>
                <span style="font-size: 12px; color: #666;">
                    Position: {entity['start']}-{entity['end']}
                </span>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 14px; font-weight: 600; color: {confidence_color};">
                    {confidence_percent}%
                </div>
                <div style="font-size: 11px; color: #999;">
                    confidence
                </div>
            </div>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence_percent}%; background: {confidence_color};"></div>
        </div>
    </div>
    """

def highlight_text_with_entities(text: str, entities: List[Dict]) -> str:
    """Highlight entities in text with HTML spans"""
    if not entities:
        return text
    
    # Sort entities by start position in reverse order
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    highlighted = text
    type_colors = {
        'PER': '#4CAF50',
        'ORG': '#2196F3',
        'LOC': '#FF9800',
        'DATE': '#9C27B0',
        'MISC': '#F44336',
        'GPE': '#795548'
    }
    
    for entity in sorted_entities:
        start = entity['start']
        end = entity['end']
        entity_text = entity['text']
        entity_type = entity['type']
        color = type_colors.get(entity_type, '#9E9E9E')
        
        # Create highlighted span
        span = f'<span style="background: {color}20; color: {color}; padding: 1px 4px; border-radius: 3px; border-left: 3px solid {color}; margin: 0 2px;" title="{entity_type} - Confidence: {entity["score"]:.2f}">{entity_text}</span>'
        highlighted = highlighted[:start] + span + highlighted[end:]
    
    return highlighted

def format_statistics(stats: Dict) -> str:
    """Format statistics as HTML"""
    total_entities = stats.get('total_entities', 0)
    entity_types = stats.get('entity_type_distribution', {})
    
    # Sort entity types by count
    sorted_types = sorted(entity_types.items(), key=lambda x: x[1], reverse=True)
    
    html = f"""
    <div class="stats-card">
        <div style="font-size: 24px; font-weight: 700; margin-bottom: 8px;">
            📊 Entity Statistics
        </div>
        <div style="font-size: 14px; opacity: 0.9;">
            Total Entities Extracted: <strong>{total_entities}</strong>
        </div>
    </div>
    """
    
    if entity_types:
        html += """
        <div style="margin-top: 20px;">
            <h4 style="margin-bottom: 12px;">📈 Entity Type Distribution</h4>
        """
        
        for entity_type, count in sorted_types[:6]:  # Show top 6
            percentage = (count / total_entities * 100) if total_entities > 0 else 0
            html += f"""
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-weight: 500;">{entity_type}</span>
                    <span style="font-weight: 600;">{count}</span>
                </div>
                <div style="height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden;">
                    <div style="width: {percentage}%; height: 100%; background: #667eea; border-radius: 4px;"></div>
                </div>
                <div style="font-size: 11px; color: #666; text-align: right;">
                    {percentage:.1f}%
                </div>
            </div>
            """
        
        html += "</div>"
    
    return html

def create_interactive_chart(entity_stats: Dict) -> go.Figure:
    """Create interactive chart for entity distribution"""
    if 'entity_type_distribution' not in entity_stats:
        fig = go.Figure()
        fig.update_layout(
            title="No entity data available",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    entity_types = list(entity_stats['entity_type_distribution'].keys())
    counts = list(entity_stats['entity_type_distribution'].values())
    
    # Sort by count
    sorted_data = sorted(zip(entity_types, counts), key=lambda x: x[1], reverse=True)
    entity_types, counts = zip(*sorted_data) if sorted_data else ([], [])
    
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#795548']
    
    fig = go.Figure(data=[go.Bar(
        x=entity_types,
        y=counts,
        marker_color=colors[:len(entity_types)],
        text=counts,
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': "📊 Entity Type Distribution",
            'font': {'size': 16, 'color': '#333'}
        },
        xaxis_title="Entity Type",
        yaxis_title="Count",
        showlegend=False,
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis={'gridcolor': '#f0f0f0'},
        yaxis={'gridcolor': '#f0f0f0'}
    )
    
    return fig

# Gradio UI Functions
def extract_single_text(text: str) -> tuple:
    """Extract entities from a single text"""
    if not text.strip():
        return "Please enter some text to analyze", "", "", ""
    
    # Extract entities
    entities = entity_extractor.extract_entities(text)
    
    # Store in ChromaDB if available
    if chroma_manager and collection and entities:
        try:
            text_embedding = entity_extractor.get_text_embeddings([text])[0]
            chroma_manager.store_entity_results(
                collection, 
                text, 
                entities, 
                text_embedding
            )
        except Exception as e:
            print(f"Warning: Could not store in ChromaDB: {e}")
    
    # Format outputs
    highlighted_text = highlight_text_with_entities(text, entities)
    
    entity_cards = ""
    if entities:
        entity_cards = "".join([format_entity_card(e) for e in entities])
    
    # Prepare summary
    summary = f"""
    <div style="background: #f8f9fa; padding: 16px; border-radius: 8px; margin: 16px 0;">
        <div style="display: flex; justify-content: space-around; text-align: center;">
            <div>
                <div style="font-size: 24px; font-weight: 700; color: #667eea;">{len(entities)}</div>
                <div style="font-size: 12px; color: #666;">Total Entities</div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: 700; color: #4CAF50;">{len(set(e['type'] for e in entities))}</div>
                <div style="font-size: 12px; color: #666;">Unique Types</div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: 700; color: #FF9800;">{int(np.mean([e['score'] for e in entities])*100 if entities else 0)}%</div>
                <div style="font-size: 12px; color: #666;">Avg Confidence</div>
            </div>
        </div>
    </div>
    """
    
    return highlighted_text, entity_cards, summary, ""

def extract_batch_texts(texts: str) -> tuple:
    """Extract entities from multiple texts"""
    if not texts.strip():
        return "Please enter texts to analyze", "", ""
    
    text_list = [t.strip() for t in texts.split('\n') if t.strip()]
    
    if not text_list:
        return "No valid texts found", "", ""
    
    # Extract entities in batches
    results = []
    all_entities = []
    
    for text in text_list[:20]:  # Limit to 20
        entities = entity_extractor.extract_entities(text)
        results.append({
            "text": text,
            "entities": entities,
            "entity_count": len(entities)
        })
        all_entities.extend(entities)
        
        # Store in ChromaDB
        if chroma_manager and collection and entities:
            try:
                text_embedding = entity_extractor.get_text_embeddings([text])[0]
                chroma_manager.store_entity_results(
                    collection, 
                    text, 
                    entities, 
                    text_embedding
                )
            except Exception as e:
                print(f"Warning: Could not store in ChromaDB: {e}")
    
    # Prepare output
    output_html = "<div style='max-height: 500px; overflow-y: auto;'>"
    
    for i, result in enumerate(results[:10]):  # Show first 10
        highlighted = highlight_text_with_entities(result['text'], result['entities'])
        output_html += f"""
        <div style="margin-bottom: 24px; padding: 16px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 12px; color: #666; margin-bottom: 8px;">Text #{i+1}</div>
            <div style="margin-bottom: 12px;">{highlighted}</div>
            <div style="font-size: 12px; color: #667eea;">
                📍 Found {result['entity_count']} entities
            </div>
        </div>
        """
    
    output_html += "</div>"
    
    # Calculate statistics
    entity_types = {}
    for entity in all_entities:
        entity_type = entity['type']
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    stats = {
        "total_texts": len(results),
        "total_entities": len(all_entities),
        "entity_type_distribution": entity_types,
        "unique_entity_types": list(entity_types.keys())
    }
    
    stats_html = format_statistics(stats)
    
    # Create summary text
    summary = f"✅ Processed {len(results)} texts, found {len(all_entities)} entities across {len(entity_types)} entity types."
    
    return output_html, stats_html, summary

def extract_file(file) -> tuple:
    """Extract entities from uploaded file"""
    if file is None:
        return "Please upload a file", "", ""
    
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
            texts = df.iloc[:, 0].dropna().astype(str).tolist()
            
        elif file.name.endswith('.txt'):
            with open(file.name, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            return "Unsupported file format. Please upload CSV or TXT.", "", ""
        
        if not texts:
            return "No valid text found in the file", "", ""
        
        # Extract entities
        results = []
        all_entities = []
        
        for text in texts[:30]:  # Limit to 30
            entities = entity_extractor.extract_entities(text)
            results.append({
                "text": text,
                "entities": entities
            })
            all_entities.extend(entities)
            
            # Store in ChromaDB
            if chroma_manager and collection and entities:
                try:
                    text_embedding = entity_extractor.get_text_embeddings([text])[0]
                    chroma_manager.store_entity_results(
                        collection, 
                        text, 
                        entities, 
                        text_embedding
                    )
                except Exception as e:
                    print(f"Warning: Could not store in ChromaDB: {e}")
        
        # Calculate statistics
        entity_types = {}
        for entity in all_entities:
            entity_type = entity['type']
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Create sample preview
        sample_html = "<div style='max-height: 300px; overflow-y: auto;'>"
        for i, result in enumerate(results[:3]):
            highlighted = highlight_text_with_entities(result['text'][:200] + ("..." if len(result['text']) > 200 else ""), 
                                                     result['entities'])
            sample_html += f"""
            <div style="margin-bottom: 16px; padding: 12px; background: #f8f9fa; border-radius: 6px;">
                <div style="font-weight: 500; margin-bottom: 8px;">Sample #{i+1}:</div>
                <div>{highlighted}</div>
                <div style="font-size: 12px; color: #666; margin-top: 8px;">
                    Found {len(result['entities'])} entities
                </div>
            </div>
            """
        sample_html += "</div>"
        
        stats = {
            "file_name": file.name,
            "total_texts": len(results),
            "total_entities": len(all_entities),
            "entity_type_distribution": entity_types
        }
        
        stats_html = format_statistics(stats)
        summary = f"📁 File: {file.name}<br>✅ Processed {len(results)} texts, found {len(all_entities)} entities"
        
        return sample_html, stats_html, summary
    except Exception as e:
        return f"Error processing file: {str(e)}", "", ""

def search_similar_entities(query: str, entity_type: str = "ALL", n_results: int = 5) -> tuple:
    """Search for similar entities in the database"""
    if not query.strip():
        return "Please enter a search query", ""
    
    if not chroma_manager or not collection:
        return "ChromaDB not available. Please run with Docker Compose for full features.", ""
    
    results = chroma_manager.semantic_search(collection, query, entity_type, n_results)
    
    if not results.get('results'):
        return f"No results found for '{query}'", ""
    
    # Format results
    results_html = "<div style='max-height: 500px; overflow-y: auto;'>"
    
    for i, result in enumerate(results['results']):
        doc_text = result['document'][:150] + "..." if len(result['document']) > 150 else result['document']
        similarity_percent = int(result['similarity'] * 100)
        
        # Parse entities from metadata
        entities_display = ""
        if 'entities' in result and result['entities']:
            entities_display = "<div style='margin-top: 8px;'>"
            for entity in result['entities'][:3]:  # Show first 3 entities
                entities_display += f"""
                <span style="display: inline-block; background: #e3f2fd; color: #1976d2; 
                           padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-right: 4px;">
                    {entity['text']} ({entity['type']})
                </span>
                """
            entities_display += "</div>"
        
        results_html += f"""
        <div style="margin-bottom: 16px; padding: 16px; background: white; border-radius: 8px; 
                    border-left: 4px solid #667eea; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                <div style="font-weight: 600; color: #333;">Result #{i+1}</div>
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; 
                          padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600;">
                    {similarity_percent}% match
                </div>
            </div>
            <div style="color: #666; margin-bottom: 8px; line-height: 1.4;">{doc_text}</div>
            {entities_display}
            <div style="font-size: 11px; color: #999; margin-top: 8px;">
                Found {len(result.get('entities', []))} entities in this text
            </div>
        </div>
        """
    
    results_html += "</div>"
    
    summary = f"🔍 Found {results['total_found']} results for '{query}'"
    
    return results_html, summary

def get_entity_statistics() -> str:  # Changed return type annotation
    """Get entity statistics from database"""
    if chroma_manager and collection:
        stats = chroma_manager.get_entity_stats(collection)
    else:
        stats = {"total_documents": 0, "entity_type_distribution": {}, "status": "ChromaDB not connected"}
    
    # Create status badges
    chroma_status = "🟢 Connected" if chroma_manager else "🔴 Not Connected"
    
    status_html = f"""
    <div style="display: flex; gap: 16px; margin-bottom: 20px;">
        <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; flex: 1;">
            <div style="font-size: 12px; color: #666;">ChromaDB Status</div>
            <div style="font-size: 16px; font-weight: 600;">{chroma_status}</div>
        </div>
        <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; flex: 1;">
            <div style="font-size: 12px; color: #666;">Model</div>
            <div style="font-size: 16px; font-weight: 600;">🤖 BERT NER</div>
        </div>
    </div>
    """
    
    stats_html = format_statistics(stats)
    
    # Convert plotly figure to HTML string instead of returning the figure object
    chart = create_interactive_chart(stats.get('entity_type_distribution', {}))
    chart_html = chart.to_html(full_html=False, include_plotlyjs='cdn')
    
    return status_html + stats_html + f'<div style="margin-top: 20px;">{chart_html}</div>'

# Gradio Interface
with gr.Blocks(title="Entity Extraction Dashboard") as demo:
    gr.Markdown("""
    # 🔍 Smart Entity Extraction Platform
    *Extract, analyze, and search named entities using advanced AI models*
    """)
    
    with gr.Tabs():
        with gr.Tab("🎯 Single Text Analysis", id="single"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📝 Input Text")
                    single_text = gr.Textbox(
                        label="",
                        placeholder="Paste your text here...",
                        lines=8,
                        elem_classes="input-text"
                    )
                    
                    with gr.Row():
                        single_extract_btn = gr.Button("🚀 Extract Entities", variant="primary", size="lg")
                        gr.ClearButton([single_text], variant="secondary")
                    
                    gr.Examples(
                        examples=[
                            ["Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976."],
                            ["Microsoft CEO Satya Nadella announced new products in Seattle."],
                            ["Elon Musk's Tesla and SpaceX are based in Texas, USA."],
                            ["The United Nations meeting in New York discussed climate change on December 15, 2023."]
                        ],
                        inputs=single_text,
                        label="💡 Try these examples:"
                    )
                
                with gr.Column(scale=3):
                    gr.Markdown("### 🔍 Analysis Results")
                    with gr.Tab("📊 Text with Highlights"):
                        highlighted_output = gr.HTML(label="Highlighted Text", elem_classes="output-box")
                    
                    with gr.Tab("📋 Entity Details"):
                        entities_output = gr.HTML(label="Extracted Entities", elem_classes="output-box")
                    
                    summary_output = gr.HTML(label="📈 Summary", elem_classes="summary-box")
            
            single_extract_btn.click(
                extract_single_text,
                inputs=single_text,
                outputs=[highlighted_output, entities_output, summary_output, gr.Textbox(visible=False)]
            )
        
        
        
        with gr.Tab("📁 File Upload", id="file"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📤 Upload Documents")
                    file_input = gr.File(
                        label="",
                        file_types=[".csv", ".txt"],
                        file_count="single"
                    )
                    
                    gr.Markdown("""
                    #### 📋 Supported Formats:
                    - **CSV Files**: First column should contain text
                    - **TXT Files**: Each line is treated as a separate text
                    
                    #### ⚡ Processing Limits:
                    - CSV: First 30 rows processed
                    - TXT: First 30 lines processed
                    """)
                    
                    file_extract_btn = gr.Button("🚀 Process File", variant="primary", size="lg")
                
                with gr.Column(scale=3):
                    gr.Markdown("### 📊 Results")
                    with gr.Tab("👀 Sample Preview"):
                        file_output = gr.HTML(label="Sample Results", elem_classes="output-box")
                    
                    with gr.Tab("📈 Statistics"):
                        file_stats = gr.HTML(label="Statistics", elem_classes="output-box")
                    
                    file_summary = gr.HTML(label="📝 Summary", elem_classes="summary-box")
            
            file_extract_btn.click(
                extract_file,
                inputs=file_input,
                outputs=[file_output, file_stats, file_summary]
            )
        
        with gr.Tab("🔍 Semantic Search", id="search"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🔎 Search Database")
                    search_query = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter keywords or phrases to search...",
                        lines=2
                    )
                    
                    with gr.Row():
                        with gr.Column():
                            entity_type_filter = gr.Dropdown(
                                choices=["ALL", "PERSON", "ORG", "LOC", "DATE", "MISC", "GPE"],
                                value="ALL",
                                label="Filter by Entity Type"
                            )
                        
                        with gr.Column():
                            n_results = gr.Slider(
                                minimum=1, maximum=20, value=5, step=1,
                                label="Number of Results"
                            )
                    
                    with gr.Row():
                        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
                        gr.ClearButton([search_query], variant="secondary")
                
                with gr.Column(scale=3):
                    gr.Markdown("### 📋 Search Results")
                    search_output = gr.HTML(label="Results", elem_classes="output-box")
                    search_summary = gr.HTML(label="📝 Summary", elem_classes="summary-box")
            
            search_btn.click(
                search_similar_entities,
                inputs=[search_query, entity_type_filter, n_results],
                outputs=[search_output, search_summary]
            )
        

        with gr.Tab("📊 Dashboard", id="dashboard"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📈 System Overview")
                    stats_btn = gr.Button("🔄 Refresh Dashboard", variant="primary", size="lg")
                    dashboard_stats = gr.HTML(label="Statistics", elem_classes="output-box")
            
            # In the dashboard tab section, update:
            stats_btn.click(
                get_entity_statistics,
                outputs=[dashboard_stats]  # Only one output now
            )
     
     
if __name__ == "__main__":
    print("=" * 50)
    print("Starting Entity Extraction Application...")
    print("Access the application at: http://localhost:7860")
    print("=" * 50)
    
    demo.launch(
        server_name="0.0.0.0",  # Required for Docker
        server_port=7860,
        share=False,  # Explicitly set to False
        show_error=True,
        debug=False,  # Disable debug mode
        quiet=True   # Reduce logs
    )