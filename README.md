# Project Athena - Research Cognition System

Project Athena is a modular research cognition system that integrates academic search, summarization, and semantic storage. 
It is designed to automate the collection, summarization, and semantic understanding of research papers.

## Overview

The system enables researchers to:

1. Query Exa for research topics (e.g., "neuroplasticity in swarm intelligence")
2. Retrieve metadata, links, and summaries for each paper
3. Generate deeper insights using GPT-4 summarization
4. Automatically extract key concepts from the papers
5. Store full papers and concepts into a local vector database for later querying

## System Architecture

Athena is built to be modular and extendable:

```
┌───────────────┐     ┌────────────────┐     ┌──────────────┐     ┌──────────────────────┐
│  Exa Scraper  │────▶│ PDF Extractor  │────▶│ GPT Summarizer│────▶│ Concept Extractor     │
│  (Search)     │     │ (Full Text)    │     │ (AI Insights) │     │ + Vector DB Injection │
└───────────────┘     └────────────────┘     └──────────────┘     └──────────────────────┘
```

## Project Structure

```
project_athena/
├── core/
│   ├── config.py         # Configuration handling
│   ├── main_runner.py    # Main system orchestration
├── modules/
│   ├── exa_scraper.py    # Exa API integration
│   ├── gpt_summarizer.py # GPT-4 integration
│   ├── pdf_extractor.py  # PDF text extraction
│   ├── obsidian_injector.py # Injects summaries into Obsidian
│   ├── concept_extractor.py # Extracts key concepts from text
│   ├── concept_vector_db.py # Stores concepts in vector database
│   ├── vector_db.py      # Core vector DB management
├── data/
│   ├── papers/           # Storage for downloaded papers
│   └── logs/             # System logs
├── README.md
```

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/project_athena.git
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Configure your environment by creating a `.env` file with:

```
# API Keys
EXA_API_KEY=your_exa_api_key
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TEMPERATURE=0.3
OPENAI_MAX_TOKENS=32000

# Vector Database Configuration
VECTOR_DB_TYPE=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=project_athena

# Obsidian Configuration
OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault

# System Configuration
BATCH_SIZE=10
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

> **Note:** You can easily change the OpenAI model, temperature, and max tokens used for summarization by modifying the corresponding values in your `.env` file or adjusting them directly in `config.py`.

## Obsidian Integration

Project Athena can automatically generate structured markdown notes for each processed research paper.  
By setting your `OBSIDIAN_VAULT_PATH` in the `.env` file, Athena will inject summaries, extracted concepts, and metadata directly into your Obsidian vault.

Each note is generated in markdown (.md) format, making it immediately readable, searchable, and linkable inside Obsidian.  
You can leverage Obsidian’s graph view, backlinks, and plugins to build a powerful research knowledge base.

> **Important:**  
> Ensure that your `OBSIDIAN_VAULT_PATH` points to an existing Obsidian vault, and that Obsidian is set to monitor that directory.
## Basic Usage

### Search and Summarize Papers

```python
from core.main_runner import AthenaRunner

# Initialize the runner
runner = AthenaRunner()

# Run a search pipeline
results = runner.run_search_pipeline(
    query="neuroplasticity in swarm intelligence", 
    max_results=10
)
```

### Query Stored Concepts

```python
from modules.vector_db import VectorDB

# Initialize the vector database
db = VectorDB()

# Query the database
results = db.query(
    query="How does neuroplasticity influence swarm adaptation?",
    limit=5
)

# Print results
for result in results:
    print(f"Title: {result['title']}")
    print(f"Similarity: {result['similarity']}")
    print(f"Concepts: {result['concepts']}")
    print("---")
```

## Current Status

The core search, summarization, concept extraction, and vector storage systems are functional.

## Planned Features

- Advanced semantic clustering of papers
- Batch ingestion of large research corpora
- Interactive command-line interface (CLI)
- GUI dashboard for querying and exploration
- **Deep Research Mode**: Intelligent expansion of searches by creating new queries based on the concepts extracted from initial results, allowing more comprehensive and connected research paths.
- **Zotero Connector Integration**: Enable optional export and synchronization of metadata, summaries, and notes directly into Zotero libraries using the Zotero Connector API.

## License

MIT License
