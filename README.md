# AMG-RAG: Agentic Medical Graph-RAG

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/your-paper-id)

## Overview

**AMG-RAG (Agentic Medical Graph-RAG)** is a comprehensive framework that automates the construction and continuous updating of Medical Knowledge Graphs (MKGs), integrates reasoning, and retrieves current external evidence for medical Question Answering (QA). Our approach addresses the challenge of rapidly evolving medical knowledge by dynamically linking new findings and complex medical concepts.

## Key Features

- **Automated Knowledge Graph Construction**: Builds and continuously updates Medical Knowledge Graphs
- **Multi-source Evidence Retrieval**: Integrates PubMed search and vector database retrieval
- **Chain-of-Thought Reasoning**: Implements structured reasoning for medical queries
- **Agentic Workflow**: Uses LangGraph for orchestrated multi-step processing
- **Real-time Updates**: Dynamically incorporates latest medical literature

## Performance

Our evaluations on standard medical QA benchmarks demonstrate superior performance:

- **MEDQA**: F1 score of 74.1%
- **MEDMCQA**: Accuracy of 66.34%

AMG-RAG surpasses both comparable models and those 10 to 100 times larger, while enhancing interpretability for medical queries.

## Architecture

The system consists of several key components:

1. **Vector Database Retrieval**: Semantic search through medical QA corpus
2. **Search Query Generation**: LLM-powered extraction of medical search terms
3. **External Evidence Retrieval**: PubMed API integration for latest research
4. **Chain-of-Thought Generation**: Structured reasoning synthesis
5. **Final Answer Generation**: Multi-evidence integration for answer selection

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key (or Ollama for local inference)
- PubMed API key (optional, for higher rate limits)

### Dependencies

```bash
pip install langchain
pip install langchain-community
pip install langchain-huggingface
pip install langchain-chroma
pip install langchain-ollama
pip install transformers
pip install langgraph
pip install neo4j
pip install pandas
pip install numpy
pip install requests
pip install wikipedia
pip install wikipediaapi
pip install duckduckgo-search
pip install networkx
pip install python-decouple
```

### Environment Setup

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
pubmed_api=your_pubmed_api_key_here
```

## Usage

### Basic Usage

```python
from qa_chain_processor import QAChainProcessor

# Initialize the processor
processor = QAChainProcessor()

# Process a dataset
jsonl_file = "dataset/MEDQA/questions/US/test.jsonl"
output_csv = "results/AMG_pubmed_test.csv"

processor.main(jsonl_file, output_csv)
```

### Single Question Processing

```python
question_data = {
    "question": "What is the most common cause of acute myocardial infarction?",
    "options": {
        "A": "Coronary artery spasm",
        "B": "Coronary thrombosis", 
        "C": "Coronary embolism",
        "D": "Coronary dissection"
    },
    "answer": "B",
    "answer_idx": 1
}

result = processor.process_question(question_data)
print(f"Model Answer: {result['model_answer']}")
```

## Data Format

The system expects input data in JSONL format with the following structure:

```json
{
  "question": "Medical question text",
  "options": {
    "A": "Option A text",
    "B": "Option B text", 
    "C": "Option C text",
    "D": "Option D text"
  },
  "answer": "B",
  "answer_idx": 1,
  "meta_info": "Additional metadata"
}
```

## Configuration

### Model Selection

The system supports both OpenAI and local Ollama models:

```python
# OpenAI (default)
self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=config('OPENAI_API_KEY'))

# Ollama (uncomment to use)
# self.llm = ChatOllama(
#     model="llama3.2",
#     temperature=0.0,
#     num_predict=200,
#     format="json"
# )
```

### Search Parameters

Adjust search behavior by modifying these parameters:

```python
self.max_entity_size = 2      # Max PubMed articles per search term
self.max_doc_search = 3       # Max Wikipedia results per search
```

## Output

The system generates comprehensive results including:

- **Question and Options**: Original query and multiple choice options
- **Model Answer**: Selected answer (A, B, C, D, or NAN)
- **Chain-of-Thought**: Detailed reasoning process
- **Search Results**: Retrieved evidence from external sources
- **Vector Database Documents**: Relevant passages from medical corpus
- **Search Terms**: Generated medical search phrases

Results are saved in CSV format with JSON-encoded complex fields for further analysis.

## File Structure

```
AMG-RAG/
├── qa_chain_processor.py     # Main processing pipeline
├── create_VDB.py            # Vector database creation utilities
├── dataset/                 # Input datasets
│   └── MEDQA/
│       └── questions/US/test.jsonl
├── results/                 # Output results
├── requirements.txt         # Python dependencies
├── .env                    # Environment variables
└── README.md              # This file
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use AMG-RAG in your research, please cite our paper:

```bibtex
@article{amg-rag-2024,
  title={AMG-RAG: Agentic Medical Graph-RAG for Enhanced Medical Question Answering},
  author={[Author Names]},
  journal={arXiv preprint arXiv:[paper-id]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangChain](https://langchain.com/) and [LangGraph](https://langgraph-sdk.vercel.app/)
- Uses [Hugging Face Transformers](https://huggingface.co/transformers/) for embeddings
- Integrates [PubMed API](https://www.ncbi.nlm.nih.gov/home/develop/api/) for medical literature retrieval
- Benchmarked on [MEDQA](https://github.com/jind11/MedQA) and [MEDMCQA](https://medmcqa.github.io/) datasets

## Support

For questions, issues, or support, please:

1. Check the [Issues](https://github.com/MrRezaeiUofT/AMG-RAG/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

---

**Note**: This is research software. Please validate results thoroughly before any clinical application.