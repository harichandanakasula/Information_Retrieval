# Web-Scale Relation Extraction using SpanBERT and Gemini

This project implements a dual-mode information extraction pipeline using both a **pretrained SpanBERT model** and **Google Gemini 2.0 API**. It extracts structured relations (e.g., *Works_For*, *Attended*, etc.) from unstructured web content retrieved via Google Search, applying large language models and deep NLP techniques.

---

## Project Overview 
- **Tech Stack**: Python, HuggingFace Transformers, spaCy, Google Gemini API, Google Search API  
- **Mode**: Command-line application deployed/tested on GCP

---

##  How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### Step 2: Execute the Pipeline

```bash
# For SpanBERT mode
python3 main.py spanbert <GOOGLE_API_KEY> <SEARCH_ENGINE_ID> <GEMINI_KEY> <RELATION_ID> <THRESHOLD> "<QUERY>" <MAX_TUPLES>

# For Gemini mode
python3 main.py gemini <GOOGLE_API_KEY> <SEARCH_ENGINE_ID> <GEMINI_KEY> <RELATION_ID> <THRESHOLD> "<QUERY>" <MAX_TUPLES>
```

### Example:

```bash
python3 main.py spanbert AI_KEY CSE_ID GEMINI_KEY 1 0.7 "bill gates microsoft" 10
```

---

## 🔁 Iterative Relation Extraction Workflow

1. Start with a **seed query** (e.g., "sergey brin stanford").
2. Use Google Custom Search API to fetch top 10 URLs.
3. For each unseen URL:
   - Extract page content using `requests` + `BeautifulSoup`
   - Limit raw text to first 10,000 characters
   - Use `spaCy` to tokenize and extract named entities
   - For `spanbert`: classify valid entity pairs
   - For `gemini`: prompt and parse relation triples
4. All unique (subject, object) pairs are added to result set.
5. New queries are generated from extracted entities and process repeats.

---

## Project Structure

| File                     | Purpose |
|--------------------------|---------|
| `main.py`               | Core orchestration logic |
| `spanbert.py`           | Loads and runs SpanBERT model |
| `gemini_helper_6111.py` | Builds prompts and parses Gemini API output |
| `spacy_help_functions.py` | Named entity recognition and entity pair extraction |
| `example_relations.py`  | Demonstration script |
| `relations.txt`         | Relation types supported |
| `requirements.txt`      | Python dependencies |

---

## Supported Relations

- `Schools_Attended`  
- `Work_For`  
- `Live_In`  
- `Top_Member_Employees`

---

## Libraries Used

- `transformers`, `torch`, `pytorch_pretrained_bert` for SpanBERT
- `spaCy (en_core_web_lg)` for NER and sentence parsing
- `requests`, `bs4`, `ast`, `re` for scraping and parsing

---

## Key Highlights

- Hybrid extraction approach using both **pretrained models** and **LLMs**
- Iterative query expansion strategy
- Modular architecture
- Clean output formatting and duplicate filtering
- Compatible with academic or real-world entity extraction tasks

---

