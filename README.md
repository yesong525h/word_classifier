# Word Classifier - Unknown Word Identifier & Spanish Translator

A Streamlit web application that identifies words you might not know in a text and provides Spanish translations using a BERT-based classification model.

## Purpose

This tool helps language learners by:
- Analyzing text word-by-word
- Identifying unknown or difficult words
- Providing Spanish translations for those words
- Highlighting unknown words in the original text

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Demo

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Upload a text file**: Click "Browse files" and select a `.txt` file
2. **Or enter text directly**: Paste text in the text area
3. **Adjust settings** (optional):
   - Toggle "Show all words" to see analysis for every word
   - Adjust confidence threshold to filter results
4. **Click "Analyze Text"**: The model will:
   - Process each word in the text
   - Identify unknown/difficult words
   - Highlight them in the original text
   - Display Spanish translations
5. **Download results**: Export translations to CSV

## Features

- ✅ Word-by-word analysis
- ✅ Confidence scores for each classification
- ✅ Visual highlighting of unknown words
- ✅ Spanish translation table
- ✅ Export to CSV
- ✅ Detailed word analysis view
- ✅ Adjustable confidence threshold

## Model Information

- **Architecture**: BERT (BertForSequenceClassification)
- **Max Sequence Length**: 512 tokens
- **Model Type**: Single-label classification
- **Tokenizer**: BERT tokenizer with WordPiece
- **Purpose**: Classifies words as "known" or "unknown" to identify vocabulary gaps

## Translation Note

The current implementation includes a placeholder translation function. To get actual Spanish translations, you can:

1. **Integrate a translation API** (Google Translate, DeepL, etc.)
2. **Use a translation model** (e.g., Helsinki-NLP models)
3. **Use a translation dictionary** if you have one

Example integration:
```python
from googletrans import Translator

def get_spanish_translation(word):
    translator = Translator()
    result = translator.translate(word, src='en', dest='es')
    return result.text
```

## Files

- `app.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- `model.safetensors`: Model weights
- `config.json`: Model configuration
- `tokenizer.json`, `tokenizer_config.json`: Tokenizer files
- `vocab.txt`: Vocabulary file
- `sample_news.txt`: Sample text file for testing
