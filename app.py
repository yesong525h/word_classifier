import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from pathlib import Path
import numpy as np
import pandas as pd
from deep_translator import GoogleTranslator

# Set page config
st.set_page_config(
    page_title="Word Classifier - Unknown Word Translator",
    page_icon="📚",
    layout="wide"
)

# Cache the word dictionary loading
@st.cache_data
def load_word_dictionary():
    """Load the word labels CSV file"""
    csv_path = Path(__file__).parent / "word_labels_final.csv"
    try:
        df = pd.read_csv(csv_path)
        # Create dictionary: word -> label (0=known, 1=unknown)
        word_dict = dict(zip(df['word'], df['label']))
        return word_dict
    except Exception as e:
        st.warning(f"Could not load word dictionary: {str(e)}")
        return {}

# Cache the model and tokenizer loading
@st.cache_resource
def load_model():
    """Load the model and tokenizer from the local directory"""
    model_path = Path(__file__).parent
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        model.eval()  # Set to evaluation mode
        
        # Get number of labels from model config
        num_labels = model.config.num_labels
        
        # Get label names if available
        id2label = getattr(model.config, 'id2label', None)
        label2id = getattr(model.config, 'label2id', None)
        
        return tokenizer, model, num_labels, id2label, label2id
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None

def clean_korean_word(word):
    """Remove Korean particles (조사) and punctuation from word"""
    # Remove punctuation
    word = word.strip(".,?!\"'""''·()[]{}")
    
    # Remove Korean particles (조사) at the end
    josa_pattern = r"(은|는|이|가|을|를|에|에서|으로|로|과|와|에게|한테|부터|까지|마다|조차|라도|처럼|밖에|뿐|■)"
    word = re.sub(josa_pattern + r"\s*$", "", word)
    
    return word

def classify_word(word, tokenizer, model, num_labels, id2label=None):
    """Classify a single word to determine if it's unknown"""
    # Tokenize the word (using max_length=64 as in training)
    inputs = tokenizer(
        word,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get predicted class and confidence
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()
    
    # Get label name if available
    # Note: LABEL_0 = unknown (어려운 단어), LABEL_1 = known (아는 단어)
    label_name = id2label.get(predicted_class, f"LABEL_{predicted_class}") if id2label else f"LABEL_{predicted_class}"
    
    return predicted_class, confidence, label_name, probabilities[0].tolist()

def process_text(text, tokenizer, model, num_labels, word_dict, translator, id2label=None):
    """Process text to identify unknown words using both dictionary and model"""
    # Split text into words
    raw_words = text.split()
    seen_words = set()  # Track processed words to avoid duplicates
    results = []
    unknown_words = []
    current_pos = 0
    
    # Process each word
    for raw_word in raw_words:
        # Clean the word (remove particles and punctuation)
        clean_word = clean_korean_word(raw_word)
        
        # Skip empty, single character, or numeric words
        if not clean_word or len(clean_word) == 1:
            # Still track position for punctuation
            pos = text.find(raw_word, current_pos)
            if pos != -1:
                results.append({
                    'word': raw_word,
                    'clean_word': clean_word,
                    'start': pos,
                    'end': pos + len(raw_word),
                    'is_unknown': False,
                    'confidence': 0.0,
                    'class': None,
                    'source': 'skipped'
                })
                current_pos = pos + len(raw_word)
            continue
        
        # Skip if contains digits
        if any(ch.isdigit() for ch in clean_word):
            pos = text.find(raw_word, current_pos)
            if pos != -1:
                results.append({
                    'word': raw_word,
                    'clean_word': clean_word,
                    'start': pos,
                    'end': pos + len(raw_word),
                    'is_unknown': False,
                    'confidence': 0.0,
                    'class': None,
                    'source': 'numeric'
                })
                current_pos = pos + len(raw_word)
            continue
        
        # Skip if already processed (avoid duplicates)
        if clean_word in seen_words:
            pos = text.find(raw_word, current_pos)
            if pos != -1:
                results.append({
                    'word': raw_word,
                    'clean_word': clean_word,
                    'start': pos,
                    'end': pos + len(raw_word),
                    'is_unknown': False,
                    'confidence': 0.0,
                    'class': None,
                    'source': 'duplicate'
                })
                current_pos = pos + len(raw_word)
            continue
        
        seen_words.add(clean_word)
        
        # Find position in original text
        pos = text.find(raw_word, current_pos)
        if pos == -1:
            continue
        start, end = pos, pos + len(raw_word)
        current_pos = pos + len(raw_word)
        
        # First check the dictionary (ground truth)
        dict_label = word_dict.get(clean_word, None)
        
        # Also get model prediction
        predicted_class, confidence, label_name, probs = classify_word(
            clean_word, tokenizer, model, num_labels, id2label
        )
        
        # Determine if word is unknown
        # Note: In CSV: 0 = unknown (needs translation), 1 = known (you know this)
        # Model: LABEL_0 (class 0) = unknown, LABEL_1 (class 1) = known
        # Priority: dictionary > model prediction
        if dict_label is not None:
            # Word is in dictionary: 0 = unknown (needs translation), 1 = known
            is_unknown = (dict_label == 0)
            source = 'dictionary'
        else:
            # Word not in dictionary, use model prediction
            # Model: LABEL_0 (class 0) = unknown, LABEL_1 (class 1) = known
            is_unknown = (predicted_class == 0)
            source = 'model'
        
        # Get Spanish translation if unknown
        spanish_translation = None
        if is_unknown and translator:
            spanish_translation = get_spanish_translation(clean_word, translator)
        
        results.append({
            'word': raw_word,
            'clean_word': clean_word,
            'start': start,
            'end': end,
            'is_unknown': is_unknown,
            'confidence': confidence,
            'class': predicted_class,
            'dict_label': dict_label,
            'label': label_name,
            'source': source,
            'spanish_translation': spanish_translation,
            'probabilities': probs
        })
        
        if is_unknown:
            unknown_words.append({
                'word': clean_word,
                'raw_word': raw_word,
                'confidence': confidence,
                'class': predicted_class,
                'dict_label': dict_label,
                'label': label_name,
                'source': source,
                'spanish_translation': spanish_translation
            })
    
    return results, unknown_words

@st.cache_resource
def get_translator():
    """Initialize Google Translator for Korean to Spanish"""
    try:
        return GoogleTranslator(source='ko', target='es')
    except Exception as e:
        st.warning(f"Translation service unavailable: {str(e)}")
        return None

def get_spanish_translation(word, translator):
    """Get Spanish translation for a Korean word"""
    if translator is None:
        return "[Translation unavailable]"
    
    try:
        translated = translator.translate(word)
        # Remove quotes from translation result
        translated = translated.strip('"\'""''')
        return translated
    except Exception as e:
        return f"[Translation error: {str(e)}]"

def main():
    st.title("📚 Unknown Word Classifier & Spanish Translator")
    st.markdown("Upload a text file or paste text to identify words you might not know and get Spanish translations.")
    
    # Load word dictionary, model, and translator
    with st.spinner("Loading word dictionary, model, and translation service..."):
        word_dict = load_word_dictionary()
        tokenizer, model, num_labels, id2label, label2id = load_model()
        translator = get_translator()
    
    if tokenizer is None or model is None:
        st.error("Failed to load model. Please check the model files.")
        return
    
    # Display model info
    label_info = f"Labels: {id2label}" if id2label else f"{num_labels} classes"
    dict_size = len(word_dict) if word_dict else 0
    translator_status = "✅ Available" if translator else "❌ Unavailable"
    st.success(f"Model loaded! ({label_info}) | Dictionary: {dict_size:,} words | Translation: {translator_status}")
    
    if id2label:
        with st.expander("Model Labels", expanded=False):
            st.json(id2label)
    
    # File uploader
    st.subheader("Upload Text File")
    uploaded_file = st.file_uploader(
        "Choose a text file",
        type=['txt'],
        help="Upload a .txt file containing text to analyze"
    )
    
    # Text input alternative
    st.subheader("Or Enter Text Directly")
    text_input = st.text_area(
        "Paste text here",
        height=200,
        help="You can also paste text directly here"
    )
    
    # Processing options
    col1, col2 = st.columns(2)
    with col1:
        show_all_words = st.checkbox("Show all words (not just unknown)", value=False)
    with col2:
        min_confidence = st.slider("Minimum confidence threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Process button
    if st.button("Analyze Text", type="primary"):
        text_to_process = None
        
        # Determine which input to use
        if uploaded_file is not None:
            text_to_process = uploaded_file.read().decode('utf-8')
            st.info(f"Processing uploaded file: {uploaded_file.name}")
        elif text_input.strip():
            text_to_process = text_input.strip()
            st.info("Processing text input")
        else:
            st.warning("Please upload a file or enter text to analyze.")
            return
        
        if text_to_process:
            # Process the text
            with st.spinner("Analyzing words and translating..."):
                results, unknown_words = process_text(
                    text_to_process, tokenizer, model, num_labels, word_dict, translator, id2label
                )
            
            # Filter unknown words by confidence
            filtered_unknown = [w for w in unknown_words if w['confidence'] >= min_confidence]
            
            # Display summary
            st.subheader("📊 Analysis Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Words", len([r for r in results if re.match(r'\w+', r['word'])]))
            with col2:
                st.metric("Unknown Words", len(filtered_unknown))
            with col3:
                if len(results) > 0:
                    unknown_pct = (len(filtered_unknown) / len([r for r in results if re.match(r'\w+', r['word'])])) * 100
                    st.metric("Unknown %", f"{unknown_pct:.1f}%")
            
            # Display text with highlighted unknown words
            st.subheader("📝 Text with Highlighted Unknown Words")
            
            # Create highlighted text
            highlighted_html = "<div style='font-size: 16px; line-height: 1.8; padding: 20px; background: #f8f9fa; border-radius: 10px;'>"
            last_end = 0
            
            for result in results:
                # Add text before this word
                if result['start'] > last_end:
                    highlighted_html += text_to_process[last_end:result['start']].replace('\n', '<br>')
                
                # Add the word with or without highlighting
                word_html = result['word']  # Use original word for display
                if result['is_unknown'] and result['confidence'] >= min_confidence:
                    conf_pct = f"{result['confidence']:.2%}"
                    translation = result.get('spanish_translation', '')
                    tooltip = f"Unknown word (confidence: {conf_pct})"
                    if translation:
                        tooltip += f" | Spanish: {translation}"
                    word_html = f"<mark style='background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; cursor: help;' title='{tooltip}'>{result['word']}</mark>"
                
                highlighted_html += word_html
                last_end = result['end']
            
            # Add remaining text
            if last_end < len(text_to_process):
                highlighted_html += text_to_process[last_end:].replace('\n', '<br>')
            
            highlighted_html += "</div>"
            st.markdown(highlighted_html, unsafe_allow_html=True)
            
            # Display unknown words with translations
            if filtered_unknown:
                st.subheader("🔍 Unknown Words & Spanish Translations")
                
                # Display table
                df_data = []
                for word_info in filtered_unknown:
                    word = word_info['word']
                    df_data.append({
                        'Word': word,
                        'Spanish Translation': word_info.get('spanish_translation', '[Translation unavailable]'),
                        'Source': word_info.get('source', 'N/A'),
                        'Dict Label': 'Known (1)' if word_info.get('dict_label') == 1 else ('Unknown (0)' if word_info.get('dict_label') == 0 else 'Not in dict'),
                        'Model Class': f"LABEL_{word_info['class']}",
                        'Confidence': f"{word_info['confidence']:.2%}"
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Download translations
                st.download_button(
                    label="📥 Download Translations",
                    data=pd.DataFrame(df_data).to_csv(index=False),
                    file_name="translations.csv",
                    mime="text/csv"
                )
            else:
                st.info("No unknown words found (or all below confidence threshold).")
            
            # Show detailed word analysis if requested
            if show_all_words:
                st.subheader("📋 Detailed Word Analysis")
                detailed_data = []
                for result in results:
                    if result.get('clean_word') and result.get('source') not in ['punctuation', 'skipped', 'numeric', 'duplicate']:
                        detailed_data.append({
                            'Word': result.get('clean_word', result['word']),
                            'Is Unknown': 'Yes' if result['is_unknown'] else 'No',
                            'Source': result.get('source', 'N/A'),
                            'Dict Label': 'Known (1)' if result.get('dict_label') == 1 else ('Unknown (0)' if result.get('dict_label') == 0 else 'Not in dict'),
                            'Model Class': f"LABEL_{result['class']}" if result['class'] is not None else 'N/A',
                            'Confidence': f"{result['confidence']:.2%}" if result['confidence'] > 0 else 'N/A',
                            'Spanish Translation': result.get('spanish_translation', '') if result['is_unknown'] else ''
                        })
                
                df_detailed = pd.DataFrame(detailed_data)
                st.dataframe(df_detailed, use_container_width=True, hide_index=True)
    
    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This tool identifies Korean words you might not know in a text
        and provides Spanish translations.
        
        **How it works:**
        1. Upload a text file or paste Korean text
        2. Words are checked against a dictionary (3,715 words)
        3. Unknown words (label=0) are highlighted
        4. Spanish translations are provided for unknown words
        
        **Features:**
        - Dictionary lookup (word_labels_final.csv)
        - BERT model classification
        - Word-by-word analysis
        - Confidence scores
        - Highlighted unknown words
        - Spanish translations
        - Export to CSV
        
        **Label System:**
        - **0** = Unknown word (needs translation)
        - **1** = Known word (you know this)
        """)
        
        st.header("Instructions")
        st.markdown("""
        1. Upload a `.txt` file or paste text
        2. Adjust confidence threshold if needed
        3. Click "Analyze Text"
        4. Review highlighted words and translations
        """)
        
        if id2label:
            st.header("Model Labels")
            st.json(id2label)

if __name__ == "__main__":
    main()
