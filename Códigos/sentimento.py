#!/usr/bin/env python
"""
Usage:
 sentimento [Options] files
 Options:
   -t 0.1    threshold for sentiment classification (default: 0.1)
   -o        output detailed analysis per paragraph
   -c        use CPU instead of GPU (slower but works on all systems)

Output
  - tab-separated-value (TSV) with
         paragraph_num  polarity  classification  text_preview
"""

import argparse
import sys
import re
import os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import spacy
import numpy as np

def load_sentiment_model(use_cpu=False):
    """Load the multilingual BERT model for sentiment analysis"""
    # Determine device
    device = -1 if use_cpu else 0 if torch.cuda.is_available() else -1
    
    # List of models to try (in order of preference)
    models_to_try = [
        "nlptown/bert-base-multilingual-uncased-sentiment",
        "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    ]
    
    for model_name in models_to_try:
        try:
            print(f"Carregando modelo: {model_name}", file=sys.stderr)
            
            # Create sentiment analysis pipeline with truncation
            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=device,
                truncation=True,
                max_length=512,
                return_all_scores=True  # Get all confidence scores
            )
            
            print(f"Modelo carregado com sucesso: {model_name}", file=sys.stderr)
            return sentiment_analyzer, model_name
            
        except Exception as e:
            print(f"Falha ao carregar {model_name}: {e}", file=sys.stderr)
            continue
    
    # If all models fail, try a simple approach
    print("Tentando modelo básico...", file=sys.stderr)
    try:
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            device=device,
            truncation=True,
            max_length=512,
            return_all_scores=True
        )
        return sentiment_analyzer, "default"
    except Exception as e:
        print(f"Falha ao carregar qualquer modelo: {e}", file=sys.stderr)
        sys.exit(1)

def load_spacy_model():
    """Load the Portuguese spaCy model"""
    try:
        return spacy.load("pt_core_news_lg")
    except Exception as e:
        print(f"Aviso: spaCy não disponível: {e}", file=sys.stderr)
        return None

def get_sentiment_score_with_confidence(text, sentiment_analyzer, model_name):
    """Get sentiment score using confidence weighting for more granular values"""
    if not text.strip():
        return 0.0
    
    try:
        # Get all scores from the model
        results = sentiment_analyzer(text)
        
        if isinstance(results, list) and len(results) > 0:
            scores = results[0] if isinstance(results[0], list) else results
        else:
            scores = results
        
        if "nlptown" in model_name:
            # NLP Town model: weighted average of star ratings
            weighted_sum = 0.0
            total_confidence = 0.0
            
            for score_info in scores:
                label = score_info['label']
                confidence = score_info['score']
                
                if 'star' in label:
                    stars = int(label.split()[0])
                    # Convert to -1 to 1 scale and weight by confidence
                    star_value = (stars - 3) / 2
                    weighted_sum += star_value * confidence
                    total_confidence += confidence
            
            return weighted_sum / total_confidence if total_confidence > 0 else 0.0
        
        elif "cardiffnlp" in model_name:
            # Cardiff model: weighted combination of pos/neg/neutral
            pos_score = next((s['score'] for s in scores if s['label'] == 'LABEL_2'), 0)
            neg_score = next((s['score'] for s in scores if s['label'] == 'LABEL_0'), 0)
            neu_score = next((s['score'] for s in scores if s['label'] == 'LABEL_1'), 0)
            
            # Weighted polarity considering neutral as dampening factor
            polarity = (pos_score - neg_score) * (1 - neu_score * 0.5)
            return polarity
        
        else:
            # Default: try to find positive/negative scores
            pos_score = next((s['score'] for s in scores if 'POSITIVE' in s['label'].upper()), 0)
            neg_score = next((s['score'] for s in scores if 'NEGATIVE' in s['label'].upper()), 0)
            return pos_score - neg_score
            
    except Exception as e:
        print(f"Erro ao analisar '{text[:30]}...': {e}", file=sys.stderr)
        return 0.0

def extract_key_phrases(text, nlp=None):
    """Extract key phrases that might carry strong sentiment"""
    key_phrases = []
    
    if nlp:
        doc = nlp(text)
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 3:
                key_phrases.append(chunk.text.strip())
        
        # Extract adjective + noun combinations
        for token in doc:
            if token.pos_ == 'ADJ' and token.head.pos_ == 'NOUN':
                phrase = f"{token.text} {token.head.text}"
                key_phrases.append(phrase)
    else:
        # Fallback: extract common patterns
        # Adjective + noun patterns
        adj_noun_pattern = r'\b(?:muito|bem|mal|pouco|bastante|extremamente)\s+\w+\b'
        key_phrases.extend(re.findall(adj_noun_pattern, text, re.IGNORECASE))
        
        # Negation patterns
        neg_pattern = r'\b(?:não|nunca|jamais|nada)\s+\w+(?:\s+\w+)?\b'
        key_phrases.extend(re.findall(neg_pattern, text, re.IGNORECASE))
    
    return list(set(key_phrases))  # Remove duplicates

def analyze_words_sentiment(text, sentiment_analyzer, model_name, nlp=None):
    """Analyze sentiment of individual meaningful words"""
    if nlp:
        doc = nlp(text)
        words = []
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                not token.is_space and 
                len(token.text) > 2 and
                token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']):
                words.append(token.lemma_.lower())
    else:
        # Fallback without spaCy
        words = re.findall(r'\b\w{3,}\b', text.lower())
        # Remove common Portuguese stop words
        stop_words = {'que', 'para', 'com', 'uma', 'por', 'mais', 'como', 'mas', 'foi', 'ele', 'ela', 'seu', 'sua', 'tem', 'ter', 'ser', 'está', 'isso', 'essa', 'este', 'esta', 'muito', 'bem', 'pode', 'fazer', 'quando', 'onde', 'porque', 'então', 'também', 'ainda', 'assim', 'depois', 'antes', 'entre', 'sobre', 'até', 'sem', 'pelo', 'pela', 'nos', 'das', 'dos', 'uma', 'uns', 'umas'}
        words = [w for w in words if w not in stop_words]
    
    # Analyze sentiment of unique words
    word_sentiments = []
    for word in set(words):
        sentiment = get_sentiment_score_with_confidence(word, sentiment_analyzer, model_name)
        word_sentiments.append(sentiment)
    
    return np.mean(word_sentiments) if word_sentiments else 0.0

def hybrid_sentiment_analysis(text, sentiment_analyzer, model_name, nlp=None):
    """
    Hybrid approach combining multiple sentiment analysis methods
    for more granular and accurate results
    """
    if not text.strip():
        return 0.0
    
    # 1. Overall paragraph sentiment (60% weight)
    overall_sentiment = get_sentiment_score_with_confidence(text, sentiment_analyzer, model_name)
    
    # 2. Average word sentiment (30% weight)
    word_sentiment = analyze_words_sentiment(text, sentiment_analyzer, model_name, nlp)
    
    # 3. Key phrases sentiment (10% weight)
    key_phrases = extract_key_phrases(text, nlp)
    phrase_sentiments = []
    for phrase in key_phrases[:5]:  # Limit to top 5 phrases
        phrase_sentiment = get_sentiment_score_with_confidence(phrase, sentiment_analyzer, model_name)
        phrase_sentiments.append(phrase_sentiment)
    
    avg_phrase_sentiment = np.mean(phrase_sentiments) if phrase_sentiments else 0.0
    
    # 4. Combine with weights
    hybrid_score = (
        overall_sentiment * 0.6 +
        word_sentiment * 0.3 +
        avg_phrase_sentiment * 0.1
    )
    
    # 5. Apply smoothing to avoid extreme values
    # Use tanh to compress extreme values while preserving granularity in the middle range
    smoothed_score = np.tanh(hybrid_score * 1.2) * 0.9
    
    return float(smoothed_score)

def split_text_into_chunks(text, max_chars=2000):
    """Split text into smaller chunks that fit within model limits"""
    if len(text) <= max_chars:
        return [text]
    
    # Try to split by sentences first
    sentences = re.split(r'[.!?]+\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) <= max_chars:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If chunks are still too long, split by words
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            words = chunk.split()
            current_word_chunk = ""
            for word in words:
                if len(current_word_chunk + word) <= max_chars:
                    current_word_chunk += word + " "
                else:
                    if current_word_chunk:
                        final_chunks.append(current_word_chunk.strip())
                    current_word_chunk = word + " "
            if current_word_chunk:
                final_chunks.append(current_word_chunk.strip())
    
    return final_chunks

def analyze_paragraph(text, para_num, sentiment_analyzer, model_name, nlp=None, threshold=0.1):
    """Analyze sentiment of a single paragraph using hybrid approach"""
    if not text.strip():
        return None
    
    try:
        # Use hybrid approach for more granular values
        polarity = hybrid_sentiment_analysis(text, sentiment_analyzer, model_name, nlp)
        
        # Determine classification
        if polarity > threshold:
            classification = "POSITIVE"
        elif polarity < -threshold:
            classification = "NEGATIVE"
        else:
            classification = "NEUTRAL"
        
        # Get first 50 characters as preview
        preview = text.strip()[:50].replace('\n', ' ').replace('\t', ' ')
        if len(text.strip()) > 50:
            preview += "..."
        
        return {
            'paragraph': para_num,
            'polarity': polarity,
            'classification': classification,
            'preview': preview,
            'full_text': text.strip()
        }
        
    except Exception as e:
        print(f"Erro geral ao analisar parágrafo {para_num}: {e}", file=sys.stderr)
        return None

def split_into_paragraphs(text, nlp=None):
    """Split text into paragraphs"""
    # First split by double newlines or single newlines followed by significant whitespace
    paragraphs = re.split(r'\n\s*\n|\n(?=\s{4,})', text)
    
    # Clean and filter paragraphs
    result = []
    for p in paragraphs:
        cleaned = p.strip()
        if cleaned and len(cleaned) > 10:  # Ignore very short paragraphs
            result.append(cleaned)
    
    return result

def print_results(results, detailed=False):
    """Print analysis results in TSV format - always includes neutral paragraphs"""
    
    # Print header
    header = ["paragraph", "polarity", "classification", "preview"]
    print("\t".join(header))
    
    # Print results (always includes all paragraphs)
    for result in results:
        row = [
            str(result['paragraph']),
            f"{result['polarity']:.6f}",  # More decimal places to show granularity
            result['classification'],
            result['preview']
        ]
        
        print("\t".join(row))
        
        if detailed:
            print(f"Full text: {result['full_text']}")
            print("-" * 50)

def read_file(filename):
    """Read text file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description='Análise híbrida de sentimento em português (valores mais granulares)')
    parser.add_argument('files', nargs='+', help='Arquivos de texto para analisar')
    parser.add_argument('-t', '--threshold', type=float, default=0.1,
                       help='Threshold para classificação de sentimento (default: 0.1)')
    parser.add_argument('-o', '--detailed', action='store_true',
                       help='Mostrar análise detalhada com texto completo do parágrafo')
    parser.add_argument('-c', '--cpu', action='store_true',
                       help='Usar CPU em vez de GPU (mais lento mas funciona em todos os sistemas)')
    
    args = parser.parse_args()
    
    print("Carregando modelos...", file=sys.stderr)
    
    # Load models
    sentiment_analyzer, model_name = load_sentiment_model(args.cpu)
    nlp = load_spacy_model()
    
    print(f"Modelos carregados! Usando: {model_name}", file=sys.stderr)
    
    all_results = []
    
    for filename in args.files:
        print(f"Analisando {filename}...", file=sys.stderr)
        text = read_file(filename)
        if text is None:
            continue
            
        paragraphs = split_into_paragraphs(text, nlp)
        print(f"Encontrados {len(paragraphs)} parágrafos", file=sys.stderr)
        
        for i, paragraph in enumerate(paragraphs, 1):
            print(f"Processando parágrafo {i}/{len(paragraphs)}", file=sys.stderr)
            result = analyze_paragraph(paragraph, i, sentiment_analyzer, model_name, nlp, args.threshold)
            if result:
                all_results.append(result)
    
    if all_results:
        print_results(all_results, args.detailed)
    else:
        print("Nenhum parágrafo encontrado para análise.", file=sys.stderr)

if __name__ == "__main__":
    main()
