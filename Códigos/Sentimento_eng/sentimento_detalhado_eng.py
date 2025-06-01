#!/usr/bin/env python
"""
Usage:
 sentiment_detailed_english [Options] files
 Options:
   -t 0.1    threshold for sentiment classification (default: 0.1)
   -w        show word-by-word analysis
   -s        show statistics summary (TSV format)
   -c        use CPU instead of GPU

Output
  - detailed analysis showing word-level sentiment ratios (always includes neutral paragraphs)
"""

import argparse
import sys
import re
import os
from transformers import pipeline
import torch
import spacy
import numpy as np
from collections import Counter, defaultdict

def load_sentiment_model(use_cpu=False):
    """Load the multilingual BERT model for sentiment analysis"""
    device = -1 if use_cpu else 0 if torch.cuda.is_available() else -1
    
    models_to_try = [
        "nlptown/bert-base-multilingual-uncased-sentiment",
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ]
    
    for model_name in models_to_try:
        try:
            print(f"Loading model: {model_name}", file=sys.stderr)
            
            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=device,
                truncation=True,
                max_length=512,
                return_all_scores=True
            )
            
            print(f"Model loaded: {model_name}", file=sys.stderr)
            return sentiment_analyzer, model_name
            
        except Exception as e:
            print(f"Failed: {e}", file=sys.stderr)
            continue
    
    sys.exit(1)

def load_spacy_model():
    """Load English spaCy model - specifically en_core_web_lg"""
    try:
        print("Loading spaCy model: en_core_web_lg", file=sys.stderr)
        nlp = spacy.load("en_core_web_lg")
        print("Successfully loaded en_core_web_lg", file=sys.stderr)
        return nlp
    except OSError as e:
        print(f"Error: en_core_web_lg not found: {e}", file=sys.stderr)
        print("Please install it with: python -m spacy download en_core_web_lg", file=sys.stderr)
        print("Continuing without spaCy (reduced functionality)...", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: spaCy error: {e}", file=sys.stderr)
        print("Continuing without spaCy (reduced functionality)...", file=sys.stderr)
        return None

def get_sentiment_score_with_confidence(text, sentiment_analyzer, model_name):
    """Get sentiment score using confidence weighting for more granular values"""
    if not text.strip():
        return 0.0
    
    try:
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
                    star_value = (stars - 3) / 2
                    weighted_sum += star_value * confidence
                    total_confidence += confidence
            
            return weighted_sum / total_confidence if total_confidence > 0 else 0.0
        
        elif "cardiffnlp" in model_name:
            # Handle different Cardiff model formats
            if any('LABEL_' in s['label'] for s in scores):
                pos_score = next((s['score'] for s in scores if s['label'] == 'LABEL_2'), 0)
                neg_score = next((s['score'] for s in scores if s['label'] == 'LABEL_0'), 0)
                neu_score = next((s['score'] for s in scores if s['label'] == 'LABEL_1'), 0)
            else:
                pos_score = next((s['score'] for s in scores if 'POSITIVE' in s['label'].upper()), 0)
                neg_score = next((s['score'] for s in scores if 'NEGATIVE' in s['label'].upper()), 0)
                neu_score = next((s['score'] for s in scores if 'NEUTRAL' in s['label'].upper()), 0)
            
            polarity = (pos_score - neg_score) * (1 - neu_score * 0.5)
            return polarity
        
        else:
            pos_score = next((s['score'] for s in scores if 'POSITIVE' in s['label'].upper()), 0)
            neg_score = next((s['score'] for s in scores if 'NEGATIVE' in s['label'].upper()), 0)
            return pos_score - neg_score
            
    except Exception as e:
        print(f"Error analyzing '{text[:30]}...': {e}", file=sys.stderr)
        return 0.0

def extract_key_phrases(text, nlp=None):
    """Extract key phrases that might carry strong sentiment"""
    key_phrases = []
    
    if nlp:
        try:
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
                    
            # Extract verb phrases with strong sentiment
            for token in doc:
                if token.pos_ == 'VERB' and any(child.pos_ == 'ADV' for child in token.children):
                    phrase_parts = [token.text]
                    for child in token.children:
                        if child.pos_ in ['ADV', 'PART']:
                            phrase_parts.append(child.text)
                    if len(phrase_parts) > 1:
                        key_phrases.append(' '.join(phrase_parts))
        except Exception as e:
            print(f"Error in spaCy processing: {e}", file=sys.stderr)
            # Fall back to regex patterns
            pass
    
    # Fallback or additional patterns: extract common English patterns
    adj_adv_pattern = r'\b(?:very|really|extremely|quite|rather|pretty|fairly|totally|completely|absolutely|incredibly|amazingly|terribly|horribly|wonderfully|beautifully|perfectly|utterly|entirely)\s+\w+\b'
    key_phrases.extend(re.findall(adj_adv_pattern, text, re.IGNORECASE))
    
    neg_pattern = r'\b(?:not|never|no|nothing|nowhere|nobody|none|neither|nor)\s+\w+(?:\s+\w+)?\b'
    key_phrases.extend(re.findall(neg_pattern, text, re.IGNORECASE))
    
    # Intensifier patterns
    intensifier_pattern = r'\b(?:so|such|too|way too|far too|much too)\s+\w+\b'
    key_phrases.extend(re.findall(intensifier_pattern, text, re.IGNORECASE))
    
    return list(set(key_phrases))

def analyze_words_sentiment_detailed(paragraph, sentiment_analyzer, model_name, nlp=None):
    """Analyze sentiment of individual words/phrases in a paragraph with hybrid approach"""
    
    # Get meaningful words
    if nlp:
        try:
            doc = nlp(paragraph)
            words = []
            word_info = []
            for token in doc:
                if (not token.is_stop and 
                    not token.is_punct and 
                    not token.is_space and 
                    len(token.text) > 2 and
                    token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']):
                    words.append(token.lemma_.lower())
                    word_info.append({
                        'word': token.lemma_.lower(),
                        'pos': token.pos_,
                        'original': token.text
                    })
        except Exception as e:
            print(f"Error in spaCy word analysis: {e}", file=sys.stderr)
            nlp = None  # Fall back to regex
    
    if not nlp:
        # Fallback without spaCy
        words = re.findall(r'\b\w{3,}\b', paragraph.lower())
        # Comprehensive English stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 
            'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 
            'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'have', 'will', 'with', 'this', 'that', 'they', 
            'from', 'been', 'said', 'each', 'make', 'most', 'move', 'must', 'name', 'part', 'take', 'than', 'time', 
            'very', 'when', 'come', 'here', 'just', 'like', 'long', 'many', 'over', 'such', 'them', 'well', 'were',
            'what', 'your', 'into', 'only', 'some', 'could', 'other', 'after', 'first', 'also', 'back', 'any', 'good',
            'woman', 'through', 'us', 'life', 'child', 'there', 'work', 'down', 'may', 'call', 'still', 'should',
            'over', 'think', 'where', 'much', 'before', 'right', 'too', 'does', 'three', 'small', 'another', 'while',
            'here', 'why', 'ask', 'went', 'men', 'read', 'need', 'land', 'different', 'home', 'move', 'try', 'kind',
            'hand', 'picture', 'again', 'change', 'off', 'play', 'spell', 'air', 'away', 'animal', 'house', 'point',
            'page', 'letter', 'mother', 'answer', 'found', 'study', 'still', 'learn', 'should', 'america', 'world'
        }
        words = [w for w in words if w not in stop_words]
        word_info = [{'word': w, 'pos': 'UNKNOWN', 'original': w} for w in words]
    
    # Analyze sentiment of individual words with confidence weighting
    word_sentiments = {}
    positive_words = []
    negative_words = []
    neutral_words = []
    
    for info in word_info:
        word = info['word']
        if word in word_sentiments:  # Skip duplicates
            continue
            
        sentiment = get_sentiment_score_with_confidence(word, sentiment_analyzer, model_name)
        word_sentiments[word] = {
            'sentiment': sentiment,
            'pos': info['pos'],
            'original': info['original']
        }
        
        if sentiment > 0.1:
            positive_words.append((word, sentiment, info['pos']))
        elif sentiment < -0.1:
            negative_words.append((word, sentiment, info['pos']))
        else:
            neutral_words.append((word, sentiment, info['pos']))
    
    return {
        'word_sentiments': word_sentiments,
        'positive_words': sorted(positive_words, key=lambda x: x[1], reverse=True),
        'negative_words': sorted(negative_words, key=lambda x: x[1]),
        'neutral_words': neutral_words,
        'total_words': len(words)
    }

def hybrid_sentiment_analysis_detailed(text, sentiment_analyzer, model_name, nlp=None):
    """
    Detailed hybrid approach combining multiple sentiment analysis methods
    """
    if not text.strip():
        return {
            'overall_sentiment': 0.0,
            'word_sentiment': 0.0,
            'phrase_sentiment': 0.0,
            'hybrid_score': 0.0,
            'confidence_factors': {}
        }
    
    # 1. Overall paragraph sentiment
    overall_sentiment = get_sentiment_score_with_confidence(text, sentiment_analyzer, model_name)
    
    # 2. Word-level analysis
    word_analysis = analyze_words_sentiment_detailed(text, sentiment_analyzer, model_name, nlp)
    word_sentiments = [info['sentiment'] for info in word_analysis['word_sentiments'].values()]
    word_sentiment = np.mean(word_sentiments) if word_sentiments else 0.0
    
    # 3. Key phrases sentiment
    key_phrases = extract_key_phrases(text, nlp)
    phrase_sentiments = []
    phrase_details = []
    
    for phrase in key_phrases[:5]:  # Limit to top 5 phrases
        phrase_sentiment = get_sentiment_score_with_confidence(phrase, sentiment_analyzer, model_name)
        phrase_sentiments.append(phrase_sentiment)
        phrase_details.append({
            'phrase': phrase,
            'sentiment': phrase_sentiment
        })
    
    avg_phrase_sentiment = np.mean(phrase_sentiments) if phrase_sentiments else 0.0
    
    # 4. Calculate confidence factors
    text_length_factor = min(len(text) / 100, 1.0)  # Longer texts get higher confidence
    word_count_factor = min(len(word_sentiments) / 10, 1.0)  # More words = higher confidence
    phrase_count_factor = min(len(phrase_sentiments) / 3, 1.0)  # More phrases = higher confidence
    
    # 5. Adaptive weighting based on confidence
    overall_weight = 0.6 * text_length_factor
    word_weight = 0.3 * word_count_factor
    phrase_weight = 0.1 * phrase_count_factor
    
    # Normalize weights
    total_weight = overall_weight + word_weight + phrase_weight
    if total_weight > 0:
        overall_weight /= total_weight
        word_weight /= total_weight
        phrase_weight /= total_weight
    else:
        overall_weight, word_weight, phrase_weight = 0.6, 0.3, 0.1
    
    # 6. Combine with adaptive weights
    hybrid_score = (
        overall_sentiment * overall_weight +
        word_sentiment * word_weight +
        avg_phrase_sentiment * phrase_weight
    )
    
    # 7. Apply smoothing with context awareness
    # More aggressive smoothing for extreme values
    if abs(hybrid_score) > 0.8:
        smoothed_score = np.tanh(hybrid_score * 0.8) * 0.95
    else:
        smoothed_score = np.tanh(hybrid_score * 1.2) * 0.9
    
    return {
        'overall_sentiment': overall_sentiment,
        'word_sentiment': word_sentiment,
        'phrase_sentiment': avg_phrase_sentiment,
        'hybrid_score': float(smoothed_score),
        'confidence_factors': {
            'text_length_factor': text_length_factor,
            'word_count_factor': word_count_factor,
            'phrase_count_factor': phrase_count_factor,
            'overall_weight': overall_weight,
            'word_weight': word_weight,
            'phrase_weight': phrase_weight
        },
        'word_analysis': word_analysis,
        'phrase_details': phrase_details
    }

def analyze_paragraph_detailed(text, para_num, sentiment_analyzer, model_name, nlp, threshold=0.1):
    """Detailed analysis of a paragraph using hybrid approach"""
    if not text.strip():
        return None
    
    # Use hybrid approach
    analysis = hybrid_sentiment_analysis_detailed(text, sentiment_analyzer, model_name, nlp)
    
    overall_polarity = analysis['hybrid_score']
    word_analysis = analysis['word_analysis']
    avg_word_sentiment = analysis['word_sentiment']
    
    # Classify overall sentiment
    if overall_polarity > threshold:
        classification = "POSITIVE"
    elif overall_polarity < -threshold:
        classification = "NEGATIVE"
    else:
        classification = "NEUTRAL"
    
    # Calculate ratios
    total_words = word_analysis['total_words']
    positive_count = len(word_analysis['positive_words'])
    negative_count = len(word_analysis['negative_words'])
    neutral_count = len(word_analysis['neutral_words'])
    
    positive_ratio = positive_count / total_words if total_words > 0 else 0
    negative_ratio = negative_count / total_words if total_words > 0 else 0
    neutral_ratio = neutral_count / total_words if total_words > 0 else 0
    
    # Get preview
    preview = text.strip()[:50].replace('\n', ' ').replace('\t', ' ')
    if len(text.strip()) > 50:
        preview += "..."
    
    return {
        'paragraph': para_num,
        'overall_polarity': overall_polarity,
        'classification': classification,
        'preview': preview,
        'full_text': text.strip(),
        'word_analysis': word_analysis,
        'total_words': total_words,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'positive_ratio': positive_ratio,
        'negative_ratio': negative_ratio,
        'neutral_ratio': neutral_ratio,
        'avg_word_sentiment': avg_word_sentiment,
        'component_breakdown': {
            'overall_component': analysis['overall_sentiment'],
            'word_component': analysis['word_sentiment'],
            'phrase_component': analysis['phrase_sentiment']
        }
    }

def print_detailed_results(results, show_words=False):
    """Print detailed hybrid analysis results"""
    
    print("=== DETAILED SENTIMENT ANALYSIS ===\n")
    
    for result in results:
        print(f"PARAGRAPH {result['paragraph']}: {result['classification']}")
        print(f"Preview: {result['preview']}")
        print(f"Overall Polarity: {result['overall_polarity']:.6f}")
        print(f"Average Word Polarity: {result['avg_word_sentiment']:.6f}")
        print()
        
        print(f"WORD STATISTICS:")
        print(f"  Total words analyzed: {result['total_words']}")
        print(f"  Positive words: {result['positive_count']} ({result['positive_ratio']:.1%})")
        print(f"  Negative words: {result['negative_count']} ({result['negative_ratio']:.1%})")
        print(f"  Neutral words: {result['neutral_count']} ({result['neutral_ratio']:.1%})")
        print()
        
        # Show top positive and negative words with more detail
        if result['word_analysis']['positive_words']:
            print("MOST POSITIVE WORDS:")
            for word, score, pos in result['word_analysis']['positive_words'][:5]:
                print(f"  {word} ({pos}): {score:.6f}")
            print()
        
        if result['word_analysis']['negative_words']:
            print("MOST NEGATIVE WORDS:")
            for word, score, pos in result['word_analysis']['negative_words'][:5]:
                print(f"  {word} ({pos}): {score:.6f}")
            print()
        
        if show_words:
            print("ALL ANALYZED WORDS:")
            for word, info in sorted(result['word_analysis']['word_sentiments'].items(), 
                                   key=lambda x: x[1]['sentiment'], reverse=True):
                print(f"  {word} ({info['pos']}): {info['sentiment']:.6f}")
            print()
        
        print("=" * 60)
        print()

def print_summary_table(results):
    """Print summary table in TSV format"""
    
    # Header
    header = ["paragraph", "overall_polarity", "avg_word_polarity", "classification", 
              "total_words", "positive_words", "negative_words", "pos_ratio", "neg_ratio", "preview"]
    print("\t".join(header))
    
    # Data
    for result in results:
        row = [
            str(result['paragraph']),
            f"{result['overall_polarity']:.6f}",
            f"{result['avg_word_sentiment']:.6f}",
            result['classification'],
            str(result['total_words']),
            str(result['positive_count']),
            str(result['negative_count']),
            f"{result['positive_ratio']:.6f}",
            f"{result['negative_ratio']:.6f}",
            result['preview']
        ]
        print("\t".join(row))

def split_into_paragraphs(text):
    """Split text into paragraphs"""
    paragraphs = re.split(r'\n\s*\n|\n(?=\s{4,})', text)
    result = []
    for p in paragraphs:
        cleaned = p.strip()
        if cleaned and len(cleaned) > 10:
            result.append(cleaned)
    return result

def read_file(filename):
    """Read text file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description='Detailed sentiment analysis with granular values (English - requires en_core_web_lg)')
    parser.add_argument('files', nargs='+', help='Text files to analyze')
    parser.add_argument('-t', '--threshold', type=float, default=0.1,
                       help='Threshold for classification (default: 0.1)')
    parser.add_argument('-w', '--show-words', action='store_true',
                       help='Show word-by-word analysis')
    parser.add_argument('-s', '--summary', action='store_true',
                       help='Show only summary table (TSV)')
    parser.add_argument('-c', '--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    print("Loading models...", file=sys.stderr)
    sentiment_analyzer, model_name = load_sentiment_model(args.cpu)
    nlp = load_spacy_model()
    
    if nlp is None:
        print("WARNING: Running without spaCy en_core_web_lg - analysis will be less accurate", file=sys.stderr)
        print("Install with: python -m spacy download en_core_web_lg", file=sys.stderr)
    
    print("Models loaded!", file=sys.stderr)
    
    all_results = []
    
    for filename in args.files:
        print(f"Analyzing {filename}...", file=sys.stderr)
        text = read_file(filename)
        if text is None:
            continue
            
        paragraphs = split_into_paragraphs(text)
        print(f"Found {len(paragraphs)} paragraphs", file=sys.stderr)
        
        for i, paragraph in enumerate(paragraphs, 1):
            print(f"Processing paragraph {i}/{len(paragraphs)}", file=sys.stderr)
            result = analyze_paragraph_detailed(paragraph, i, sentiment_analyzer, model_name, nlp, args.threshold)
            if result:
                all_results.append(result)
    
    if all_results:
        if args.summary:
            print_summary_table(all_results)
        else:
            print_detailed_results(all_results, args.show_words)
    else:
        print("No paragraphs found for analysis.", file=sys.stderr)

if __name__ == "__main__":
    main()
