#!/usr/bin/env python3
"""
Analysis of Pride and Prejudice text dataset
Focusing on tokens, vocabulary, and text characteristics
"""

import re
from collections import Counter

def simple_tokenize(text):
    """Simple word tokenization using regex"""
    # Remove punctuation and split on whitespace
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words

def simple_sentence_tokenize(text):
    """Simple sentence tokenization using regex"""
    # Split on sentence endings
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def analyze_text_dataset(file_path):
    """Analyze the Pride and Prejudice text from a dataset perspective"""
    
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    print("=" * 80)
    print("PRIDE AND PREJUDICE TEXT DATASET ANALYSIS")
    print("=" * 80)
    
    # Basic text statistics
    print(f"\n1. BASIC TEXT STATISTICS:")
    print(f"   Total characters: {len(text):,}")
    print(f"   Total lines: {text.count(chr(10)):,}")
    print(f"   Total words (rough estimate): {len(text.split()):,}")
    
    # Remove Project Gutenberg header/footer
    # Find the actual novel content
    start_marker = "CHAPTER I."
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        novel_text = text[start_idx:end_idx].strip()
    else:
        novel_text = text
    
    print(f"   Novel content characters: {len(novel_text):,}")
    
    # Tokenization analysis
    print(f"\n2. TOKENIZATION ANALYSIS:")
    
    # Word tokens
    word_tokens = simple_tokenize(novel_text)
    
    print(f"   Total word tokens: {len(word_tokens):,}")
    print(f"   Unique word tokens: {len(set(word_tokens)):,}")
    
    # Sentence tokens
    sentences = simple_sentence_tokenize(novel_text)
    print(f"   Total sentence tokens: {len(sentences):,}")
    
    # Average sentence length
    avg_sentence_length = len(word_tokens) / len(sentences) if sentences else 0
    print(f"   Average sentence length: {avg_sentence_length:.1f} words")
    
    # Vocabulary analysis
    print(f"\n3. VOCABULARY ANALYSIS:")
    
    # Word frequency
    word_freq = Counter(word_tokens)
    
    print(f"   Most common words:")
    for word, count in word_freq.most_common(15):
        print(f"     '{word}': {count:,} times")
    
    # Vocabulary richness
    vocab_richness = len(set(word_tokens)) / len(word_tokens)
    print(f"   Vocabulary richness (type-token ratio): {vocab_richness:.4f}")
    
    # Stop words analysis (basic English stop words)
    basic_stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
    }
    
    content_words = [word for word in word_tokens if word not in basic_stop_words]
    
    print(f"   Content words (excluding stop words): {len(content_words):,}")
    print(f"   Unique content words: {len(set(content_words)):,}")
    
    # Character analysis
    print(f"\n4. CHARACTER ANALYSIS:")
    
    # Character names mentioned
    character_patterns = [
        r'\bMr\.\s+[A-Z][a-z]+\b',
        r'\bMrs\.\s+[A-Z][a-z]+\b',
        r'\bMiss\s+[A-Z][a-z]+\b',
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # Full names
    ]
    
    characters = []
    for pattern in character_patterns:
        matches = re.findall(pattern, novel_text)
        characters.extend(matches)
    
    character_freq = Counter(characters)
    print(f"   Character mentions (top 10):")
    for char, count in character_freq.most_common(10):
        print(f"     '{char}': {count} times")
    
    # Dialogue analysis
    print(f"\n5. DIALOGUE ANALYSIS:")
    
    # Count dialogue lines (text in quotes)
    dialogue_pattern = r'"([^"]*)"'
    dialogues = re.findall(dialogue_pattern, novel_text)
    
    print(f"   Total dialogue lines: {len(dialogues):,}")
    print(f"   Dialogue word count: {sum(len(d.split()) for d in dialogues):,}")
    
    # Average dialogue length
    if dialogues:
        avg_dialogue_length = sum(len(d.split()) for d in dialogues) / len(dialogues)
        print(f"   Average dialogue length: {avg_dialogue_length:.1f} words")
    
    # Chapter analysis
    print(f"\n6. CHAPTER ANALYSIS:")
    
    chapters = re.findall(r'CHAPTER [IVX]+\.', novel_text)
    print(f"   Total chapters: {len(chapters)}")
    
    # Word length distribution
    print(f"\n7. WORD LENGTH DISTRIBUTION:")
    word_lengths = [len(word) for word in word_tokens]
    length_freq = Counter(word_lengths)
    
    for length in sorted(length_freq.keys()):
        if length <= 15:  # Show up to 15-letter words
            print(f"   {length}-letter words: {length_freq[length]:,}")
    
    # Longest words
    longest_words = sorted(set(word_tokens), key=len, reverse=True)[:10]
    print(f"   Longest words: {', '.join(longest_words)}")
    
    # Dataset characteristics summary
    print(f"\n8. DATASET CHARACTERISTICS SUMMARY:")
    print(f"   • Text type: Classic English novel (1813)")
    print(f"   • Author: Jane Austen")
    print(f"   • Genre: Romantic comedy of manners")
    print(f"   • Time period: Early 19th century English society")
    print(f"   • Writing style: Formal, dialogue-rich, character-driven")
    print(f"   • Language: Early Modern English with some archaic forms")
    print(f"   • Primary themes: Marriage, social class, pride, prejudice")
    
    # Tokenization insights
    print(f"\n9. TOKENIZATION INSIGHTS:")
    print(f"   • Average words per sentence: {avg_sentence_length:.1f}")
    print(f"   • Vocabulary diversity: {vocab_richness:.1%}")
    print(f"   • Content word ratio: {len(content_words)/len(word_tokens):.1%}")
    print(f"   • Dialogue ratio: {sum(len(d.split()) for d in dialogues)/len(word_tokens):.1%}")
    
    # Additional dataset insights
    print(f"\n10. ADDITIONAL DATASET INSIGHTS:")
    print(f"    • Token density: {len(word_tokens)/len(novel_text):.2f} tokens per character")
    print(f"    • Unique token ratio: {len(set(word_tokens))/len(word_tokens):.1%}")
    print(f"    • Most frequent word: '{word_freq.most_common(1)[0][0]}' ({word_freq.most_common(1)[0][1]:,} occurrences)")
    print(f"    • Words appearing only once: {sum(1 for count in word_freq.values() if count == 1):,}")
    print(f"    • Words appearing 10+ times: {sum(1 for count in word_freq.values() if count >= 10):,}")
    
    return {
        'total_chars': len(text),
        'total_words': len(word_tokens),
        'unique_words': len(set(word_tokens)),
        'sentences': len(sentences),
        'chapters': len(chapters),
        'dialogues': len(dialogues),
        'vocab_richness': vocab_richness,
        'avg_sentence_length': avg_sentence_length
    }

if __name__ == "__main__":
    # Analyze the Pride and Prejudice dataset
    stats = analyze_text_dataset("datasets/pride_prejudice.txt") 