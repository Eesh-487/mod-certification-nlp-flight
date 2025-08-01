"""
Text preprocessing utilities for aircraft modification descriptions
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Union
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """
    Comprehensive text preprocessing for aircraft modification descriptions
    """
    
    def __init__(self, use_spacy=True):
        """
        Initialize preprocessor with optional spaCy support
        
        Args:
            use_spacy (bool): Whether to use spaCy for advanced preprocessing
        """
        self.use_spacy = use_spacy
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add aviation-specific stop words
        aviation_stopwords = {
            'aircraft', 'airplane', 'flight', 'aviation', 'system', 'systems',
            'installation', 'installed', 'modify', 'modification', 'mod',
            'equipment', 'component', 'device', 'unit', 'assembly'
        }
        self.stop_words.update(aviation_stopwords)
        
        # Load spaCy model if available
        if self.use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("spaCy model not found. Falling back to NLTK.")
                self.use_spacy = False
                
        # Aviation-specific patterns
        self.aviation_patterns = {
            'regulation': r'\b(CS|AMC)\s*[\d\-\.]+\b',
            'part_number': r'\b[A-Z]{2,4}[\d\-]{3,10}\b',
            'aircraft_model': r'\b(A\d{3}|B\d{3}|ATR|CRJ|ERJ)\w*\b',
            'measurement': r'\d+\s*(mm|cm|m|ft|in|kg|lb|psi|bar)\b'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep aviation-relevant ones
        text = re.sub(r'[^\w\s\-\.\(\)/]', ' ', text)
        
        # Clean up multiple spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_aviation_terms(self, text: str) -> Dict[str, List[str]]:
        """
        Extract aviation-specific terms using regex patterns
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of extracted terms by category
        """
        extracted = {}
        
        for category, pattern in self.aviation_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted[category] = list(set(matches))  # Remove duplicates
            
        return extracted
    
    def tokenize_and_filter(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize text and filter tokens
        
        Args:
            text (str): Input text
            remove_stopwords (bool): Whether to remove stop words
            
        Returns:
            list: List of filtered tokens
        """
        if self.use_spacy:
            doc = self.nlp(text)
            tokens = [token.lemma_ for token in doc 
                     if not token.is_punct and not token.is_space and len(token.text) > 2]
        else:
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in string.punctuation and len(token) > 2]
        
        if remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        return tokens
    
    def preprocess_for_ml(self, texts: Union[str, List[str]], 
                         extract_features: bool = True) -> Dict:
        """
        Comprehensive preprocessing for machine learning
        
        Args:
            texts (str or list): Input text(s)
            extract_features (bool): Whether to extract aviation-specific features
            
        Returns:
            dict: Preprocessed data and features
        """
        if isinstance(texts, str):
            texts = [texts]
        
        results = {
            'cleaned_texts': [],
            'tokens': [],
            'features': [] if extract_features else None
        }
        
        for text in texts:
            # Clean text
            cleaned = self.clean_text(text)
            results['cleaned_texts'].append(cleaned)
            
            # Tokenize
            tokens = self.tokenize_and_filter(cleaned)
            results['tokens'].append(tokens)
            
            # Extract features
            if extract_features:
                features = self.extract_aviation_terms(text)
                # Add text statistics
                features['word_count'] = len(tokens)
                features['char_count'] = len(cleaned)
                features['avg_word_length'] = np.mean([len(word) for word in tokens]) if tokens else 0
                results['features'].append(features)
        
        return results

class FeatureExtractor:
    """
    Extract numerical features from text for ML models
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """
        Initialize feature extractor
        
        Args:
            max_features (int): Maximum number of TF-IDF features
            ngram_range (tuple): N-gram range for TF-IDF
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf_vectorizer = None
        self.sentence_transformer = None
        
    def fit_tfidf(self, texts: List[str]) -> 'FeatureExtractor':
        """
        Fit TF-IDF vectorizer on texts
        
        Args:
            texts (list): List of texts to fit on
            
        Returns:
            self: For method chaining
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        self.tfidf_vectorizer.fit(texts)
        return self
    
    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using fitted TF-IDF vectorizer
        
        Args:
            texts (list): Texts to transform
            
        Returns:
            numpy.ndarray: TF-IDF features
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")
        
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def load_sentence_transformer(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Load sentence transformer model
        
        Args:
            model_name (str): Name of the sentence transformer model
        """
        try:
            self.sentence_transformer = SentenceTransformer(model_name)
            print(f"Loaded sentence transformer: {model_name}")
        except Exception as e:
            print(f"Failed to load sentence transformer: {e}")
            self.sentence_transformer = None
    
    def get_sentence_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get sentence embeddings using transformer model
        
        Args:
            texts (list): Texts to embed
            
        Returns:
            numpy.ndarray: Sentence embeddings
        """
        if self.sentence_transformer is None:
            raise ValueError("Sentence transformer not loaded. Call load_sentence_transformer first.")
        
        embeddings = self.sentence_transformer.encode(texts, convert_to_numpy=True)
        return embeddings

def create_feature_matrix(df: pd.DataFrame, 
                         text_column: str = 'mod_description',
                         use_embeddings: bool = True) -> Dict:
    """
    Create comprehensive feature matrix for ML models
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of text column
        use_embeddings (bool): Whether to include sentence embeddings
        
    Returns:
        dict: Feature matrices and metadata
    """
    preprocessor = TextPreprocessor()
    extractor = FeatureExtractor()
    
    # Preprocess texts
    texts = df[text_column].fillna('').tolist()
    processed = preprocessor.preprocess_for_ml(texts)
    
    results = {
        'processed_texts': processed['cleaned_texts'],
        'tokens': processed['tokens'],
        'features': {}
    }
    
    # TF-IDF features
    extractor.fit_tfidf(processed['cleaned_texts'])
    tfidf_features = extractor.transform_tfidf(processed['cleaned_texts'])
    results['features']['tfidf'] = tfidf_features
    
    # Sentence embeddings
    if use_embeddings:
        extractor.load_sentence_transformer()
        if extractor.sentence_transformer is not None:
            embeddings = extractor.get_sentence_embeddings(processed['cleaned_texts'])
            results['features']['embeddings'] = embeddings
    
    # Metadata
    results['feature_names'] = {
        'tfidf': extractor.tfidf_vectorizer.get_feature_names_out().tolist()
    }
    results['preprocessor'] = preprocessor
    results['extractor'] = extractor
    
    return results

if __name__ == "__main__":
    # Example usage
    sample_text = """Installation of a new VHF antenna on the dorsal fuselage 
                    affecting structural and avionics systems for improved 
                    communication range in oceanic flights according to CS 25.1309."""
    
    preprocessor = TextPreprocessor()
    result = preprocessor.preprocess_for_ml(sample_text)
    
    print("Original text:", sample_text)
    print("\nCleaned text:", result['cleaned_texts'][0])
    print("\nTokens:", result['tokens'][0])
    print("\nExtracted features:", result['features'][0])
