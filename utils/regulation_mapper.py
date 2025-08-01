"""
Regulation mapping utilities for aircraft modification certification
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import warnings
warnings.filterwarnings('ignore')

class RegulationMapper:
    """
    Maps aircraft modifications to relevant EASA CS-25/AMC regulations
    """
    
    def __init__(self, regulations_path: str = None):
        """
        Initialize regulation mapper
        
        Args:
            regulations_path (str): Path to regulations database CSV
        """
        self.regulations_df = None
        self.regulation_embeddings = None
        self.tfidf_vectorizer = None
        self.mlb = MultiLabelBinarizer()
        self.classifier = None
        self.regulation_keywords = {}
        
        if regulations_path:
            self.load_regulations(regulations_path)
    
    def load_regulations(self, path: str):
        """
        Load regulations database
        
        Args:
            path (str): Path to regulations CSV file
        """
        try:
            self.regulations_df = pd.read_csv(path)
            print(f"Loaded {len(self.regulations_df)} regulations")
            self._build_regulation_index()
        except Exception as e:
            print(f"Error loading regulations: {e}")
    
    def _build_regulation_index(self):
        """
        Build searchable index of regulations
        """
        if self.regulations_df is None:
            return
        
        # Combine title and description for better matching
        self.regulations_df['combined_text'] = (
            self.regulations_df['title'].fillna('') + ' ' + 
            self.regulations_df['description'].fillna('')
        )
        
        # Create TF-IDF index
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        self.regulation_embeddings = self.tfidf_vectorizer.fit_transform(
            self.regulations_df['combined_text']
        )
        
        # Build keyword mappings
        self._extract_regulation_keywords()
    
    def _extract_regulation_keywords(self):
        """
        Extract keywords for each regulation category
        """
        categories = self.regulations_df['category'].unique()
        
        for category in categories:
            cat_regs = self.regulations_df[
                self.regulations_df['category'] == category
            ]
            
            # Combine all text for this category
            combined_text = ' '.join(cat_regs['combined_text'].tolist())
            
            # Extract key terms (simplified approach)
            words = combined_text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top keywords
            top_keywords = sorted(word_freq.items(), 
                                key=lambda x: x[1], reverse=True)[:20]
            self.regulation_keywords[category] = [kw[0] for kw in top_keywords]
    
    def map_by_similarity(self, mod_description: str, 
                         top_k: int = 5) -> List[Dict]:
        """
        Map modification to regulations using text similarity
        
        Args:
            mod_description (str): Modification description
            top_k (int): Number of top regulations to return
            
        Returns:
            list: List of matched regulations with scores
        """
        if self.tfidf_vectorizer is None or self.regulation_embeddings is None:
            return []
        
        # Transform input text
        mod_vector = self.tfidf_vectorizer.transform([mod_description])
        
        # Calculate similarities
        similarities = cosine_similarity(mod_vector, self.regulation_embeddings)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            reg_info = self.regulations_df.iloc[idx]
            results.append({
                'regulation_id': reg_info['regulation_id'],
                'title': reg_info['title'],
                'category': reg_info['category'],
                'similarity_score': similarities[idx],
                'description': reg_info['description']
            })
        
        return results
    
    def map_by_keywords(self, mod_description: str) -> Dict[str, float]:
        """
        Map modification to regulation categories using keyword matching
        
        Args:
            mod_description (str): Modification description
            
        Returns:
            dict: Category scores based on keyword matches
        """
        mod_words = mod_description.lower().split()
        category_scores = {}
        
        for category, keywords in self.regulation_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in mod_words:
                    score += 1
            
            # Normalize by number of keywords
            category_scores[category] = score / len(keywords) if keywords else 0
        
        return category_scores
    
    def map_by_patterns(self, mod_description: str) -> List[str]:
        """
        Extract explicitly mentioned regulations using regex patterns
        
        Args:
            mod_description (str): Modification description
            
        Returns:
            list: List of regulation IDs found in text
        """
        # Patterns for CS and AMC regulations
        patterns = [
            r'\bCS\s*25\.[\d]+\b',          # CS 25.xxx
            r'\bAMC\s*25[-\s]*[\d]+\b',     # AMC 25-xxx or AMC 25 xxx
            r'\bAMC\s*20[-\s]*[\d]+\b',     # AMC 20-xxx
            r'\bCS\s*36\.[\d]+\b'           # CS 36.xxx (noise)
        ]
        
        found_regulations = []
        for pattern in patterns:
            matches = re.findall(pattern, mod_description, re.IGNORECASE)
            found_regulations.extend(matches)
        
        # Clean up matches
        cleaned = []
        for reg in found_regulations:
            reg = re.sub(r'\s+', ' ', reg.strip())
            cleaned.append(reg.upper())
        
        return list(set(cleaned))  # Remove duplicates
    
    def predict_regulations(self, mod_description: str, 
                          confidence_threshold: float = 0.1) -> Dict:
        """
        Comprehensive regulation prediction combining multiple methods
        
        Args:
            mod_description (str): Modification description
            confidence_threshold (float): Minimum confidence for predictions
            
        Returns:
            dict: Comprehensive prediction results
        """
        results = {
            'explicit_regulations': [],
            'similar_regulations': [],
            'category_scores': {},
            'recommended_regulations': []
        }
        
        # 1. Extract explicitly mentioned regulations
        results['explicit_regulations'] = self.map_by_patterns(mod_description)
        
        # 2. Find similar regulations
        results['similar_regulations'] = self.map_by_similarity(mod_description)
        
        # 3. Category-based scoring
        results['category_scores'] = self.map_by_keywords(mod_description)
        
        # 4. Combine results for final recommendations
        recommendations = set()
        
        # Add explicit regulations
        recommendations.update(results['explicit_regulations'])
        
        # Add high-similarity regulations
        for reg in results['similar_regulations']:
            if reg['similarity_score'] > confidence_threshold:
                recommendations.add(reg['regulation_id'])
        
        # Add regulations from high-scoring categories
        for category, score in results['category_scores'].items():
            if score > confidence_threshold:
                cat_regs = self.regulations_df[
                    self.regulations_df['category'] == category
                ]['regulation_id'].tolist()
                recommendations.update(cat_regs[:3])  # Top 3 from category
        
        results['recommended_regulations'] = list(recommendations)
        
        return results
    
    def train_ml_classifier(self, mod_descriptions: List[str], 
                           regulation_lists: List[List[str]]):
        """
        Train ML classifier for regulation prediction
        
        Args:
            mod_descriptions (list): List of modification descriptions
            regulation_lists (list): List of regulation lists for each mod
        """
        # Prepare features
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            X = self.tfidf_vectorizer.fit_transform(mod_descriptions)
        else:
            X = self.tfidf_vectorizer.transform(mod_descriptions)
        
        # Prepare labels
        y = self.mlb.fit_transform(regulation_lists)
        
        # Train classifier
        self.classifier = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
        self.classifier.fit(X, y)
        
        print(f"Trained classifier on {len(mod_descriptions)} samples")
        print(f"Number of unique regulations: {len(self.mlb.classes_)}")
    
    def predict_with_ml(self, mod_description: str, 
                       threshold: float = 0.3) -> List[str]:
        """
        Predict regulations using trained ML classifier
        
        Args:
            mod_description (str): Modification description
            threshold (float): Confidence threshold
            
        Returns:
            list: Predicted regulation IDs
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train_ml_classifier first.")
        
        # Transform input
        X = self.tfidf_vectorizer.transform([mod_description])
        
        # Get predictions with probabilities
        probabilities = self.classifier.predict_proba(X)[0]
        
        # Apply threshold
        predicted_regulations = []
        for i, prob_dist in enumerate(probabilities):
            if len(prob_dist) > 1 and prob_dist[1] > threshold:  # Binary classifier
                predicted_regulations.append(self.mlb.classes_[i])
        
        return predicted_regulations
    
    def save_model(self, path: str):
        """
        Save trained models to file
        
        Args:
            path (str): Path to save the model
        """
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'mlb': self.mlb,
            'classifier': self.classifier,
            'regulation_keywords': self.regulation_keywords
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load trained models from file
        
        Args:
            path (str): Path to load the model from
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.mlb = model_data['mlb']
        self.classifier = model_data['classifier']
        self.regulation_keywords = model_data['regulation_keywords']
        
        print(f"Model loaded from {path}")

def create_regulation_features(mod_descriptions: List[str], 
                              regulations_path: str) -> Dict:
    """
    Create regulation-based features for ML models
    
    Args:
        mod_descriptions (list): List of modification descriptions
        regulations_path (str): Path to regulations database
        
    Returns:
        dict: Feature matrices and metadata
    """
    mapper = RegulationMapper(regulations_path)
    
    features = {
        'explicit_reg_counts': [],
        'category_scores': [],
        'similarity_features': []
    }
    
    for desc in mod_descriptions:
        # Count explicit regulations
        explicit_regs = mapper.map_by_patterns(desc)
        features['explicit_reg_counts'].append(len(explicit_regs))
        
        # Category scores
        cat_scores = mapper.map_by_keywords(desc)
        features['category_scores'].append(list(cat_scores.values()))
        
        # Similarity features (top 5 similarity scores)
        similar_regs = mapper.map_by_similarity(desc, top_k=5)
        sim_scores = [reg['similarity_score'] for reg in similar_regs]
        features['similarity_features'].append(sim_scores)
    
    return {
        'features': features,
        'mapper': mapper,
        'category_names': list(mapper.regulation_keywords.keys())
    }

if __name__ == "__main__":
    # Example usage
    mapper = RegulationMapper('../data/regulations_db.csv')
    
    sample_description = """Installation of a new VHF antenna on the dorsal 
                           fuselage affecting structural and avionics systems"""
    
    results = mapper.predict_regulations(sample_description)
    
    print("Regulation Mapping Results:")
    print("Explicit regulations:", results['explicit_regulations'])
    print("Category scores:", results['category_scores'])
    print("Recommended regulations:", results['recommended_regulations'])
