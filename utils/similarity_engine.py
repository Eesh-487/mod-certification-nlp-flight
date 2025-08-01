"""
Semantic similarity engine for finding similar aircraft modifications
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Try to import FAISS, make it optional
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: FAISS not available. Similarity search will use sklearn cosine_similarity instead.")
    FAISS_AVAILABLE = False

class SimilarityEngine:
    """
    Engine for finding semantically similar aircraft modifications
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize similarity engine
        
        Args:
            model_name (str): Name of sentence transformer model
        """
        self.model_name = model_name
        self.sentence_transformer = None
        self.embeddings = None
        self.faiss_index = None
        self.mod_data = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        self._load_sentence_transformer()
    
    def _load_sentence_transformer(self):
        """Load sentence transformer model"""
        try:
            self.sentence_transformer = SentenceTransformer(self.model_name)
            print(f"Loaded sentence transformer: {self.model_name}")
        except Exception as e:
            print(f"Error loading sentence transformer: {e}")
            self.sentence_transformer = None
    
    def load_modifications(self, df: pd.DataFrame, 
                          text_column: str = 'mod_description',
                          id_column: str = 'mod_id'):
        """
        Load modification data and create embeddings
        
        Args:
            df (pd.DataFrame): DataFrame with modification data
            text_column (str): Column name containing descriptions
            id_column (str): Column name containing modification IDs
        """
        self.mod_data = df.copy()
        descriptions = df[text_column].fillna('').tolist()
        
        print(f"Loading {len(descriptions)} modifications...")
        
        # Create sentence embeddings
        if self.sentence_transformer:
            self.embeddings = self.sentence_transformer.encode(
                descriptions, 
                convert_to_numpy=True,
                show_progress_bar=True
            )
            self._build_faiss_index()
        
        # Create TF-IDF embeddings as fallback
        self._build_tfidf_index(descriptions)
        
        print("Similarity engine ready!")
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        if self.embeddings is None or not FAISS_AVAILABLE:
            print("FAISS not available, using sklearn cosine similarity instead")
            return
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings.astype('float32'))
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        print(f"FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def _build_tfidf_index(self, descriptions: List[str]):
        """Build TF-IDF index as fallback"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions)
        print("TF-IDF index built")
    
    def find_similar_sbert(self, query: str, top_k: int = 10,
                          min_similarity: float = 0.3) -> List[Dict]:
        """
        Find similar modifications using sentence transformers
        
        Args:
            query (str): Query modification description
            top_k (int): Number of similar items to return
            min_similarity (float): Minimum similarity threshold
            
        Returns:
            list: List of similar modifications with metadata
        """
        if self.sentence_transformer is None or self.embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.sentence_transformer.encode([query])
        
        if FAISS_AVAILABLE and self.faiss_index is not None:
            # Use FAISS for fast search
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            similarities, indices = self.faiss_index.search(query_embedding, top_k + 1)
            similarities = similarities[0]
            indices = indices[0]
        else:
            # Use sklearn cosine similarity as fallback
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            indices = np.argsort(similarities)[::-1][:top_k + 1]
            similarities = similarities[indices]
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities, indices)):
            if similarity < min_similarity:
                continue
                
            # Skip self-match if query is in dataset
            mod_data = self.mod_data.iloc[idx]
            if mod_data['mod_description'] == query:
                continue
            
            results.append({
                'mod_id': mod_data['mod_id'],
                'mod_description': mod_data['mod_description'],
                'mod_type': mod_data.get('mod_type', 'Unknown'),
                'regulations': mod_data.get('regulations', ''),
                'loi': mod_data.get('loi', 'Unknown'),
                'similarity_score': float(similarity),
                'rank': len(results) + 1
            })
        
        return results
    
    def find_similar_tfidf(self, query: str, top_k: int = 10,
                          min_similarity: float = 0.1) -> List[Dict]:
        """
        Find similar modifications using TF-IDF
        
        Args:
            query (str): Query modification description
            top_k (int): Number of similar items to return
            min_similarity (float): Minimum similarity threshold
            
        Returns:
            list: List of similar modifications with metadata
        """
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k + 1]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            
            if similarity < min_similarity:
                continue
            
            # Skip self-match
            mod_data = self.mod_data.iloc[idx]
            if mod_data['mod_description'] == query:
                continue
            
            results.append({
                'mod_id': mod_data['mod_id'],
                'mod_description': mod_data['mod_description'],
                'mod_type': mod_data.get('mod_type', 'Unknown'),
                'regulations': mod_data.get('regulations', ''),
                'loi': mod_data.get('loi', 'Unknown'),
                'similarity_score': float(similarity),
                'rank': len(results) + 1
            })
        
        return results
    
    def find_similar(self, query: str, method: str = 'sbert',
                    top_k: int = 10, min_similarity: float = 0.3) -> List[Dict]:
        """
        Find similar modifications using specified method
        
        Args:
            query (str): Query modification description
            method (str): Similarity method ('sbert' or 'tfidf')
            top_k (int): Number of similar items to return
            min_similarity (float): Minimum similarity threshold
            
        Returns:
            list: List of similar modifications
        """
        if method == 'sbert':
            return self.find_similar_sbert(query, top_k, min_similarity)
        elif method == 'tfidf':
            return self.find_similar_tfidf(query, top_k, min_similarity)
        else:
            raise ValueError("Method must be 'sbert' or 'tfidf'")
    
    def find_similar_by_category(self, query: str, category: str,
                               top_k: int = 5) -> List[Dict]:
        """
        Find similar modifications within a specific category
        
        Args:
            query (str): Query modification description
            category (str): Modification category to search within
            top_k (int): Number of results to return
            
        Returns:
            list: List of similar modifications in category
        """
        # Filter data by category
        category_data = self.mod_data[
            self.mod_data['mod_type'] == category
        ].copy()
        
        if category_data.empty:
            return []
        
        # Create temporary engine for category
        temp_engine = SimilarityEngine(self.model_name)
        temp_engine.sentence_transformer = self.sentence_transformer
        temp_engine.load_modifications(category_data)
        
        return temp_engine.find_similar(query, top_k=top_k)
    
    def get_modification_clusters(self, n_clusters: int = 5) -> Dict:
        """
        Cluster modifications for analysis
        
        Args:
            n_clusters (int): Number of clusters
            
        Returns:
            dict: Clustering results
        """
        if self.embeddings is None:
            return {}
        
        from sklearn.cluster import KMeans
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.embeddings)
        
        # Organize results
        clusters = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_mods = self.mod_data[cluster_mask]
            
            clusters[f'cluster_{i}'] = {
                'size': int(np.sum(cluster_mask)),
                'modifications': cluster_mods['mod_id'].tolist(),
                'common_types': cluster_mods['mod_type'].value_counts().to_dict(),
                'sample_descriptions': cluster_mods['mod_description'].head(3).tolist()
            }
        
        return clusters
    
    def explain_similarity(self, query: str, similar_mod: Dict) -> Dict:
        """
        Explain why two modifications are similar
        
        Args:
            query (str): Query modification description
            similar_mod (dict): Similar modification data
            
        Returns:
            dict: Explanation of similarity
        """
        explanation = {
            'similarity_score': similar_mod['similarity_score'],
            'common_keywords': [],
            'shared_patterns': []
        }
        
        # Extract common keywords
        query_words = set(query.lower().split())
        mod_words = set(similar_mod['mod_description'].lower().split())
        common_words = query_words.intersection(mod_words)
        
        # Filter meaningful words (length > 3)
        explanation['common_keywords'] = [
            word for word in common_words if len(word) > 3
        ]
        
        # Check for shared patterns
        patterns = [
            'installation', 'modification', 'upgrade', 'replacement',
            'antenna', 'system', 'avionics', 'structural'
        ]
        
        for pattern in patterns:
            if pattern in query.lower() and pattern in similar_mod['mod_description'].lower():
                explanation['shared_patterns'].append(pattern)
        
        return explanation
    
    def save_index(self, path: str):
        """
        Save similarity index to file
        
        Args:
            path (str): Path to save index
        """
        index_data = {
            'embeddings': self.embeddings,
            'mod_data': self.mod_data,
            'model_name': self.model_name,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix
        }
        
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """
        Load similarity index from file
        
        Args:
            path (str): Path to load index from
        """
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.embeddings = index_data['embeddings']
        self.mod_data = index_data['mod_data']
        self.model_name = index_data['model_name']
        self.tfidf_vectorizer = index_data['tfidf_vectorizer']
        self.tfidf_matrix = index_data['tfidf_matrix']
        
        # Rebuild FAISS index
        if self.embeddings is not None:
            self._build_faiss_index()
        
        # Reload sentence transformer
        self._load_sentence_transformer()
        
        print(f"Index loaded from {path}")

class AdvancedSimilarityEngine(SimilarityEngine):
    """
    Advanced similarity engine with additional features
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__(model_name)
        self.regulation_weights = {}
        self.type_weights = {}
    
    def set_regulation_weights(self, weights: Dict[str, float]):
        """
        Set weights for different regulation types
        
        Args:
            weights (dict): Regulation type weights
        """
        self.regulation_weights = weights
    
    def find_similar_weighted(self, query: str, query_regulations: List[str],
                            top_k: int = 10) -> List[Dict]:
        """
        Find similar modifications with regulation-based weighting
        
        Args:
            query (str): Query description
            query_regulations (list): Query regulations
            top_k (int): Number of results
            
        Returns:
            list: Weighted similarity results
        """
        # Get base similarity results
        base_results = self.find_similar(query, top_k=top_k * 2)
        
        # Apply regulation-based weighting
        weighted_results = []
        for result in base_results:
            mod_regulations = result['regulations'].split(',')
            mod_regulations = [reg.strip() for reg in mod_regulations]
            
            # Calculate regulation overlap bonus
            overlap = set(query_regulations).intersection(set(mod_regulations))
            reg_bonus = len(overlap) * 0.1  # 10% bonus per overlapping regulation
            
            # Apply bonus
            weighted_score = result['similarity_score'] + reg_bonus
            result['weighted_score'] = min(weighted_score, 1.0)  # Cap at 1.0
            result['regulation_overlap'] = list(overlap)
            
            weighted_results.append(result)
        
        # Sort by weighted score and return top k
        weighted_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        return weighted_results[:top_k]

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    
    # Load sample data
    df = pd.read_csv('../data/mods_dataset.csv')
    
    # Initialize engine
    engine = SimilarityEngine()
    engine.load_modifications(df)
    
    # Test similarity search
    query = "Installation of VHF communication antenna on fuselage"
    results = engine.find_similar(query, top_k=5)
    
    print(f"Query: {query}")
    print("\nSimilar modifications:")
    for result in results:
        print(f"- {result['mod_id']}: {result['similarity_score']:.3f}")
        print(f"  {result['mod_description'][:100]}...")
        print()
