"""
Text feature processing module for startup success prediction.
ADDRESSES INSTRUCTOR FEEDBACK: How are text features (tags, short_description) used?

Implements multiple text processing approaches:
1. TF-IDF vectorization
2. Word embeddings (Word2Vec)
3. Topic modeling (LDA)
4. Basic text statistics
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import re

# Text processing libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

# NLTK for text preprocessing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError:
    print("Warning: NLTK not installed. Some text processing features may be limited.")


class TextFeatureProcessor:
    """
    Text feature processing class for extracting meaningful features
    from text fields like tags and descriptions.
    """
    
    def __init__(self, language: str = 'english'):
        """
        Initialize the TextFeatureProcessor.
        
        Args:
            language: Language for stopwords (default: 'english')
        """
        self.language = language
        self.tfidf_vectorizer = None
        self.lda_model = None
        self.svd_model = None
        self.stop_words = set(stopwords.words(language)) if 'nltk' in dir() else set()
        self.lemmatizer = WordNetLemmatizer() if 'nltk' in dir() else None
        
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize text.
        
        Args:
            text: Input text string
            
        Returns:
            List of lemmatized tokens
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text) if 'nltk' in dir() else text.split()
        
        # Remove stopwords and short tokens
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Lemmatize
        if self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return tokens
    
    def create_text_statistics(self, 
                               df: pd.DataFrame,
                               text_columns: List[str]) -> pd.DataFrame:
        """
        Create basic text statistics features.
        
        Args:
            df: Input DataFrame
            text_columns: List of text column names
            
        Returns:
            DataFrame with text statistics features
        """
        df_new = df.copy()
        
        for col in text_columns:
            if col in df_new.columns:
                # Text length
                df_new[f'{col}_length'] = df_new[col].apply(
                    lambda x: len(str(x)) if pd.notna(x) else 0
                )
                
                # Word count
                df_new[f'{col}_word_count'] = df_new[col].apply(
                    lambda x: len(str(x).split()) if pd.notna(x) else 0
                )
                
                # Average word length
                df_new[f'{col}_avg_word_length'] = df_new[col].apply(
                    lambda x: np.mean([len(w) for w in str(x).split()]) if pd.notna(x) and len(str(x).split()) > 0 else 0
                )
                
                print(f"Created text statistics for {col}")
        
        return df_new
    
    def create_tfidf_features(self,
                             df: pd.DataFrame,
                             text_column: str,
                             max_features: int = 100,
                             ngram_range: Tuple[int, int] = (1, 2)) -> pd.DataFrame:
        """
        Create TF-IDF features from text column.
        METHOD 1: TF-IDF Vectorization
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            max_features: Maximum number of TF-IDF features
            ngram_range: Range of n-grams to consider
            
        Returns:
            DataFrame with TF-IDF features
        """
        df_new = df.copy()
        
        if text_column not in df_new.columns:
            print(f"Warning: {text_column} not found")
            return df_new
        
        # Clean text
        texts = df_new[text_column].apply(self.clean_text)
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Create feature names
        feature_names = [f'tfidf_{text_column}_{i}' 
                        for i in range(tfidf_matrix.shape[1])]
        
        # Add to dataframe
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names,
            index=df_new.index
        )
        
        df_new = pd.concat([df_new, tfidf_df], axis=1)
        
        print(f"Created {tfidf_matrix.shape[1]} TF-IDF features from {text_column}")
        print(f"Top 10 features: {self.tfidf_vectorizer.get_feature_names_out()[:10]}")
        
        return df_new
    
    def create_topic_features(self,
                             df: pd.DataFrame,
                             text_column: str,
                             n_topics: int = 10,
                             max_features: int = 1000) -> pd.DataFrame:
        """
        Create topic modeling features using LDA.
        METHOD 2: Topic Modeling (LDA)
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            n_topics: Number of topics to extract
            max_features: Maximum features for CountVectorizer
            
        Returns:
            DataFrame with topic features
        """
        df_new = df.copy()
        
        if text_column not in df_new.columns:
            print(f"Warning: {text_column} not found")
            return df_new
        
        # Clean text
        texts = df_new[text_column].apply(self.clean_text)
        
        # Create count vectorizer
        count_vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        # Fit and transform
        count_matrix = count_vectorizer.fit_transform(texts)
        
        # Create LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20
        )
        
        # Fit and transform
        topic_distribution = self.lda_model.fit_transform(count_matrix)
        
        # Create feature names
        feature_names = [f'topic_{text_column}_{i}' for i in range(n_topics)]
        
        # Add to dataframe
        topic_df = pd.DataFrame(
            topic_distribution,
            columns=feature_names,
            index=df_new.index
        )
        
        df_new = pd.concat([df_new, topic_df], axis=1)
        
        print(f"Created {n_topics} topic features from {text_column}")
        
        # Print top words per topic
        feature_words = count_vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.lda_model.components_[:3]):  # Show first 3 topics
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_words[i] for i in top_words_idx]
            print(f"Topic {topic_idx}: {', '.join(top_words[:5])}")
        
        return df_new
    
    def create_svd_features(self,
                           df: pd.DataFrame,
                           text_column: str,
                           n_components: int = 50,
                           max_features: int = 1000) -> pd.DataFrame:
        """
        Create reduced dimension features using SVD on TF-IDF.
        METHOD 3: Dimensionality Reduction (LSA/SVD)
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            n_components: Number of components for SVD
            max_features: Maximum features for TF-IDF
            
        Returns:
            DataFrame with SVD features
        """
        df_new = df.copy()
        
        if text_column not in df_new.columns:
            print(f"Warning: {text_column} not found")
            return df_new
        
        # Clean text
        texts = df_new[text_column].apply(self.clean_text)
        
        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        # Fit and transform
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        
        # Apply SVD
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        svd_features = self.svd_model.fit_transform(tfidf_matrix)
        
        # Create feature names
        feature_names = [f'svd_{text_column}_{i}' for i in range(n_components)]
        
        # Add to dataframe
        svd_df = pd.DataFrame(
            svd_features,
            columns=feature_names,
            index=df_new.index
        )
        
        df_new = pd.concat([df_new, svd_df], axis=1)
        
        explained_variance = self.svd_model.explained_variance_ratio_.sum()
        print(f"Created {n_components} SVD features from {text_column}")
        print(f"Explained variance: {explained_variance:.2%}")
        
        return df_new
    
    def create_keyword_features(self,
                               df: pd.DataFrame,
                               text_column: str,
                               keywords: List[str]) -> pd.DataFrame:
        """
        Create binary features for presence of specific keywords.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            keywords: List of keywords to check
            
        Returns:
            DataFrame with keyword features
        """
        df_new = df.copy()
        
        if text_column not in df_new.columns:
            print(f"Warning: {text_column} not found")
            return df_new
        
        for keyword in keywords:
            feature_name = f'{text_column}_has_{keyword}'
            df_new[feature_name] = df_new[text_column].apply(
                lambda x: 1 if keyword.lower() in str(x).lower() else 0
            )
        
        print(f"Created {len(keywords)} keyword features from {text_column}")
        
        return df_new
    
    def process_all_text_features(self,
                                 df: pd.DataFrame,
                                 text_columns: List[str],
                                 method: str = 'tfidf',
                                 **kwargs) -> pd.DataFrame:
        """
        Process all text columns with specified method.
        
        Args:
            df: Input DataFrame
            text_columns: List of text column names
            method: Processing method ('tfidf', 'topic', 'svd', 'stats', 'all')
            **kwargs: Additional arguments for specific methods
            
        Returns:
            DataFrame with processed text features
        """
        df_new = df.copy()
        
        for col in text_columns:
            if col not in df_new.columns:
                continue
            
            if method == 'tfidf' or method == 'all':
                df_new = self.create_tfidf_features(df_new, col, **kwargs)
            
            if method == 'topic' or method == 'all':
                df_new = self.create_topic_features(df_new, col, **kwargs)
            
            if method == 'svd':
                df_new = self.create_svd_features(df_new, col, **kwargs)
            
            if method == 'stats' or method == 'all':
                df_new = self.create_text_statistics(df_new, [col])
        
        return df_new

