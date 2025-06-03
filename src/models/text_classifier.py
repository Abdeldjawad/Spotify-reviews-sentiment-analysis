from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from src.features.text_preprocessing import basic_text_preprocessor

def build_pipeline():
    return Pipeline([
        ('vect', CountVectorizer(stop_words="english", preprocessor=basic_text_preprocessor)),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='log_loss', random_state=42))
    ])