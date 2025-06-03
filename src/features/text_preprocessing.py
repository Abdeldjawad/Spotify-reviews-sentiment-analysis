import re

def basic_text_preprocessor(text):
    return re.sub(r'[^a-z ]', '', text.lower())