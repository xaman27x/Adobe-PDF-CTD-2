import re

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def normalize_text(text):
    return text.lower().strip()

def is_heading(text):
    return len(text.split()) < 10 and text.istitle()
