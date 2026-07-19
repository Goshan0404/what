from typing import List
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TextTilingTokenizer


def simple_split(data: str, max_chars: int = 400) -> List[str]:
    return [data[i:i+max_chars] for i in range(0, len(data), max_chars)]

def split_by_sentences(text, max_chars=400):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + sent
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sent
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def split_with_overlap(text, max_chars=400, overlap_chars=200):
    sentences = sent_tokenize(text)
    chunks = []
    
    i = 0
    while i < len(sentences):
        current_chunk = ""
        start = i
        
        # собираем чанк
        while i < len(sentences) and len(current_chunk) + len(sentences[i]) < max_chars:
            current_chunk += " " + sentences[i]
            i += 1
        
        # если не смогли добавить ни одного предложения
        if i == start:
            current_chunk = sentences[i][:max_chars]
            i += 1
        
        chunks.append(current_chunk.strip())
        
        # считаем overlap через индексы
        overlap_len = 0
        j = i - 1
        
        while j >= 0 and overlap_len + len(sentences[j]) < overlap_chars:
            overlap_len += len(sentences[j])
            j -= 1
        
        i = max(j + 1, start + 1)  # гарантируем прогресс
    
    return chunks


def split_by_texttiling(text, w=1, k=1):
    tokenizer = TextTilingTokenizer(w=w, k=k)
    try:
        chunks = tokenizer.tokenize(text)
    except Exception as e:
        print(f"TextTiling error: {e}")
        chunks = [text]
    return chunks
