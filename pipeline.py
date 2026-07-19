from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from extract_embeddings import *
from data import *
from split_data import *
from chromadb_collection import *



class ChromaPipeline:
    def __init__(self, model_name, data_path, split_fn, split_args, extract_function, collection_name, description):
        self.model_name = model_name
        self.split_fn = split_fn
        self.split_args = split_args
        self.extract_function = extract_function
        self.collection_name = collection_name
        self.description = description

        self.chroma_client = ChromaDBClient()
        
        self.data_files, self.reference_files = get_data_from_files()

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name, dtype=torch.float32, device_map='auto')

        self.data_files_splitted = split_articles(self.data_files, self.reference_files, lambda x: split_fn(x, **split_args))

        embeddings, metadata = extract_embeddings(
            self.data_files_splitted,
            self.extract_function,
            tokenizer,
            model
        )

        self.chroma_client.create_collection(self.collection_name, self.description)
        ids = [f"doc_{i}" for i in range(len(embeddings))]
        self.chroma_client.add_documents(self.collection_name, ids, embeddings, metadata)   

        print("Collection created and documents added successfully!")



ChromaPipeline(
    model_name="ibm-granite/granite-embedding-311m-multilingual-r2",
    data_path="./data",
    split_fn=split_by_sentences,
    split_args={"max_chars": 200},
    extract_function=get_mean_embedding,
    collection_name="my_collection",
    description="Collection of document chunks with sentence splitting and sentence-transformers embeddings"
)