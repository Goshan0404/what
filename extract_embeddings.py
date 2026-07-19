from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from tqdm import tqdm
import torch.nn.functional as F

def cls(inputs, tokenizer, model, tokenized=False):
    inputs_tokenized = inputs
    if not tokenized:
        inputs_tokenized = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').to(model.device)
    with torch.no_grad():
        cls = model(inputs_tokenized['input_ids']).last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        cls = F.normalize(cls, p=2, dim=1)
        return cls

# def get_mean_embedding(inputs, tokenizer, model, tokenized=False):
#     inputs_tokenized = tokenizer(inputs, padding=True, 
#                                   truncation=True, return_tensors='pt').to(model.device)
#     with torch.no_grad():
#         output = model(**inputs_tokenized)
    
#     mask = inputs_tokenized["attention_mask"].unsqueeze(-1)
#     mean_emb = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
    
#     mean_emb = F.normalize(mean_emb, p=2, dim=1)
#     return mean_emb.squeeze().cpu().numpy()

def get_mean_embedding(inputs, tokenizer, model, tokenized=False):
    encoded = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=512,       
        return_tensors="pt"
    )

    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

        mask = encoded["attention_mask"].unsqueeze(-1)
        mean_emb = (outputs.last_hidden_state * mask).sum(1)
        mean_emb = mean_emb / mask.sum(1)
        mean_emb = F.normalize(mean_emb, p=2, dim=1)

    result = mean_emb.squeeze().cpu().numpy()

    del outputs
    del encoded
    del mask
    del mean_emb

    return result

def extract_embeddings(articles_splitted, extraction_function, tokenizer, model):

    embeddings = []
    metadata = []
    for split, id in tqdm(articles_splitted):
        cls = extraction_function(split, tokenizer, model, tokenized=False)
        embeddings.append(cls)
        metadata.append({"id": str(id), "text": split})
    return embeddings, metadata
