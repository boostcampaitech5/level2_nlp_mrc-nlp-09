import pandas as pd
from tqdm import tqdm
import random

def k_multi_passages(df, k=3):
    multi_df = df.copy()
    len_df = len(df)
    
    corpus_path = '../data/wikipedia_documents.json'
    corpus_df = pd.read_json(corpus_path).transpose()
    len_corpus_df = len(corpus_df)
    
    for idx, md in tqdm(multi_df.iterrows(), total=len_df):
        random_document_ids = [md.document_id]
        for _ in range(k - 1):
            random_document_id = random.randint(0, len_corpus_df - 1)
            while random_document_id == md.document_id:
                random_document_id = random.randint(0, len_corpus_df - 1)
            random_document_ids.append(random_document_id)
        
        random.shuffle(random_document_ids)
        multi_df.at[idx, 'context'] = " ".join(corpus_df.iloc[random_document_ids].text.tolist())
            
    return multi_df