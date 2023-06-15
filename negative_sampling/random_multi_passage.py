import pandas as pd
from tqdm import tqdm
import random

def k_multi_passages(df, k=3, sep=" "):
    random.seed(42)
    multi_df = df.copy()
    len_df = len(df)
    
    corpus_path = '../data/wikipedia_documents.json'
    corpus_df = pd.read_json(corpus_path).transpose()
    unique_passages = corpus_df['text'].unique()
    selected_documents = corpus_df[corpus_df['text'].isin(unique_passages)]
    clean_corpus_df = selected_documents.copy()
    len_corpus_df = len(clean_corpus_df)
    
    for idx, md in tqdm(multi_df.iterrows(), total=len_df):
        random_document_ids = [md.document_id]
        for _ in range(k - 1):
            random_document_id = random.randint(0, len_corpus_df - 1)
            while random_document_id == md.document_id:
                random_document_id = random.randint(0, len_corpus_df - 1)
            random_document_ids.append(random_document_id)
        
        random.shuffle(random_document_ids)
        multi_df.at[idx, 'context'] = sep.join(clean_corpus_df.iloc[random_document_ids].text.tolist())
        
        answers = eval(md.answers)
        original_document_id_index = random_document_ids.index(md.document_id)
        new_answers = {
            "answer_start": [answers["answer_start"][0] + \
                len(sep.join(clean_corpus_df.iloc[random_document_ids[:original_document_id_index]].text.tolist())) + \
                    min(original_document_id_index, 1)],
            "text": answers["text"]
            }
        multi_df.at[idx, 'answers'] = str(new_answers)
            
    return multi_df

if __name__ == "__main__":
    df = pd.read_csv('../data/eval.csv')
    check_how_many = 100
    print("[ORIGINAL]")
    for idx, d in df.iterrows():
        # print(d.answers, d.context, sep='\n')
        print(f"Context length: {len(d.context)}")
        d_answer_start = eval(d.answers)['answer_start'][0]
        d_answer_end = d_answer_start + len(eval(d.answers)['text'][0])
        print(f"Original answer_start: {eval(d.answers)['answer_start'][0]}")
        print(f"Text in answer_start: {d.context[d_answer_start:d_answer_end]}", end='\n\n')
        if idx == check_how_many:
            break
    
    k = 3
    multi_df = k_multi_passages(df, k=k)
    print("[MULTI_PASSAGE]")
    for idx, md in multi_df.iterrows():
        # print(md.answers, md.context, sep='\n')
        print(f"Context length: {len(md.context)}")
        md_answer_start = eval(md.answers)['answer_start'][0]
        md_answer_end = md_answer_start + len(eval(md.answers)['text'][0])
        print(f"Multi-passage answer_start: {eval(md.answers)['answer_start'][0]}")
        print(f"Text in answer_start: {md.context[md_answer_start:md_answer_end]}", end='\n\n')
        if idx == check_how_many:
            break
        
    multi_df.to_csv(f'../data/eval_{k}_multi_passages.csv', index=False)
    