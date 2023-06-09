import pandas as pd
import os
from datasets import load_from_disk
import pprint
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def transform_df(df, just_for_check=False, save_path=None):
    if just_for_check:
        answers = df.answers
        answer_starts = pd.DataFrame([answer['answer_start'][0] for answer in answers])
        answer_starts.columns =['answer_start']
        answer_text = pd.DataFrame([answer['text'][0] for answer in answers])
        answer_text.columns = ['answer_text']
    
        df_modified = df.drop(['answers', '__index_level_0__'], axis=1)
        df_modified = pd.concat([df_modified, answer_starts, answer_text], axis=1)
        df_modified = df_modified[['id', 'title', 'question', 'answer_start', 'answer_text', 'context', 'document_id']]
    else:
        df_modified = df.drop(['__index_level_0__'], axis=1)
        df_modified = df_modified[['id', 'title', 'question', 'answers', 'context', 'document_id']]
    
    if save_path:
        df_modified.to_csv(save_path, index=False)
    
    return df_modified


def plot_token_length_hist(texts, feature_name='context', data_name='train', bins=15):
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    token_lengths = []
    # 텍스트를 토큰화하여 토큰의 길이를 계산
    for text in tqdm(texts):
        token_length = len(tokenizer.tokenize(text))
        token_lengths.append(token_length)

    # 히스토그램으로 토큰의 길이 분포 시각화
    plt.figure(figsize=(13, 7))
    n, bins, patches = plt.hist(token_lengths, bins=bins)
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature_name.title()} ({data_name.title()}) Token Lengths')
    
    for i, (rate, count) in enumerate(zip(bins, n)):
        plt.text(rate + (bins[i+1] - bins[i]) / 2, count, str(int(count)), color='black', ha='center', va='bottom')
    
    plt.show()
    
    print("Max Token Length:", max(token_lengths))
    print("Average Token Length:", sum(token_lengths)/len(token_lengths))
    print("Min Token Length:", min(token_lengths))
    
    
def plot_answer_position_rate_hist(df, data_name='train', bins=20):
    contexts = df.context
    answers_dicts = df.answers
    
    answer_position_rates = []
    for context, answer_dict in zip(tqdm(contexts), answers_dicts):
        answer_start = answer_dict['answer_start'][0]
        
        answer_position_rates.append(answer_start / len(list(context)))
        
    # 히스토그램으로 정답 위치 분포 시각화
    plt.figure(figsize=(13, 7))
    n, bins, patches = plt.hist(answer_position_rates, bins=bins)
    plt.xlabel('Answer Position Rates')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {data_name.title()} Answer Position Rates')
    
    # 데이터 숫자와 x축 눈금 설정
    for i, (rate, count) in enumerate(zip(bins, n)):
        plt.text(rate + (bins[i+1] - bins[i]) / 2, count, str(int(count)), color='black', ha='center', va='bottom')
    
    plt.show()
    
    print("Max Position Rate: {:.4f}".format(max(answer_position_rates)))
    print("Average Position Rate: {:.4f}".format(sum(answer_position_rates)/len(answer_position_rates)))
    print("Min Position Rate: {:.4f}".format(min(answer_position_rates)))
    
    
def plot_answer_inclusion_hist(df, data_name='train'):
    df_modified = transform_df(df)
    contexts = df_modified.context
    answers_texts = df_modified.answer_text
    
    answer_inclusion_rate = []
    for context, answer_text in zip(tqdm(contexts), answers_texts):
        answer_inclusion_rate.append(context.count(answer_text))
    
    # 히스토그램으로 정답 위치 분포 시각화
    plt.figure(figsize=(13, 7))
    plt.hist(answer_inclusion_rate, bins=np.arange(min(answer_inclusion_rate), max(answer_inclusion_rate) + 2) - 0.5)
    plt.xlabel('Answer Inclusion Rates')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {data_name.title()} Answer Inclusion Rates')
    
    plt.xticks(np.arange(min(answer_inclusion_rate), max(answer_inclusion_rate) + 1))
    for i, v in zip(range(max(answer_inclusion_rate)+1), np.bincount(answer_inclusion_rate)):
        plt.text(i, v, str(v), color='black', ha='center', va='bottom')
    
    plt.show()
    
    print("Max Inclusion Rate: {:.4f}".format(max(answer_inclusion_rate)))
    print("Average Inclusion Rate: {:.4f}".format(sum(answer_inclusion_rate)/len(answer_inclusion_rate)))
    print("Min Inclusion Rate: {:.4f}".format(min(answer_inclusion_rate)))