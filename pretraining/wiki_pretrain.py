import torch
import json
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaForMaskedLM
from transformers import TextDataset, LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import set_seed
from sklearn.model_selection import train_test_split


def main():
    wandb.init(project="klue_mrc_pretrain", name="wiki_pretrain")
    
    set_seed(42)

    # 모델 및 토크나이저 불러오기
    model_name = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name)

    def json_to_datasets(data_path, output_path, train_path, eval_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print("[Extracting Texts from Documents]")
        texts = [data[key]['text'] for key in tqdm(data)]
        print("[Extracting Lines from Texts]")
        lines = [line for text in tqdm(texts) for line in text.split('\n') if line.strip()]

        # lines를 텍스트 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(lines))

        train_lines, eval_lines = train_test_split(lines, test_size=0.2, random_state=42)
        with open(train_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(train_lines))
        with open(eval_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(eval_lines))

        train_dataset = LineByLineTextDataset(tokenizer, file_path=train_path, block_size=384)
        eval_dataset = LineByLineTextDataset(tokenizer, file_path=eval_path, block_size=384)
        
        return train_dataset, eval_dataset

    train_dataset, eval_dataset = json_to_datasets(
        '../data/wikipedia_documents.json', 
        '../data/lines.txt',
        '../data/train_lines.txt',
        '../data/eval_lines.txt'
        )

    # 데이터 콜레이터 생성
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,  # Masked Language Modeling을 위한 데이터셋인 경우 True로 설정
        mlm_probability=0.15  # 마스크된 토큰을 대체할 확률
    )

    # 훈련 인자 설정
    training_args = TrainingArguments(
        output_dir=f"../pretrained_models/{model_name.replace('/', '_').replace('-', '_')}",  # 저장할 디렉토리 경로
        overwrite_output_dir=True,  # 기존 결과 디렉토리를 덮어쓸지 여부
        num_train_epochs=5,  # pretraining epoch 수
        per_device_train_batch_size=16,  # 배치 크기
        per_device_eval_batch_size=16,  # 배치 크기
        save_total_limit=2,  # 저장할 체크포인트의 최대 개수
        prediction_loss_only=True,  # MLM 손실 함수만 사용
        learning_rate=1e-4,  # 학습률
        logging_steps=100,
        save_steps=5000,
        eval_steps=5000,
        save_strategy="steps",
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    # Trainer 객체 생성 및 훈련 실행
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    eval_result = trainer.evaluate()
    perplexity = torch.exp(eval_result["eval_loss"])
    print(f"Perplexity: {perplexity.item()}")

if __name__ == "__main__":
    main()