import torch
import transformers

from transformers import AutoTokenizer
from transformers import LlamaTokenizer


class EssayDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        df,
        train=True
    ):
        self.config = config
        self.df = df
        self.texts = df["full_text"].values
        self.tokenizer = LlamaTokenizer.from_pretrained("philschmid/llama-2-7b-instruction-generator",
                                                       use_fast=True)
        self.labels = None
        
        if "score" in df.columns and train:
            self.labels = df["score"].values - 1

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=512,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return {"input_ids": inputs['input_ids'], "attention_mask": inputs['attention_mask'], "labels": label}
        return {"input_ids": inputs['input_ids'], "attention_mask": inputs['attention_mask']}