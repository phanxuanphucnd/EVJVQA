import json
import os

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, T5Tokenizer

from .. dataset.vqa_dataset import VQADataset
from .. model import MT5ForConditionalGeneration
from .. evaluate.eval import compute_avg_bleu, compute_f1


class InferEngine:
    def __init__(self, config):
        self.config = config
        self.device = config['device']

        self.tokenizer = T5Tokenizer.from_pretrained(config['model']['pretrained'])
        self.model = MT5ForConditionalGeneration.from_pretrained(config['model']['pretrained'])
        self.model.to(config['device'])

        self.model.load_state_dict(torch.load(self.config['inference']['file_path'], map_location=self.device))

        self.test_dataset = VQADataset(
            root=config['data']['test_root'],
            file_path=config['data']['test_file'],
            tokenizer=self.tokenizer
        )
        print(f"Test dataset: {len(self.test_dataset)} samples")

    def run_test(self):
        dataloader = self.test_dataset.get_loader(
            batch_size=self.config['inference']['bs'],
            shuffle=False,
        )
        pbar = tqdm(dataloader)

        qid2pred = {}
        qid2gt = {}
        self.model.eval()
        with torch.no_grad():
            for data in pbar:
                pixel_values = data["pixel_values"].to(self.device)
                question_ids = data["question_ids"].to(self.device)
                question_masks = data["question_mask"].to(self.device)
                answer_ids = data['answer_ids']
                qid = data['qid']

                generated_ids = self.model.generate(
                    input_ids=question_ids,
                    pixel_values=pixel_values,
                    attention_mask=question_masks,
                    max_length=64,
                    num_beams=1,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True
                )
                preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                         generated_ids]
                target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in answer_ids]
                assert len(preds) == len(qid)

                for k in range(len(qid)):
                    id_ = qid[k]
                    qid2pred[id_] = preds[k].replace("<extra_id_0>", "").strip().lower()
                    qid2gt[id_] = target[k].lower()

        f1 = compute_f1(qid2gt, qid2pred)
        bleu = compute_avg_bleu(qid2gt, qid2pred)
        print("F1: {:f}\nBLEU: {:f}".format(f1, bleu))

        with open(f'./outputs/private-test/results.json', 'w', encoding='utf-8') as f:
            json.dump(qid2pred, f, indent=4, ensure_ascii=False)
            
        with open('./groundtruth_results.json', 'w', encoding='utf-8') as f:
            json.dump(qid2gt, f, indent=4, ensure_ascii=False)

        return f1, bleu
