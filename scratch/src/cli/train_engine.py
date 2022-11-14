import os
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm
from .. dataset.vqa_dataset import VQADataset
from .. model import MT5ForConditionalGeneration
from .. model.xglm.modeling_xglm import XGLMForConditionalGeneration
from .. evaluate.eval import compute_f1, compute_avg_bleu
from transformers import AutoTokenizer, T5Tokenizer, XGLMTokenizer, XGLMForCausalLM

class TrainEngine:
    def __init__(self, config):
        self.config = config
        self.device = config['device']

        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

        if self.config['model']['pretrained_type'] == 'mt5':
            self.tokenizer = T5Tokenizer.from_pretrained(config['model']['pretrained'])
            self.model = MT5ForConditionalGeneration.from_pretrained(config['model']['pretrained'])
        elif self.config['model']['pretrained_type'] == 'xglm':
            self.tokenizer = XGLMTokenizer.from_pretrained(config['model']['pretrained'])
            self.model = XGLMForConditionalGeneration.from_pretrained(config['model']['pretrained'])

        self.model.to(config['device'])

        self.train_dataset = VQADataset(
            root=config['data']['train_root'],
            file_path=config['data']['train_file'],
            tokenizer=self.tokenizer,
        )
        self.valid_dataset = VQADataset(
            root=config['data']['test_root'],
            file_path=config['data']['test_file'],
            tokenizer=self.tokenizer,
        )

        # self.test_dataset = VQADataset(
        #     root=config['data']['test_root'],
        #     file_path=config['data']['test_file'],
        #     tokenizer=self.tokenizer
        # )
        
        print('--------------------------')
        print(f"Train dataset: {len(self.train_dataset)} samples.")
        print(f"Valid dataset: {len(self.valid_dataset)} samples.")
        # print(f"Test dataset: {len(self.test_dataset)} samples.")
        print('--------------------------')

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=config['training']['lr']
        )

    def train(self):
        dataloader = self.train_dataset.get_loader(
            batch_size=self.config['training']['bs'],
            shuffle=True,
        )
        if not os.path.exists(self.config['training']['output_dir']):
            os.mkdir(self.config['training']['output_dir'])

        min_val_loss = 1000
        max_f1 = 0
        for epoch in range(self.config['training']['n_epochs']):
            print("Epoch: ", epoch)
            self.model.train()
            pbar = tqdm(dataloader)
            total_loss = 0
            evg_loss = 0
            for i, data in enumerate(pbar):
                pixel_values = data["pixel_values"].to(self.device)
                question_ids = data["question_ids"].to(self.device)
                question_masks = data["question_mask"].to(self.device)
                answer_ids = data['answer_ids'].to(self.device)
                y_ids = answer_ids[:, :-1].contiguous()
                lm_labels = answer_ids[:, 1:].clone().detach()
                lm_labels[answer_ids[:, 1:] == self.tokenizer.pad_token_id] = -100

                outputs = self.model(
                    input_ids=question_ids,
                    pixel_values=pixel_values,
                    attention_mask=question_masks,
                    decoder_input_ids=y_ids,
                    labels=lm_labels,
                )

                loss = outputs[0]
                total_loss += loss
                pbar.set_description(f"Train loss: {total_loss/(i+1)}")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            valid_loss = total_loss/(i+1)
            valid_f1, valid_bleu = self.compute_score(self.valid_dataset)

            # if valid_loss <= min_val_loss:
            #     print(f">>> Horaay!! Save new best loss checkpoint - Valid Loss: {valid_loss}")
            #     min_val_loss = valid_loss
            #     path = os.path.join(self.config['training']['output_dir'], "best_loss_ckpt.pt")
            #     torch.save(self.model.state_dict(), path)

            if valid_f1 >= max_f1:
                print(f">>> Horaay!! Save new best F1 checkpoint - Valid F1: {valid_f1} | Valid Bleu: {valid_bleu}")
                max_f1 = valid_f1
                path = os.path.join(self.config['training']['output_dir'], "best_f1_ckpt.pt")
                torch.save(self.model.state_dict(), path)
            else:
                print(f"** Evaluate valid: | F1 = {valid_f1} | Bleu = {valid_bleu}.")

            if self.config['training']['save_last']:
                path = os.path.join(self.config['training']['output_dir'], f"last_ckpt.pt")
                torch.save(self.model.state_dict(), path)

            if (epoch + 1) % 10 == 0 and (epoch + 1) > 50:
                path = os.path.join(self.config['training']['output_dir'], f"{(epoch + 1)}_ckpt.pt")
                torch.save(self.model.state_dict(), path)

                
    def compute_score(self, dataset):
        dataloader = dataset.get_loader(
            batch_size=64,
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
            
        return f1, bleu

    def validate(self):
        dataloader = self.valid_dataset.get_loader(
            batch_size=16,
            shuffle=False
        )
        pbar = tqdm(dataloader)
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(pbar):
                pixel_values = data["pixel_values"].to(self.device)
                question_ids = data["question_ids"].to(self.device)
                question_masks = data["question_mask"].to(self.device)
                answer_ids = data['answer_ids'].to(self.device)
                y_ids = answer_ids[:, :-1].contiguous()
                lm_labels = answer_ids[:, 1:].clone().detach()
                lm_labels[answer_ids[:, 1:] == self.tokenizer.pad_token_id] = -100

                outputs = self.model(
                    input_ids=question_ids,
                    pixel_values=pixel_values,
                    attention_mask=question_masks,
                    decoder_input_ids=y_ids,
                    labels=lm_labels,
                )

                loss = outputs[0]
                total_loss += loss
                pbar.set_description(f"valid-loss: {total_loss / (i + 1)}")

        return total_loss/len(dataloader)

