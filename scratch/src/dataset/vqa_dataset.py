import json
import os
import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoFeatureExtractor

def read_data_json(file_path):
    data = json.load(open(file_path))
    img_id2filename = {}
    for item in data["images"]:
        if item["id"] not in img_id2filename:
            img_id2filename[item["id"]] = item["filename"]
        else:
            print("DUPLICATED IMG ID: ", item)

    return img_id2filename, data["annotations"]


class VQADataset(VisionDataset):
    def __init__(
            self,
            root: str,
            file_path: str,
            tokenizer,
            lowercase = False,
            max_length = 64,
            transform = None,
            target_transform = None,
            max_samples: int = None,
    ):
        super(VQADataset, self).__init__(root, transform, target_transform)

        image_paths = []
        questions = []
        answers = []
        qid = []

        imgid2path, annotated_data = read_data_json(file_path)

        vision_pretrained = "facebook/deit-base-patch16-224"

        for data_item in annotated_data:
            img_id = data_item["image_id"]
            img_file = imgid2path[img_id]
            if os.path.exists(os.path.join(self.root, img_file)):
                image_paths.append(img_file)
                questions.append(f"{data_item['language']}: {data_item['question']}")
                answers.append(data_item['answer'])
                qid.append(data_item['id'])

        if max_samples is None:
            max_samples = len(questions)

        self.image_paths = image_paths[:max_samples]
        self.questions = questions[:max_samples]
        self.answers = answers[:max_samples]
        self.qid = qid[:max_samples]

        if lowercase:
            self.questions = [q.lower() for q in self.questions]
            self.answers = [a.lower() for a in self.answers]

        self.tokenizer = tokenizer
        self.max_length = max_length

        # self.transform = transforms.Resize((800, 800))
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(vision_pretrained)

    def _load_image(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(os.path.join(self.root, path)) #.convert("RGB")
        # image = self.transform(image)
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()
        return pixel_values

    def _load_text(self, idx: int):
        question_text = self.questions[idx]
        question_text = " ".join(question_text.split())
        question = self.tokenizer.batch_encode_plus(
            [question_text],
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        answer_text = self.answers[idx]
        answer_text = " ".join(answer_text.split())
        answer = self.tokenizer.batch_encode_plus(
            [answer_text],
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return question, answer

    def __getitem__(self, idx: int):
        question, answer = self._load_text(idx)
        question_ids = question["input_ids"].squeeze()
        question_mask = question["attention_mask"].squeeze()
        answer_ids = answer["input_ids"].squeeze()

        pixel_values = self._load_image(idx)
        return pixel_values, question_ids, question_mask, answer_ids, self.qid[idx]

    def __len__(self):
        return len(self.questions)

    def collate_fn(self, items):
        # pixel_value_batch = (
        #     torch.stack([item[0] for item in items])
        # )
        # print([item[0] for item in items])
        batch = {
            "pixel_values": torch.stack([item[0] for item in items]),
            "question_ids": torch.stack([item[1] for item in items]),
            "question_mask": torch.stack([item[2] for item in items]),
            "answer_ids": torch.stack([item[3] for item in items]),
            "qid": [item[4] for item in items]
        }
        return batch


    def get_loader(
            self,
            batch_size=8,
            shuffle=True,
            drop_last=False,
            num_workers=0,
    ):
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )