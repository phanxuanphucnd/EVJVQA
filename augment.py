import copy
import json
import string

import httpx
from googletrans import Translator
from langdetect import detect
from num2words import num2words
from tqdm import tqdm

# from ofa.ofa_infer import OFAInference
from translator import EnViVinAITranslator, JaEnMarianTranslator

envi_translator = EnViVinAITranslator()
# jaem_translator = JaEnMarianTranslator()

with open('./data/train/evjvqa_train_lang-short-anw.json', 'r', encoding='utf=8') as f:
    train_data = json.load(f)
    
images = train_data['images']
annotations = train_data['annotations']

new_annotations = []

def remove_punc(text):
    text = text.lower()
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)


timeout = httpx.Timeout(30) 
gg_translator = Translator(timeout=timeout)


idx = 23785

for anno in tqdm(annotations):
    if anno['language'] == 'en':
        #Todo: Translate to vi
        # new_sample = copy.deepcopy(anno)
        # question = envi_translator.translate_en2vi(anno['question'])
        # answer = envi_translator.translate_en2vi(anno['answer'])
        
        # new_sample['question'] = question.lower()
        # new_sample['answer'] = remove_punc(answer)
        # new_sample['language'] = 'vi'
        # new_sample['id'] = idx

        # idx += 1
        
        # new_annotations.append(new_sample)
        
        # Todo: Translate to ja
        new_sample = copy.deepcopy(anno)
        question = gg_translator.translate(anno['question'], src='en', dest='ja').text
        answer = gg_translator.translate(anno['answer'], src='en', dest='ja').text
        
        new_sample['question'] = question.lower()
        new_sample['answer'] = remove_punc(answer)
        new_sample['language'] = 'ja'
        new_sample['id'] = idx

        idx += 1
        
        new_annotations.append(new_sample)
        
    elif anno['language'] == 'vi':
        #Todo: Translate to en
        new_sample = copy.deepcopy(anno)
        question = envi_translator.translate_vi2en(anno['question'])
        answer = envi_translator.translate_vi2en(anno['answer'])
        
        new_sample['question'] = question.lower()
        new_sample['answer'] = remove_punc(answer)
        new_sample['language'] = 'en'

        new_sample['id'] = idx

        idx += 1
        
        new_annotations.append(new_sample)

        # Todo: Translate to ja
        new_sample = copy.deepcopy(anno)
        question = gg_translator.translate(anno['question'], src='vi', dest='ja').text
        answer = gg_translator.translate(anno['answer'], src='vi', dest='ja').text
        
        new_sample['question'] = question.lower()
        new_sample['answer'] = remove_punc(answer)
        new_sample['language'] = 'ja'
        new_sample['id'] = idx

        idx += 1
        
        new_annotations.append(new_sample)


annotations.extend(new_annotations)

with open('./data/train/evjvqa_train_lang-short-augment-ja.json', 'w', encoding='utf=8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)
