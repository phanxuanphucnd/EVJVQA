# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan

import json
import string

import httpx
from fuzzywuzzy import fuzz
from googletrans import Translator
from num2words import num2words
from tqdm import tqdm

vi_question_type_mapping = {
    'bao nhiêu': 'HOW_MANY',
    'ở đâu': 'WHERE',
    'màu gì': 'WHAT_COLOR',
    ' ai ': 'WHO',
    ' ai?': 'WHO',
    'làm gì': 'WHAT_DO',
    'cái gì': 'WHAT_IS',
    'vật gì': 'WHAT_IS',
    'như thế nào': 'HOW'
}

vi_question_type_mapping_inversed = {
    'HOW_MANY': ['bao nhiêu'],
    'WHERE': ['ở đâu'],
    'WHAT_COLOR': ['màu gì'],
    'WHO': [' ai ', ' ai?'],
    'WHAT_DO': ['làm gì'],
    'WHAT_IS': ['cái gì', 'vật gì'],
    'HOW': ['như thế nào']
}

en_question_type_mapping = {
    'how many': 'HOW_MANY',
    'where': 'WHERE',
    'what color': 'WHAT_COLOR',
    'who': 'WHO',
    'whom': 'WHO',
    'what is': 'WHAT_IS',
    'what are': 'WHAT_IS',
    'what do': 'WHAT_DO',
    'what does': 'WHAT_DO',
    'which': 'WHICH',
    'how': 'HOW'
}


# ofa_maps = ['HOW_MANY', 'WHERE', 'WHAT_COLOR', 'WHO']
ofa_maps = ['HOW_MANY', 'WHAT_COLOR']


class ReWriting(object):
    def __init__(self) -> None:
        timeout = httpx.Timeout(30) 
        self.gg_translator = Translator(timeout=timeout)
        self.vi_num2words_dict  ={
            0: 'không',
            1: 'một',
            2: 'hai',
            3: 'ba',
            4: 'bốn',
            5: 'năm',
            6: 'sáu',
            7: 'bảy',
            8: 'tám',
            9: 'chín',
            10: 'mười'
        }

    def remove_punc(self, text):
        text = text.lower()
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def check_question_type(self, text, language):
        if language == "vi":
            for k, v in vi_question_type_mapping.items():
                if k in text.lower():
                    return v
            
            return 'OTHERS'
        
        elif language == 'en':
            for k, v in en_question_type_mapping.items():
                if k in text.lower():
                    if v == 'WHAT_IS':
                        if ' do?' in text.lower() or  'doing?' in text.lower():
                            return 'WHAT_DO'
                    return v
            
            return 'OTHERS'
        
        elif language == 'ja':
            en_text = self.gg_translator.translate(text, src='ja', dest='en').text
            for k, v in en_question_type_mapping.items():
                if k in en_text.lower():
                    return v
            
            return 'OTHERS'

    def get_samples_by_id(self, image_id, annotations):
        return_samples = []
        for anno in annotations:
            if anno['image_id'] == image_id:
                return_samples.append(anno)

        return return_samples

    
    def run(self, ofa_predict_file, scratch_predict_file, test_file):
        with open(ofa_predict_file, 'r', encoding='utf-8') as f:
            ofa_data = json.load(f)

        ofa_predicts = []
        for k, v in ofa_data.items():
            ofa_predicts.append(v)

        with open(scratch_predict_file, 'r', encoding='utf-8') as f:
            scratch_data = json.load(f)

        scratch_predicts = []
        for k, v in scratch_data.items():
            scratch_predicts.append(v)

        
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        annotations = test_data['annotations']

        return_answers = {}

        for i, anno in tqdm(enumerate(annotations)):
            question_type = anno['question_type']
            if question_type in ofa_maps:           #TODO: Use model OFA
                if anno['language'] == 'en':                        
                    answer = ofa_predicts[i]
                    split_ans = answer.split()
                    ans = []
                    for w in split_ans:
                        try:
                            ans.append(num2words(int(w)).replace('-', ' '))
                        except:
                            ans.append(w)
                    answer = ' '.join(ans)

                    # if question_type == 'HOW_MANY':
                    #     if answer in ['one', '1']:
                    #         new_answer = anno['question'].replace('how many', f"there is {answer}")
                    #     else:
                    #         new_answer = anno['question'].replace('how many', f"there are {answer}")

                    #     answer = new_answer
                else:
                    gr_samples = self.get_samples_by_id(anno['image_id'], annotations)
                    en_question = self.gg_translator.translate(anno['question'], src=anno['language'], dest='en').text
                    en_question = en_question.lower()
                    
                    max_score = 0
                    similar_sample = None
                    for sample in gr_samples:                    
                        if sample['language'] == 'en':
                            fuzze_score = fuzz.partial_ratio(en_question, sample['question'])
                            if fuzze_score > max_score:
                                similar_sample = sample
                                max_score = fuzze_score

                    if max_score >= 85 and similar_sample:
                        tmp_answer = ofa_predicts[similar_sample['id']]
                        answer = self.gg_translator.translate(tmp_answer, src='en', dest=anno['language']).text
                        if anno['language'] == 'vi':
                            split_ans = answer.split()
                            ans = []
                            for w in split_ans:
                                try:
                                    # ans.append(self.vi_num2words_dict[int(w)])
                                    ans.append(num2words(int(w), lang='vi'))
                                except:
                                    ans.append(w)
                            answer = ' '.join(ans)
                            if question_type in ['WHO']:
                                new_answer = anno['question']
                                for cand in vi_question_type_mapping_inversed[question_type]:
                                    new_answer = new_answer.replace(cand, f" {answer} ")
                            else:
                                new_answer = anno['question']
                                for cand in vi_question_type_mapping_inversed[question_type]:
                                    new_answer = new_answer.replace(cand, answer)

                            new_answer = self.remove_punc(new_answer).strip()
                            answer = new_answer
                            
                            # split = new_answer.split()
                            # if len(split) >= 5:
                            #     if answer in ' '.join(split[: int((len(split))/ 2)]):
                            #         answer = ' '.join(split[: int((len(split))/ 2)])
                            #         print(answer, ' - ', split)
                            #     else:
                            #         answer = ' '.join(split[int((len(split))/ 2): ])
                            # else:
                            #     answer = new_answer
                        else:
                            if question_type == "HOW_MANY":
                                # split_ans = answer.split('')
                                # ans = []
                                # for w in answer:
                                #     try:
                                #         ans.append(num2words(int(w), lang='ja'))
                                #     except:
                                #         ans.append(w)
                                # answer = ''.join(ans)
                                new_answer = scratch_predicts[i]
                                tmp_answer = []
                                for w in new_answer:
                                    if w.isdigit():
                                        tmp_answer.append(answer)
                                    else:
                                        tmp_answer.append(w)

                                answer = ''.join(tmp_answer)
                            else:
                                answer = scratch_predicts[i]

                    else:
                        answer = scratch_predicts[i]

            else:
                if anno['language'] == 'en':
                    answer = ofa_predicts[i]
                    split_ans = answer.split()
                    ans = []
                    for w in split_ans:
                        try:
                            ans.append(num2words(int(w)).replace('-', ' '))
                        except:
                            ans.append(w)
                    answer = ' '.join(ans)
                else:
                    answer = scratch_predicts[i]
        
            return_answers[i] = answer

        
        with open('outputs/results.json', 'w', encoding='utf-8') as f:
            json.dump(return_answers, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':

    rewriting = ReWriting()

    # rewriting.run(
    #     ofa_predict_file='outputs/results-ofa.json',
    #     scratch_predict_file='outputs/results-0.234.json',
    #     test_file='data/test/official_evjvqa_public_test_lang_qtype.json'
    # )


    #TODO: Run identify Question Type
    with open('data/test/official_evjvqa_public_test_lang_qtype.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    annotations = test_data['annotations']

    for anno in tqdm(annotations):
        # if anno['question_type'] == 'OTHERS':
        qtype = rewriting.check_question_type(anno['question'], language=anno['language'])
        anno['question_type'] = qtype

    with open('data/test/official_evjvqa_public_test_lang_qtype-detailed.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
