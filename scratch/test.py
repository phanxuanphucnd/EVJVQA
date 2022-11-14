import json
from tqdm.autonotebook import tqdm
from src.evaluate.eval import compute_f1, compute_avg_bleu

with open('datasets/evjvqa_public_test-lang-qtype-answer.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

test_annotations = test_data['annotations']

qid2lang = {}
qid2qtype = {}

for anno in tqdm(test_annotations):
    qid2lang[str(anno['id'])] = anno['language']
    qid2qtype[str(anno['id'])] = anno['question_type']
    
print(len(test_annotations), len(qid2lang), len(qid2qtype))


with open('groundtruth_results.json', 'r', encoding='utf-8') as f:
    gt_results = json.load(f)

with open('results.json', 'r', encoding='utf-8') as f:
    mt5vit_results = json.load(f)

sep_qtype = True
languages = ['en', 'vi', 'ja']
# question_types = list(set(qid2qtype.values()))
question_types = ['HOW_MANY', 'WHAT_COLOR', 'WHERE', 'WHO', 'HOW', 'WHAT_IS', 'WHAT_DO', 'WHICH', 'OTHERS']

for lang in languages:
    pred = {}
    grth = {}
    print(f'------------------- {lang.upper()} -------------------')
    if sep_qtype:
        for qtype in question_types:
            pred = {}
            grth = {}
            for k, v in mt5vit_results.items():
                if qid2lang[k] == lang and qid2qtype[k] == qtype:
                    pred[k] = v
                    grth[k] = gt_results[k]
            
            f1 = compute_f1(a_gold=grth, a_pred=pred)
            bleu = compute_avg_bleu(a_gold=grth, a_pred=pred)
            
            print(f"Metrics of Language={lang} - Question Type={qtype}: F1 = {f1} and Bleu = {bleu}")
    
        print(f'------------------- END -------------------')
    else:
        for k, v in mt5vit_results.items():
            if qid2lang[k] == lang:
                pred[k] = v
                grth[k] = gt_results[k]

        f1 = compute_f1(a_gold=grth, a_pred=pred)
        bleu = compute_avg_bleu(a_gold=grth, a_pred=pred)
            
        print(f"Metrics of Language={lang}: F1 = {f1} and Bleu = {bleu}")
