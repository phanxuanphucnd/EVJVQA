# -*- coding: utf-8 -*-
# Copyright (c) 2022 by Phuc Phan


from re import L
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class EnViVinAITranslator:
    def __init__(self) -> None:
        self.tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en", src_lang="vi_VN")
        self.model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en")

        self.tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi", src_lang="en_XX")
        self.model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi")

    def translate_en2vi(self, text: str):
        input_ids = self.tokenizer_en2vi(text, return_tensors="pt").input_ids
        output_ids = self.model_en2vi.generate(
            input_ids,
            do_sample=True,
            top_k=100,
            top_p=0.8,
            decoder_start_token_id=self.tokenizer_en2vi.lang_code_to_id["vi_VN"],
            num_return_sequences=1,
        )
        vi_text = self.tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
        vi_text = " ".join(vi_text)

        return vi_text

    def translate_vi2en(self, text: str):
        input_ids = self.tokenizer_vi2en(text, return_tensors="pt").input_ids
        output_ids = self.model_vi2en.generate(
            input_ids,
            do_sample=True,
            top_k=100,
            top_p=0.8,
            decoder_start_token_id=self.tokenizer_vi2en.lang_code_to_id["en_XX"],
            num_return_sequences=1,
        )
        en_text = self.tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
        en_text = " ".join(en_text)

        return en_text


class JaEnMarianTranslator:
    def __init__(self) -> None:
        self.tokenizer_ja2en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-jap-en")
        self.model_ja2en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-jap-en")
        self.tokenizer_en2ja = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-jap")
        self.model_en2ja = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-jap")

    def translate_ja2en(self, text: str):
        input_ids = self.tokenizer_ja2en(f"{text}", return_tensors="pt", padding=True)
        output_ids = self.model_ja2en.generate(**input_ids)
        en_text = [self.tokenizer_ja2en.decode(t, skip_special_tokens=True) for t in output_ids]

        return en_text

    def translate_en2ja(self, text: str):
        input_ids = self.tokenizer_en2ja(f"{text}", return_tensors="pt", padding=True)
        output_ids = self.model_en2ja.generate(**input_ids)
        ja_text = [self.tokenizer_en2ja.decode(t, skip_special_tokens=True) for t in output_ids]

        return ja_text


if __name__ == "__main__":
    jaen_translator = JaEnMarianTranslator()
    ja_text = jaen_translator.translate_en2ja("My name is Wolfgang and I live in Berlin")
    en_text = jaen_translator.translate_ja2en('わが 名 は シナル と い い , また " わたし は 永遠 に 生き る " と .')

    print(ja_text)
    print(en_text)
