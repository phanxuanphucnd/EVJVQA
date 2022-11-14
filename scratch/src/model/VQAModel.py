import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import ViTModel, SwinModel, DeiTModel, CLIPVisionModel, AutoModel
from transformers import AutoConfig, EncoderDecoderConfig, EncoderDecoderModel

class VQAModel(nn.Module):
    def __init__(self, config):
        super(VQAModel, self).__init__()

        self.config = config

        model = EncoderDecoderModel.from_encoder_decoder_pretrained(config['pretrained'], config['pretrained'])

        if 'vit' in config['visual_pretrained']:
            self.img_encoder = ViTModel.from_pretrained(config['visual_pretrained'])
        elif 'clip' in config['visual_pretrained']:
            self.img_encoder = CLIPVisionModel.from_pretrained(config['visual_pretrained'])
        elif 'deit' in config['visual_pretrained']:
            self.img_encoder = DeiTModel.from_pretrained(config['visual_pretrained'])
        elif 'swin' in config['visual_pretrained']:
            self.img_encoder = SwinModel.from_pretrained(config['visual_pretrained'])
        else:
            self.img_encoder = AutoModel.from_pretrained(config['visual_pretrained'])

        self.text_encoder = model.encoder
        self.text_decoder = model.decoder

        self.lm_head = nn.Linear(self.text_encoder.config.hidden_size,
                                 self.text_decoder.config.vocab_size,
                                 bias=False)
    def forward(
            self,
            pixel_values,
            question_ids,
            answer_ids=None,

    ):
        pixel_embeds = self.img_encoder(pixel_values).last_hidden_state
        question_embeds = self.text_encoder(question_ids).last_hidden_state

        encoder_hidden_states = torch.cat([question_embeds, pixel_embeds], dim=1)

        decoder_outputs = self.text_decoder(
            input_ids=answer_ids,
            encoder_hidden_states=encoder_hidden_states
        )
        # print(decoder_outputs.logits.shape, self.text_decoder.config.vocab_size)
        lm_logits = decoder_outputs[0]
        # lm_logits = self.lm_head(sequence_output)

        loss = None
        if answer_ids is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), answer_ids.view(-1))

        output = lm_logits
        return (loss, output) if loss is not None else output
