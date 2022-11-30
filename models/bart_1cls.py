from torch import nn
from transformers.models.bart.modeling_bart import BartForConditionalGeneration,BartModel
from torch.nn import CrossEntropyLoss
from models.base.cot import ComplementEntropy
import torch

class bart_1cls(nn.Module):

    def __init__(self, args):
        super(bart_1cls, self).__init__()

        self.mlm = BartForConditionalGeneration.from_pretrained(args.pretrained_model_path)
        self.args = args
        if args.vocab_size is not None:
            self.mlm.resize_token_embeddings(args.vocab_size)
        self.config = self.mlm.config
        self.dropout = nn.Dropout(self.config.classifier_dropout)
        self.dropout = nn.Dropout(0.0)
        self.dense = nn.Linear(self.config.d_model, self.config.d_model)
        self.out_proj = nn.Linear(self.config.d_model, 1)
    
    def forward(
        self,encode_inputs,encode_masks,decode_inputs,decode_masks,labels,targets
    ):
        batch_size,num_choices,decode_len = decode_inputs.size()
        encode_len = encode_inputs.size()[-1]

        encode_inputs = encode_inputs.reshape(-1,encode_len)
        encode_masks = encode_masks.reshape(-1,encode_len)
        decode_inputs = decode_inputs.reshape(-1,decode_len)
        decode_masks = decode_masks.reshape(-1,decode_len)
      
        outputs = self.mlm.model(
            input_ids=encode_inputs,
            attention_mask=encode_masks,
            decoder_input_ids=decode_inputs,
            decoder_attention_mask=decode_masks
        )
        hidden_states = outputs.last_hidden_state
        eos_mask = decode_inputs.eq(self.config.eos_token_id)
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        hidden_states = self.dropout(sentence_representation)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        logits = hidden_states.view(batch_size,-1)
        if self.args.loss_fct == 'CrossEntropyLoss':
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits,targets)
        elif self.args.loss_fct == 'ComplementEntropy':
            loss_fct = CrossEntropyLoss()
            loss1 = loss_fct(logits,targets)
            loss_fct = ComplementEntropy()
            loss2 = loss_fct(logits,targets)
            loss = loss1+self.args.beta*loss2
        return loss,logits,None
        