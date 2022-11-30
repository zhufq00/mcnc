from sre_constants import RANGE
from torch import nn
from transformers.models.bart.modeling_bart import BartForConditionalGeneration,BartLearnedPositionalEmbedding
from transformers.modeling_outputs import Seq2SeqLMOutput
from models.base.cot import ComplementEntropy
from torch.nn import CrossEntropyLoss
import torch
# torch.set_printoptions(precision=8,sci_mode=False)
import random
from transformers import BartTokenizer

class bart_mask_random(nn.Module):

    def __init__(self, args):
        super(bart_mask_random, self).__init__()

        self.mlm = BartForConditionalGeneration.from_pretrained(args.pretrained_model_path)
        if args.vocab_size is not None:
            self.mlm.resize_token_embeddings(args.vocab_size)
        self.tokenizer = BartTokenizer.from_pretrained(args.pretrained_model_path)
        self.args = args
        self.config = self.mlm.config

    def forward(
        self,encode_inputs,encode_masks,decode_inputs,decode_masks,labels,targets
    ):
        if self.args.pretrain and self.training:
            batch_size,decode_len = decode_inputs.size()
            encode_len = encode_inputs.size()[-1]
            labels[labels == self.config.pad_token_id] = -100
            outputs = self.mlm(
                input_ids=encode_inputs,
                attention_mask=encode_masks,
                decoder_input_ids = decode_inputs,
                decoder_attention_mask = decode_masks
            )
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss,None,None
        elif self.args.pretrain==False and self.training:
            batch_size,num_choices,decode_len = decode_inputs.size()
            encode_len = encode_inputs.size()[-1]

            encode_inputs = encode_inputs.reshape(-1,encode_len)
            encode_masks = encode_masks.reshape(-1,encode_len)
            decode_inputs = decode_inputs.reshape(-1,decode_len)
            decode_masks = decode_masks.reshape(-1,decode_len)
            labels = labels.reshape(-1,decode_len)
            labels[labels == self.config.pad_token_id] = -100
            outputs = self.mlm(
                input_ids=encode_inputs,
                attention_mask=encode_masks,
                decoder_input_ids = decode_inputs,
                decoder_attention_mask = decode_masks,
                output_attentions=True
            )
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            logits = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            logits = logits.reshape(batch_size,num_choices,decode_len-1)

            # logits = logits/self.args.temperature
            
            if self.args.pro_type=='sqrt':
                with torch.no_grad():
                    nonzero = torch.count_nonzero(logits,dim=2)+self.args.denominator_correction_factor
                logits = -(torch.sum(logits,dim=2)/nonzero)
            elif self.args.pro_type=='mul':
                with torch.no_grad():
                    nonzero = torch.count_nonzero(logits,dim=2)
                logits = -(torch.sum(logits,dim=2)+torch.log(nonzero.float()))
            else:
                logits = -torch.sum(logits,dim=2)
                
            if self.args.loss_fct == 'CrossEntropyLoss':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits,targets)
            elif self.args.loss_fct == 'MarginRankingLoss':
                # if self.args.temperature != 0:
                #     logits = logits/self.args.temperature
                if self.args.softmax:
                    logits = torch.softmax(logits,dim=1)
                scores_false = []
                scores_true = []
                for score,target in zip(logits,targets):
                    score_false =  torch.stack([score[i] for i in range(num_choices) if i != target ])
                    score_true = torch.stack([score[target] for i in range(num_choices-1)])
                    scores_false.append(score_false)
                    scores_true.append(score_true)
                scores_false = torch.stack(scores_false).view(-1)
                scores_true = torch.stack(scores_true).view(-1)
                loss_fct = nn.MarginRankingLoss(margin=self.args.margin)
                loss = loss_fct(scores_true,scores_false,torch.ones_like(scores_true))
            elif self.args.loss_fct == 'ComplementEntropy':
                loss_fct = CrossEntropyLoss()
                loss1 = loss_fct(logits,targets)
                loss_fct = ComplementEntropy()
                loss2 = loss_fct(logits,targets)
                loss = loss1+self.args.beta*loss2
            return loss,logits,targets
        else:
            batch_size,num_choices,decode_len = decode_inputs.size()
            encode_len = encode_inputs.size()[-1]

            encode_inputs = encode_inputs.reshape(-1,encode_len)
            encode_masks = encode_masks.reshape(-1,encode_len)
            decode_inputs = decode_inputs.reshape(-1,decode_len)
            decode_masks = decode_masks.reshape(-1,decode_len)
            labels = labels.reshape(-1,decode_len)

            labels[labels == self.config.pad_token_id] = -100
            outputs = self.mlm(
                input_ids=encode_inputs,
                attention_mask=encode_masks,
                decoder_input_ids = decode_inputs,
                decoder_attention_mask = decode_masks
            )

            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            logits = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            logits = logits.reshape(batch_size,num_choices,decode_len-1)

            # origin_logits = logits
            # sum_logits = -torch.sum(logits,dim=2)
            # encode_input_str = ' '.join([self.tokenizer.decode(token) if token != 1 else '' for token in encode_inputs[0]])
            # decode_input_str = [[self.tokenizer.decode(token) if token != 1 else '' for token in decode_inputs[i]] for i in range(5)]


            if self.args.pro_type=='sqrt':
                logits = -(torch.sum(logits,dim=2)/(torch.count_nonzero(logits,dim=2))+self.args.denominator_correction_factor)
            elif self.args.pro_type=='mul':
                logits = -(torch.sum(logits,dim=2)+torch.log(torch.count_nonzero(logits,dim=2).float()))
            else:
                logits = -torch.sum(logits,dim=2)
            
            # sum_target = torch.argmax(sum_logits)
            # predict_target = torch.argmax(logits)
            # print(encode_input_str[0:12])
            # if encode_input_str[0:12] == '<s> john son':
            #     print()
            # if predict_target == targets and sum_target != targets:

            #     print(encode_input_str)
            #     for i in range(5):
            #         for j in range(len(decode_input_str[i])):
            #             if decode_input_str[i][j+1] == '':
            #                 break
            #             print('%10s'%decode_input_str[i][j+1],end='')
            #         print()
            #         for j in range(len(decode_input_str[i])):
            #             if decode_input_str[i][j+1] == '':
            #                 break
            #             print('%10s'%('%.1f'%origin_logits[0][i][j].item()),end='')
            #         print()
            #     print(targets[0].item())
            #     print()
            
            return None,logits,None
