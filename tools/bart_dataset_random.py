from base64 import encode
from torch.utils.data import Dataset
from transformers import BartTokenizer
import random
import numpy as np
import copy
import torch


def negg_event2str(event):
    def mystr(s):
        return '' if s is None else str(' '+s)
    return mystr(event[3]) + mystr(event[0].replace('+',' ')) + mystr(event[4]) + mystr(event[5])

def event2str(event):
    def mystr(s):
        return '' if s == 'None' else str(' '+s)
    return mystr(event[2]) + mystr(event[1].replace('+',' ')) + mystr(event[3]) + mystr(event[4])

def new_verb_event2str(event):
    def mystr(s):
        return '' if s == 'None' else str(' '+s)
    return mystr(event[2]) + mystr(event[0].replace('+',' ')) + mystr(event[3]) + mystr(event[4])

class bart_dataset_random(Dataset):
    def __init__(self,raw_data,args,state):
        self.raw_data = raw_data
        self.args = args
        self.tokenizer = BartTokenizer.from_pretrained(args.pretrained_model_path) 
        if self.args.data_dir.split('/')[-1] == 'negg_data':
            self.event2str = negg_event2str
        elif self.args.data_dir.split('/')[-1] == 'raw_data_verb':
            self.event2str = new_verb_event2str
        else:
            self.event2str = event2str
        
        self.state = state
            
    
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        if self.args.random_span and self.state == 'train':
            if len(self.raw_data[index]) == 4:
                context,sent,answers,target = self.raw_data[index]
            else:
                context,answers,target = self.raw_data[index]
            
            raw_event_list = []
            for event in context:
                event_repr = self.event2str(event)
                raw_event_list.append(event_repr[1:])
            raw_event_list.append(self.event2str(answers[target])[1:])
            raw_tokens_list = [self.tokenizer.tokenize(event) for event in raw_event_list]
            raw_tokens_list_flatten = []
            for event in raw_tokens_list:
                raw_tokens_list_flatten.extend(event)
                raw_tokens_list_flatten.append('.')
            mask_num = random.randint(1,self.args.mask_num)
            mask_indexs = random.sample(range(0, 9),mask_num)
            list.sort(mask_indexs)
            mask_len = [len(raw_tokens_list[idx]) for idx in mask_indexs]
            sum_len = len(raw_tokens_list_flatten)
            mask_span = []
            for i in range(mask_num):
                max_legal_idx = sum_len-sum(mask_len)-1
                start_idx = random.randint(0,max_legal_idx)
                end_idx = start_idx+mask_len[0]
                mask_len = mask_len[1:]
                mask_span.append([start_idx,end_idx])
            decode_token_list = ['.']
            mask_len = [len(raw_tokens_list[idx]) for idx in mask_indexs]
            for i in range(len(mask_len)):
                decode_token_list.extend(raw_tokens_list_flatten[mask_span[i][0]:mask_span[i][1]])
                decode_token_list.append('.')
            
            for i,span in enumerate(mask_span):
                del raw_tokens_list_flatten[span[0]:span[1]]
                raw_tokens_list_flatten.insert(span[0],'<mask>')
                if i!=len(mask_span)-1:
                    mask_span[i+1][0] -= mask_len[i]-1
                    mask_span[i+1][1] -= mask_len[i]-1

            raw_tokens_list_flatten  = raw_tokens_list_flatten[0:self.args.encode_max_length]
            encode_inputs = ['<s>'] + raw_tokens_list_flatten + ['</s>'] + ['<pad>' for i in range(self.args.encode_max_length-len(raw_tokens_list_flatten)-2)]
            encode_inputs = [self.tokenizer._convert_token_to_id(id) for id in encode_inputs]
            encode_masks = [1 for _ in range(len(raw_tokens_list_flatten)+2)]+[0 for _ in range(self.args.encode_max_length-len(raw_tokens_list_flatten)-2)]

            decode_token_list  = decode_token_list[0:self.args.decode_max_length]
            decode_inputs = ['<s>'] + decode_token_list + ['</s>'] + ['<pad>' for i in range(self.args.decode_max_length-len(decode_token_list)-2)]
            decode_inputs = [self.tokenizer._convert_token_to_id(id) for id in decode_inputs]
            decode_masks = [1 for _ in range(len(decode_token_list)+2)]+[0 for _ in range(self.args.decode_max_length-len(decode_token_list)-2)]
            labels = copy.deepcopy(decode_inputs)
            example = [encode_inputs,encode_masks,decode_inputs,decode_masks,labels,target]
            example = [torch.tensor(t,dtype=torch.int32) for t in example]
            return example
        elif self.state == 'train' and self.args.pretrain: 
            if len(self.raw_data[index]) == 4:
                context,sent,answers,target = self.raw_data[index]
            else:
                context,answers,target = self.raw_data[index]
            raw_event_list = []
            for event in context:
                event_repr = self.event2str(event)
                raw_event_list.append(event_repr[1:])
            raw_event_list.append(self.event2str(answers[target])[1:])
            # raw_tokens_list = [self.tokenizer.tokenize(event) for event in raw_event_list]

            mask_num = random.randint(1,self.args.mask_num)
            mask_indexs = random.sample(range(0, 9),mask_num)
            list.sort(mask_indexs)

            encode_input = ''
            for i in range(9):
                if i in mask_indexs:
                    encode_input += '<mask> . '
                else:
                    encode_input+=  raw_event_list[i] + ' . '
            decode_input = '. '
            for i in mask_indexs:
                decode_input += raw_event_list[i] + ' . '
            decode_input = decode_input[:-1]

            
            encode_input_tokenized = self.tokenizer(encode_input,
                                    add_special_tokens=True,
                                    return_token_type_ids=False,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.args.encode_max_length)
            
            decode_input_tokenized = self.tokenizer(decode_input,
                                    add_special_tokens=True,
                                    return_token_type_ids=False,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.args.decode_max_length)
            encode_inputs = encode_input_tokenized['input_ids']
            encode_masks = encode_input_tokenized['attention_mask']
            decode_inputs = decode_input_tokenized['input_ids']
            decode_masks = decode_input_tokenized['attention_mask']
            labels = copy.deepcopy(decode_input_tokenized['input_ids'])
            example = [encode_inputs,encode_masks,decode_inputs,decode_masks,labels,target]
            example = [torch.tensor(t,dtype=torch.int32) for t in example]
            return example
        else:
            if len(self.raw_data[index]) == 4:
                context,sent,answers,target = self.raw_data[index]
            else:
                context,answers,target = self.raw_data[index]
            encode_inputs = []
            encode_masks = []
            decode_inputs = []
            decode_masks = []
            labels  = []
            
            raw_event_list = []
            for event in context:
                event_repr = self.event2str(event)
                raw_event_list.append(event_repr[1:])
            # raw_event_list.append(self.event2str(answers[target])[1:])
            # raw_tokens_list = [self.tokenizer.tokenize(event) for event in raw_event_list]

            encode_input = ''
            for i in range(9):
                if i==8:
                    encode_input += '<mask> . '
                else:
                    encode_input +=  raw_event_list[i] + ' . '
            encode_input_tokenized = self.tokenizer(encode_input,
                        add_special_tokens=True,
                        return_token_type_ids=False,
                        padding="max_length",
                        truncation=True,
                        max_length=self.args.encode_max_length)
            for i in range(5):
                encode_inputs.append(encode_input_tokenized['input_ids'])
                encode_masks.append(encode_input_tokenized['attention_mask'])

            for answer in answers:
                decode_input = '. ' + self.event2str(answer)[1:] + ' .'
                if decode_input == '. city categorize links .':
                    decode_input = '. city city city .'
                # decode_input = self.event2str(answer)[1:]
                decode_input_tokenized = self.tokenizer(decode_input,
                                        add_special_tokens=True,
                                        return_token_type_ids=False,
                                        padding="max_length",
                                        truncation=True,
                                        max_length=self.args.eval_decode_max_length)
                decode_inputs.append(decode_input_tokenized['input_ids'])
                decode_masks.append(decode_input_tokenized['attention_mask'])
            labels = copy.deepcopy(decode_inputs)
            example = [encode_inputs,encode_masks,decode_inputs,decode_masks,labels,target]
            example = [torch.tensor(t,dtype=torch.int32) for t in example]
            return example