import gc
import pickle
import torch
import random
import time
import logging
import math
import datetime
import numpy as np
import os
import argparse
from datetime import date
from models.base.cot import ComplementEntropy
from transformers import RobertaTokenizer,get_linear_schedule_with_warmup,RobertaForMultipleChoice,AdamW,get_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from models.bart_1cls import bart_1cls
from models.bart_mask_random import bart_mask_random
from tools.bart_dataset_random import bart_dataset_random
from multiprocessing import Pool
from torch.optim import Adam
from tools.common import seed_everything,Args,format_time
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import get_rank
from torch.utils.tensorboard import SummaryWriter  
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from apex import amp

MODEL_CLASSES = {
    'bart_1cls': bart_1cls,
    'bart_mask_random' : bart_mask_random
}

def class_acc(preds, labels):    

    correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()         
    acc = correct.sum().item() / len(correct)
    return acc

def train(args,train_dataloader,model,optimizer,lr_scheduler,writer,logger=None,global_step=0):
    t0 = time.time()
    avg_loss, avg_acc = [],[]

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    model.zero_grad()
    for step, batch in enumerate(train_dataloader):
        model.train()
        batch = [t.long() for t in batch]
        batch = tuple(t.to(args.device) for t in batch)
        output = model(*tuple(batch))
        loss, logits,labels = output
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        loss = loss.item()
        avg_loss.append(loss)
        if labels is None :
            labels =  batch[-1] 
        acc = class_acc(logits, labels) if logits!=None else 0
        avg_acc.append(acc)
        if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0*args.gradient_accumulation_steps)
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()
            global_step += 1
        
        # if args.COT:
        #     _,logits,targets = model(*tuple(batch))
        #     loss_fct = ComplementEntropy()
        #     complement_loss = loss_fct(logits,targets)
        #     complement_optimizer.zero_grad()
        #     complement_loss.backward()
        #     complement_optimizer.step()

        if global_step % 10==0 and args.local_rank in [-1,0]:
                writer.add_scalar('loss', loss, global_step)
                writer.add_scalar('avg_acc', np.array(avg_acc).mean(), global_step)
                writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], global_step)
        if (step+1) % args.log_step == 0 and args.local_rank in [-1,0]:
            elapsed = format_time(time.time() - t0)
            logger.info('Batch {:>5,} of {:>5,}.Loss: {:} Acc:{:} Avg_acc:{:} Elapsed:{:}.'
            .format(step+1, len(train_dataloader),format(loss, '.4f'),format(acc, '.4f'),format(np.array(avg_acc).mean(), '.4f'),elapsed))
 
    avg_loss = np.array(avg_loss).mean()
    avg_acc = np.array(avg_acc).mean()
    return avg_loss, avg_acc,global_step


def evaluate(test_dataloader,model,args):
    avg_acc = []
    model.eval()   
    with torch.no_grad():
        for batch in tqdm(test_dataloader,total=len(test_dataloader)):
            batch = tuple(t.to(args.device) for t in batch)
            batch = [t.long() for t in batch]
            loss, logits,labels = model(*tuple(batch))
            if labels is None :
                labels =  batch[-1] 
            acc = class_acc(logits, labels)
            avg_acc.append(acc)
    avg_acc = np.array(avg_acc).mean()
    return avg_acc
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',help='config file path')
    config_file = parser.parse_args().config_file
    args = Args(config_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    start_date = date.today().strftime('%m-%d')
    if args.eval:
        log_path = './log/{}/{}-eval.log'.format(start_date,args.annotation)
    else:
        log_path = './log/{}/{}.log'.format(start_date,args.annotation)
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    torch.cuda.empty_cache()
    seed_everything(args.seed)
    if args.multi_gpu and args.use_gpu:
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else :
        args.local_rank = -1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_gpu == False:
        device = torch.device('cpu')
    args.device = device
    logger = None
    if args.local_rank in [-1,0]:
        logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s | %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            filename=log_path,
            filemode=args.filemode)
        logger = logging.getLogger()
        logger.info("Process rank: {}, device: {}, distributed training: {}".format(
                    args.local_rank,device, bool(args.local_rank != -1)))
        logger.info("Training/evaluation parameters %s", args.to_str())

    if args.eval:
        model = MODEL_CLASSES[args.model_type](args)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'],strict=False)
        model.to(args.device)
        dev_batch = pickle.load(open(os.path.join(args.data_dir,'test'),'rb'))
        dev_data = bart_dataset_random(dev_batch,args,'eval')
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=8)
        test_acc = evaluate(dev_dataloader,model,args)
        logger.info("test_acc{}".format(test_acc))
        return
    start_epoch = 0

    model = MODEL_CLASSES[args.model_type](args)
    if args.resume:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(dict([(n, p) for n, p in checkpoint['model'].items()]), strict=False)
    if args.noise_lambda != 0:
        for name ,para in model.named_parameters():
            model.state_dict()[name][:] += (torch.rand(para.size())-0.5)*args.noise_lambda*torch.std(para)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #optimizer_to(optimizer,args.device)
    start_epoch = 0
  
    optimizer = Adam(optimizer_grouped_parameters,eps = args.epsilon,betas=(0.9,0.98),lr=args.lr)
    # complement_optimizer = Adam(optimizer_grouped_parameters,eps = args.epsilon,betas=(0.9,0.98),lr=args.lr)
    # len(data) = 1440295 140331
    if args.data_dir.split('/')[-1] == 'negg_data':
        train_num = 140331
    else:
        train_num = 1440295
    num_update_steps_per_epoch = math.ceil(((train_num/(args.gpu_num*args.per_gpu_train_batch_size))) / args.gradient_accumulation_steps)
    if args.max_train_steps == 0:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    args.num_warmup_steps = args.max_train_steps*0.05
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )    
    model.to(args.device)
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.local_rank in [-1,0]:
        tensorboard_path = './tensorboard/{}/{}'.format(start_date,args.annotation)
        if not os.path.exists(os.path.dirname(tensorboard_path)):
            os.makedirs(os.path.dirname(tensorboard_path))
        writer = SummaryWriter(tensorboard_path)
    else:
        writer = None
    global_step = 0
    best_performance = 0
    best_checkpoint_path = None
    if args.debug:
        train_raw_data = pickle.load(open('','rb'))
    else:
        train_raw_data = pickle.load(open(os.path.join(args.data_dir,'train'),'rb'))
    patience = args.patience
    fail_time = 0
    for epoch in range(int(args.num_train_epochs)):
        if fail_time>=patience:
            break
        if epoch < start_epoch:
            continue
        if args.local_rank in [-1,0]:
            logger.info('local_rank={},epoch={}'.format(args.local_rank, epoch))
        train_dataset = bart_dataset_random(train_raw_data,args,'train')
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size,num_workers=8)
        torch.cuda.empty_cache()
        train_loss, train_acc,global_step = train(args, train_dataloader,model,optimizer,lr_scheduler,writer,logger,global_step)
        if args.local_rank in [-1,0]:
            logger.info('epoch={},train_acc={},loss={}'.format(epoch, train_acc, train_loss))
        torch.cuda.empty_cache()
        gc.collect()
        if args.local_rank in [-1,0]:
            dev_batch = pickle.load(open(os.path.join(args.data_dir,'dev'),'rb'))
            dev_data = bart_dataset_random(dev_batch,args,'eval')
            dev_sampler = SequentialSampler(dev_data)
            dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=8)
            dev_acc = evaluate(dev_dataloader,model,args)
            writer.add_scalar('dev_acc', dev_acc, epoch)
            logger.info("epoch={},dev_acc={}".format(epoch,dev_acc))
            if dev_acc > best_performance:
                checkpoints_path = './checkpoints/{}/{}/epoch{}'.format(start_date,args.annotation,epoch)
                if not os.path.exists(checkpoints_path):
                    os.makedirs(checkpoints_path)
                best_checkpoint_path = os.path.join(checkpoints_path,'best_checkpoint.pt')
                #optimizer_to(optimizer,torch.device('cpu'))
                model.to(torch.device('cpu'))
                #'optimizer':optimizer.state_dict(),
                torch.save({'model':model.state_dict(),'epoch':epoch},best_checkpoint_path)
                logger.info('Save best checkpoint to {}'.format(best_checkpoint_path))
                #optimizer_to(optimizer,args.device)
                model.to(args.device)
                best_performance = dev_acc
                fail_time = 0
                logger.info("best_performance={},best_checkpoint_path={}".format(dev_acc,best_checkpoint_path))
            else:
                fail_time+=1
    if args.test and args.local_rank in [-1,0]:
        logger.info("test best_checkpoint_path={}".format(best_checkpoint_path))
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        dev_batch = pickle.load(open(os.path.join(args.data_dir,'test'),'rb'))
        dev_data = bart_dataset_random(dev_batch,args,'eval')
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=8)
        test_acc = evaluate(dev_dataloader,model,args)
        logger.info("best epoch={},test_acc={}".format(checkpoint['epoch'], test_acc))

if __name__=='__main__':
    main()

