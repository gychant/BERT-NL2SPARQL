# coding: utf-8

"""Train and evaluate the model"""

import argparse
import random
import logging
import os
import json

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange, tqdm
from collections import OrderedDict

from model import BertNTR
from data_loader import DataLoader
from bert.optimization import BertAdam, warmup_linear
from bert import BertConfig
import utils
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

#predefine placeholders
qmark_placeholder = '[unused21]'
uri_placeholder = '[unused22]'
whitespace_placeholder = '[unused10]'
double_whitespace_placeholder = '[unused11]'

# warmup_linear_constant
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='WebQSP_0824_67', help="Directory containing the dataset")
parser.add_argument('--bert_model_dir', default='uncased_L-24_H-1024_A-16', help="Directory containing the BERT model in PyTorch")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir containing weights to reload before training")
parser.add_argument('--clip_grad', default=2, help="")
parser.add_argument('--min_epoch_num', default=3, help="minimal num of training epoch")
parser.add_argument('--patience', default=0.001, help="")
parser.add_argument('--patience_num', default=1000, help="num of epoch without improvement, then early stop")
parser.add_argument('--seed', type=int, default=9012, help="random seed for initialization")
parser.add_argument('--schedule', default='warmup_linear', help="schedule for optimizer")
parser.add_argument('--multi_gpu', default=False, action='store_true', help="Whether to use multiple GPUs if available")
parser.add_argument('--weight_decay', default=0.01, help="")
parser.add_argument('--warmup', default=0.1, help="")
parser.add_argument('--debug', default=False, help="if use debug mode")
parser.add_argument('--model_dir', default='experiments_websp/014', help="model directory")
parser.add_argument('--epoch_num', default=300, help="num of epoch") # sq:150->100, lcquad:600, sq:40
parser.add_argument('--batch_size', default=16, help="batch size") # sq:64, lcquad:16, sq:16
parser.add_argument('--max_len', default=300, help="max sequence length") # sq:300, lcquad:200, sq:300
parser.add_argument('--learning_rate', default=2e-5, help="learning rate")
parser.add_argument('--repeat_multi_sub', default=False, help="if to repeat multi subs instead of sampling")
# Ema
parser.add_argument('--ema', default=True, help="if use ema")
parser.add_argument('--ema_decay', default=0.99, help="")  # 0.999
# Ban
parser.add_argument('--ban', default=False, help="if use born again to train")
parser.add_argument('--teacher_only', default=False, help="if only use teacher outputs to train")
parser.add_argument('--teacher_ckpt_path', default='experiments/084/best.pth.tar', help="the teacher model path")
parser.add_argument('--do_train_and_eval', default=True, help="do_train_and_eval")
parser.add_argument('--do_eval', default=True, help="do_train_and_eval")
args = parser.parse_args()

# must be a new empty folder
if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)
elif args.do_train_and_eval:
    raise ValueError("dir exists")
else:
    pass

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
            # if name == 'bert.embeddings.word_embeddings.weight':
            #     print(param.data[0])
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
            # if name == 'bert.embeddings.word_embeddings.weight':
            #     print(param.data[0])
        self.backup = {}

def _ns_type(pre, obj):
    if 'ns#type' in pre:
        if obj[-1] == 's':
            obj = obj[:-1]
    return obj

def _post_process(tokens):
    post_process_dic = {'fwd. us': 'fwd.us', 'u. s.': 'u.s.', 'f. c.': 'f.c.', 'o. co': 'o.co', '8. 1': '8.1',
                        'd. c.': 'd.c.', 'l. p.': 'l.p.', 'u. n. i. t. y.': 'u.n.i.t.y.', 'outlook. com': 'outlook.com',
                        'g. s. s.': 'g.s.s.', 'google. by': 'google.by', 't. i.': 't.i.', 'r. e. p.': 'r.e.p.',
                        'c. d.': 'c.d', 'l. r.': 'l.r.', 'drop. io': 'drop.io', 'at & t': 'at&t'}
    # 'c + +', a-changin'(musical),john madden football'92
    for dic_element in post_process_dic.keys():
        if dic_element in tokens:
            tokens = tokens.replace(dic_element, post_process_dic[dic_element])

    post_process_dic2 = ['c + +', 'gtk +', '30 +']
    for dic_element2 in post_process_dic2:
        if dic_element2 in tokens:
            tokens = tokens.replace(' +', '+')

    if '\'(' in tokens:     # '(
        index = tokens.find('\'')
        if tokens[index + 1] == '(':
            tokens = tokens[:index + 1] + ' ' + tokens[index + 1:]

    return tokens


def train(model, data_iterator, optimizer, args, epoch, ema, model_teacher=None):
    """Train the model on `steps` batches"""
    model.train()
    loss_avg = utils.RunningAverage()
    t = trange(args.train_steps, desc='Train')
    for i in t:
        batch_sub_task, batch_obj_task = next(data_iterator)
        if args.ban and model_teacher:
            with torch.no_grad():
                teacher_logits = model_teacher(batch_sub_task, batch_obj_task, get_logits=True)
        else:
            teacher_logits = None
        loss = model(batch_sub_task, batch_obj_task, teacher_logits, teacher_only=args.teacher_only)
        if args.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu
        loss.backward()
        optimizer.step()
        if args.ema:
            ema.update()
        optimizer.zero_grad()
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.4f}'.format(loss_avg()))

def evaluate(model, data_loader, data, output_file_path):
    writer = open(output_file_path, 'w', encoding='utf-8')
    # todo scores
    # score_writer = open(os.path.join(args.model_dir, 'result_scores.json'), 'w', encoding='utf-8')

    data_iterator = data_loader.data_iterator(data, batch_size=1)  # can only batch size == 1
    model.eval()
    with torch.no_grad():
        num_pred, num_gold, num_correct = 1e-10, 1e-10, 1e-10
        num_correct_sent = 1e-10
        t = trange(len(data), desc='Eval')
        # print(len(data))
        for i in t:
            # fetch the next evaluation batch
            batch_tokens = next(data_iterator)
            triplet_spans = model.extract(batch_tokens)  # shape: (batch_size, max_len)
            tokens = data[i]['sub']['tokens']
            sent = data[i]['sent']

            sent_id = data[i]['id']

            gold_triplets = data[i]['gold_triplets']
            pred_triplets = set()
            pred_triplets_dup_scores = set()
            for tri in triplet_spans:
                #decoding
                # sub = ''.join(tokens[tri[0][0]: tri[0][1]]).replace('□', ' ').replace('##', '').lower().strip()

                sub = ''.join(tokens[tri[0][0]: tri[0][1]]).replace('##', '').\
                            replace(double_whitespace_placeholder, '  ').\
                            replace(whitespace_placeholder, ' ').\
                            replace(qmark_placeholder, '?x').replace(uri_placeholder, '?y').lower()    # for the first 3 dataset ?uri
                # sub = ' '.join(tokens[tri[0][0]: tri[0][1]]).replace('( ', '(').replace(' )', ')').\
                #     replace(' ,', ',').replace(' .', '.').replace(' ##', '').replace(' :', ':').\
                #     replace(' -', '-').replace('- ', '-').replace(' !', '!').replace(' \'', '\'').replace('\' ', '\'').\
                #     replace(' –', '–').replace('– ', '–').replace(' /', '/').replace('/ ', '/').\
                #     replace(qmark_placeholder, '?x').replace(uri_placeholder, '?uri').lower()

                # replace('. ', '.'), james f. o'brien, P. Elmo Futrell, Fr. Agnel Multipurpose, Louis D. Astorino
                # replace(' \'', '\''), "africa '70 (band)")

                sub = _post_process(sub)

                pre = data_loader.idx2pre[tri[1]]

                # obj = ''.join(tokens[tri[2][0]: tri[2][1]]).replace('□', ' ').replace('##', '').lower().strip()

                obj = ''.join(tokens[tri[2][0]: tri[2][1]]).replace('##', ''). \
                    replace(double_whitespace_placeholder, '  '). \
                    replace(whitespace_placeholder, ' '). \
                    replace(qmark_placeholder, '?x').replace(uri_placeholder, '?y').lower()  # for the first 3 dataset ?uri
                # obj = ' '.join(tokens[tri[2][0]: tri[2][1]]).replace('( ', '(').replace(' )', ')').\
                #     replace(' ,', ',').replace(' .', '.').replace(' ##', '').replace(' :', ':').\
                #     replace(' -', '-').replace('- ', '-').replace(' !', '!').replace(' \'', '\'').replace('\' ', '\'').\
                #     replace(' –', '–').replace('– ', '–').replace(' /', '/').replace('/ ', '/').\
                #     replace(qmark_placeholder, '?x').replace(uri_placeholder, '?uri').lower()

                obj = _post_process(obj)

                # obj = _ns_type(pre, obj)

                pred_triplets.add((sub, pre, obj))

                # todo add probs/scores
                # pre_score = tri[3]
                # pred_triplets_dup_scores.add((sub, pre, obj, pre_score))

            if gold_triplets == pred_triplets:   # gold_triplets, pred_triplets
                num_correct_sent += 1
            correct_triplets = pred_triplets.intersection(gold_triplets)
            num_pred += len(pred_triplets)
            num_gold += len(gold_triplets)
            num_correct += len(correct_triplets)

            _write_json(writer, sent, sent_id, pred_triplets)

            # todo scores
            # _write_json_dup_score(score_writer, sent, sent_id, pred_triplets_dup_scores)

            t.set_postfix(f1='{:05.4f}'.format(2.*num_correct/(num_pred+num_gold)))
        # logging loss, f1, acc and report
        precision, recall = num_correct/num_pred, num_correct/num_gold
        f1 = 2 * precision * recall / (precision + recall)
        macro_acc = num_correct_sent / len(data)   # compute the macro acc sparql

        metrics = OrderedDict()
        metrics['accuracy'] = macro_acc
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        metrics_str = "; ".join("{}: {:05.4f}".format(k, v) for k, v in metrics.items())
        logging.info("metrics: "+metrics_str)
    writer.close()
    return metrics

def _write_json(writer, text, sent_id, triplets):
    sample = {'text': text, 'id': sent_id}
    spo_list = []
    for triplet in triplets:
        spo_list.append({'subject': triplet[0], 'predicate': triplet[1], 'object': triplet[2], 'subject_type': None, 'object_type': None})
    sample['spo_list'] = spo_list
    sample['postag'] = None
    json.dump(sample, writer, ensure_ascii=False)
    writer.write('\n')

def _write_json_dup_score(writer, text, sent_id, triplets):
    sample = {'text': text, 'id': sent_id}
    spo_list = []
    for triplet in triplets:
        spo_list.append({'subject': triplet[0], 'predicate': triplet[1], 'object': triplet[2], 'predicate_score': triplet[3], 'subject_type': None, 'object_type': None})
    sample['spo_list'] = spo_list
    sample['postag'] = None
    json.dump(sample, writer, ensure_ascii=False)
    writer.write('\n')

def train_and_evaluate(model, data_loader, train_data, dev_data, optimizer, args, ema=None, model_teacher=None):
    """Train the model and evaluate every epoch."""
    best_val_f1 = 0.0
    best_val_acc = 0.0

    patience_counter = 0
    for epoch in range(1, args.epoch_num + 1):
        logging.info("Epoch {}/{}".format(epoch, args.epoch_num))
        train_data_iterator = data_loader.data_iterator(train_data, batch_size=args.batch_size, seed=args.seed+epoch, is_train=True, shuffle=True)
        train(model, train_data_iterator, optimizer, args, epoch, ema, model_teacher)
        if args.ema:
            ema.apply_shadow()
        # if not args.multi_gpu:
        if epoch >= args.epoch_num // 10 and epoch % 5 == 0:   # do eval after xx epochs, eval every x epoch
            val_metrics = evaluate(model, data_loader, dev_data, os.path.join(args.model_dir, 'dev_%d.json'%epoch))
            val_f1 = val_metrics['f1']
            val_acc = val_metrics['accuracy']

            improve_f1 = val_f1 - best_val_f1
            improve_acc = val_acc - best_val_acc

            # Save weights of the network
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            utils.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model_to_save.state_dict(),
                                'optim_dict': optimizer.state_dict()},
                                is_best=improve_acc>0,
                                # is_best=improve_f1,
                                checkpoint=args.model_dir)
            ## 1.
            if improve_acc > 0:
                logging.info("- Found new best acc")
                best_val_acc = val_acc
            ## 2.
            # if improve_f1 > 0:
            #     logging.info("- Found new best f1")
            #     best_val_f1 = val_f1

                if improve_acc < args.patience:
                    patience_counter += 1
                else:
                    patience_counter = 0
            else:
                patience_counter += 1
            # Early stopping and logging best f1
            if (patience_counter >= args.patience_num and epoch > args.min_epoch_num) or epoch == args.epoch_num:
                ## 1.
                logging.info("Best val acc: {:05.2f}".format(best_val_acc))
                ## 2.
                # logging.info("Best val f1: {:05.2f}".format(best_val_f1))
                break
        # else:
        #     # 多卡不支持train and evaluate，因为viterbi解码不是tensor计算
        #     model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        #     utils.save_checkpoint({'epoch': epoch + 1,
        #                         'state_dict': model_to_save.state_dict(),
        #                         'optim_dict': optimizer.state_dict()},
        #                         is_best=True,
        #                         checkpoint=args.model_dir)
        if args.ema:
            # restore backup weights
            ema.restore()

if __name__ == '__main__':
    # Use GPUs if available
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.n_gpu = torch.cuda.device_count()
    print("args.n_gpu:", args.n_gpu)
    args.multi_gpu = args.multi_gpu
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info("device: {}, n_gpu: {}".format(args.device, args.n_gpu))
    logging.info("Hyper params:%r"%args.__dict__)
    # Create the input data pipeline
    logging.info("Loading the datasets...")
    # Initialize the DataLoader
    data_loader = DataLoader(args)
    # Load training data and test data
    if not args.debug:
        data_names = ['train_re_pp', 'test_re_pp', 'test_re_pp'] # train_dev_2_part_verb_re_pp, test_2_re_pp, train_dev_re_pp
        # 'train_re_pp_l', 'test_re_pp_l', 'test_re_pp_l'
        # corrected rl new : 'train_dev_part_verb_re_pp', 'test_re_pp', 'test_re_pp'
    else:
        data_names = ['train_debug', 'dev', 'test']

    if args.ban and args.teacher_ckpt_path:
        logging.info("Loading teacher model from %s" % args.teacher_ckpt_path)
        config_path = os.path.join(args.bert_model_dir, 'bert_config.json')
        config = BertConfig.from_json_file(config_path)
        model_teacher = BertNTR(config, num_class=len(args.idx2pre)) # idx2pre should be exactly the same with the teacher model
        model_teacher.to(args.device)
        if args.n_gpu > 1 and args.multi_gpu:
            model_teacher = torch.nn.DataParallel(model_teacher)
        utils.load_checkpoint(args.teacher_ckpt_path, model_teacher)
        model_teacher.eval()
    else:
        model_teacher = None
    if args.do_train_and_eval:
        train_data = data_loader.load_data(data_names[0],
                                           max_len=args.max_len,
                                           repeat_multi_sub=args.repeat_multi_sub)
        dev_data = data_loader.load_data(data_names[1])
        # Prepare model
        model = BertNTR.from_pretrained(args.bert_model_dir, num_class=len(args.idx2pre))
        model.to(args.device)
        if args.n_gpu > 1 and args.multi_gpu:
            model = torch.nn.DataParallel(model)
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
                'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
                'weight_decay_rate': 0.0}
        ]
        if args.ema:
            # initialize EMA
            ema = EMA(model, args.ema_decay)
            # initialize shadow weights
            ema.register()
        else:
            ema = None
        args.total_steps = len(train_data) * args.epoch_num // args.batch_size
        args.train_steps = len(train_data) // args.batch_size
        optimizer = BertAdam(optimizer_grouped_parameters, 
                                lr=args.learning_rate, 
                                warmup=args.warmup, 
                                t_total=args.total_steps,
                                max_grad_norm=args.clip_grad,
                                schedule=args.schedule)
        # Train and evaluate the model
        logging.info("Starting training for {} epoch(s)".format(args.epoch_num))
        train_and_evaluate(model, data_loader, train_data, dev_data, optimizer, args, ema, model_teacher)

    ## -------------------------------------- 
    if args.do_eval:
        # Evaluate and predict dev set and test set
        logging.info("Loading best model...")
        # Define the model
        config_path = os.path.join(args.bert_model_dir, 'bert_config.json')
        config = BertConfig.from_json_file(config_path)
        model = BertNTR(config, num_class=len(args.idx2pre))
        model.to(args.device)
        # loading the best model
        utils.load_checkpoint(os.path.join(args.model_dir, 'best.pth.tar'), model)
        # inference test file
        test_data = data_loader.load_data(data_names[2])
        # test_data = data_loader.load_data('test')
        logging.info("Starting prediction with the best model...")
        evaluate(model, data_loader, test_data, output_file_path=os.path.join(args.model_dir, 'result.json'))

        # # inference final test file
        # test_data = data_loader.load_data('corrected_questions-verb-copy')
        # # test_data = data_loader.load_data('test')
        # logging.info("Starting prediction with the best model...")
        # evaluate(model, data_loader, test_data, output_file_path=os.path.join(args.model_dir, 'result_final.json'))
