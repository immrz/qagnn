import random
import argparse
import time

try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

from utils.optimization_utils import OPTIMIZER_CLASSES
from utils.parser_utils import get_parser
from utils.utils import bool_flag, check_path, export_config, freeze_net, unfreeze_net, pretty_args
from utils.wrapper_utils import dataset_wrapper, model_wrapper
from utils.data_utils import load_statement_dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

import socket
import os
import subprocess
import datetime


DECODER_DEFAULT_LR = {
    'csqa': 1e-3,
    'obqa': 3e-4,
}


print(socket.gethostname())
print("pid:", os.getpid())
print("screen: %s" % subprocess.check_output('echo $STY', shell=True).decode('utf'))
print("gpu: %s" % subprocess.check_output('echo $CUDA_VISIBLE_DEVICES', shell=True).decode('utf'))


def evaluate_accuracy(eval_set, model):
    n_samples, n_correct = 0, 0
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for qids, labels, *input_data in eval_set:
            logits, _ = model(*input_data)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)

            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    loss = nn.functional.cross_entropy(all_logits, all_labels, reduction='mean')

    return n_correct / n_samples, loss.item()


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'eval_detail'], help='run training or evaluation')
    parser.add_argument('--save_dir', default='./saved_models/qagnn/', help='model output directory')
    parser.add_argument('--save_model', default=False, const=True, type=bool_flag, nargs='?', help='whether to save model')
    parser.add_argument('--save_test_preds', default=True, const=True, type=bool_flag, nargs='?')
    parser.add_argument('--save_best_model_only', type=bool_flag, default=True, const=True, nargs='?',
                        help='only save best dev acc model instead of each epoch')
    parser.add_argument('--load_model_path', default=None)

    # data
    parser.add_argument('--num_relation', default=38, type=int, help='number of relations')
    parser.add_argument('--train_adj', default=f'data/{args.dataset}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'data/{args.dataset}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'data/{args.dataset}/graph/test.graph.adj.pk')
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?',
                        const=True, help='use cached data to accelerate data loading')

    # model architecture
    parser.add_argument('-k', '--k', default=5, type=int, help='perform k-layer message passing')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
    parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?',
                        const=True, help='freeze entity embedding layer')
    parser.add_argument('--lm_as_edge_encoder', action='store_true')

    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')

    # regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args = parser.parse_args()
    if args.simple:
        parser.set_defaults(k=1)
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval_detail':
        # raise NotImplementedError
        eval_detail(args)
    else:
        raise ValueError('Invalid mode')


def train(args):
    print(pretty_args(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step,train_acc,dev_acc,test_acc,train_loss,dev_loss,test_loss\n')

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)

    if torch.cuda.device_count() >= 2 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
    elif torch.cuda.device_count() == 1 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")

    dataset = dataset_wrapper(args, (device0, device1))

    ###################################################################################################
    #   Build model                                                                                   #
    ###################################################################################################
    model = model_wrapper(args, (device0, device1), cp_emb)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        print(f'Max steps for linear scheduler is {max_steps}')
        try:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)

    print('parameters:')
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print('\ttotal:', num_params)

    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    print()
    print('-' * 71)
    global_step, best_dev_epoch = 0, 0
    best_dev_acc, final_test_acc, total_loss = 0.0, 0.0, 0.0
    start_time = time.time()
    model.train()
    freeze_net(model.encoder)

    for epoch_id in range(args.n_epochs):
        train_logits, train_labels, train_acc = [], [], 0.0

        if epoch_id == args.unfreeze_epoch:
            unfreeze_net(model.encoder)
        if epoch_id == args.refreeze_epoch:
            freeze_net(model.encoder)
        model.train()

        # for each batch
        for qids, labels, *input_data in dataset.train():
            optimizer.zero_grad()
            bs = labels.size(0)
            train_labels.append(labels)

            # for each mini batch
            for a in range(0, bs, args.mini_batch_size):
                b = min(a + args.mini_batch_size, bs)
                logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)

                # save logits
                train_logits.append(logits.detach().clone())

                # compute loss
                if args.loss == 'margin_rank':
                    num_choice = logits.size(1)
                    flat_logits = logits.view(-1)
                    correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)  # of length batch_size*num_choice
                    # length: batch_size*(num_choice-1)
                    correct_logits = flat_logits[correct_mask == 1].contiguous() \
                                                                   .view(-1, 1) \
                                                                   .expand(-1, num_choice - 1) \
                                                                   .contiguous() \
                                                                   .view(-1)

                    wrong_logits = flat_logits[correct_mask == 0]
                    y = wrong_logits.new_ones((wrong_logits.size(0),))
                    loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss

                elif args.loss == 'cross_entropy':
                    loss = loss_func(logits, labels[a:b])

                loss = loss * (b - a) / bs
                loss.backward()  # gradient accumulation
                total_loss += loss.item()

            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # batch logging
            if (global_step + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                print('| step {:5} |  lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(
                    global_step, scheduler.get_lr()[0], total_loss, ms_per_batch))
                total_loss = 0
                start_time = time.time()
            global_step += 1

        # compute train accuracy and loss
        train_logits = torch.cat(train_logits, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        train_acc = (train_logits.argmax(1) == train_labels).sum().item() / train_labels.size(0)
        train_loss = loss_func(train_logits, train_labels).item()  # NOTE: assume CE loss is used

        # compute dev and test accuracy
        model.eval()
        dev_acc, dev_loss = evaluate_accuracy(dataset.dev(), model)
        if not args.save_test_preds:
            test_acc, test_loss = evaluate_accuracy(dataset.test(), model) if args.test_statements else 0.0
        else:
            eval_set = dataset.test()
            total_acc = []
            all_logits, all_labels = [], []
            count = 0
            preds_path = os.path.join(args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
            with open(preds_path, 'w') as f_preds:
                with torch.no_grad():
                    for qids, labels, *input_data in eval_set:
                        count += 1
                        logits, _, concept_ids, node_type_ids, edge_index, edge_type = model(*input_data, detail=True)
                        all_logits.append(logits)
                        all_labels.append(labels)
                        predictions = logits.argmax(1)  # [bsize, ]
                        preds_ranked = (-logits).argsort(1)  # [bsize, n_choices]
                        for i, (qid, label, pred, _preds_ranked, cids, ntype, edges, etype) in \
                                enumerate(zip(qids, labels, predictions, preds_ranked,
                                              concept_ids, node_type_ids, edge_index, edge_type)):
                            acc = int(pred.item() == label.item())
                            print('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
                            f_preds.flush()
                            total_acc.append(acc)
            test_acc = float(sum(total_acc)) / len(total_acc)
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            test_loss = loss_func(all_logits, all_labels).item()  # NOTE: assume CE loss is used

        # epoch logging and save accuracy
        print('-' * 125)
        print('| epoch {:3} | train_acc {:7.4f} | dev_acc {:7.4f} | test_acc {:7.4f} | train_loss {:7.4f} | dev_loss {:7.4f} | test_loss {:7.4f} |'.format(
            epoch_id, train_acc, dev_acc, test_acc, train_loss, dev_loss, test_loss))
        print('-' * 125)
        with open(log_path, 'a') as fout:
            fout.write('{},{},{},{},{},{},{}\n'.format(global_step, train_acc, dev_acc, test_acc,
                                                       train_loss, dev_loss, test_loss))

        if dev_acc >= best_dev_acc:
            best_dev_acc = dev_acc
            final_test_acc = test_acc
            best_dev_epoch = epoch_id

            # save best dev acc model here
            if args.save_model:
                torch.save([model, args], model_path)
                with open(model_path + ".log.txt", 'w') as f:
                    for p in model.named_parameters():
                        print(p, file=f)
                print(f'Best model saved to {model_path} at epoch {epoch_id}')

        # save model per epoch
        if args.save_model and (not args.save_best_model_only):
            torch.save([model, args], model_path + ".{}".format(epoch_id))
            with open(model_path + ".{}.log.txt".format(epoch_id), 'w') as f:
                for p in model.named_parameters():
                    print(p, file=f)
            print(f'Epoch model saved to {model_path}')

        model.train()
        start_time = time.time()
        if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
            break

    print(f'Final test accuracy is {final_test_acc}')
    # except (KeyboardInterrupt, RuntimeError) as e:
    #     print(e)


def eval_detail(args):
    assert args.load_model_path is not None
    model_path = args.load_model_path
    model, old_args = torch.load(model_path)
    if torch.cuda.device_count() >= 2 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
    elif torch.cuda.device_count() == 1 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
    model.encoder.to(device0)
    model.decoder.to(device1)
    model.eval()

    statement_dic = {}
    for statement_path in (args.train_statements, args.dev_statements, args.test_statements):
        statement_dic.update(load_statement_dict(statement_path))

    use_contextualized = 'lm' in old_args.ent_emb

    print('inhouse?', args.inhouse)

    print('args.train_statements', args.train_statements)
    print('args.dev_statements', args.dev_statements)
    print('args.test_statements', args.test_statements)
    print('args.train_adj', args.train_adj)
    print('args.dev_adj', args.dev_adj)
    print('args.test_adj', args.test_adj)

    args.encoder = old_args.encoder
    args.max_node_num = old_args.max_node_num
    args.max_seq_len = old_args.max_seq_len
    dataset = dataset_wrapper(args, (device0, device1))

    save_test_preds = args.save_model
    if not save_test_preds:
        test_acc, _ = evaluate_accuracy(dataset.test(), model) if args.test_statements else 0.0
    else:
        eval_set = dataset.test()
        total_acc = []
        count = 0
        dt = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
        preds_path = os.path.join(args.save_dir, 'test_preds_{}.csv'.format(dt))
        with open(preds_path, 'w') as f_preds:
            with torch.no_grad():
                for qids, labels, *input_data in tqdm(eval_set):
                    count += 1
                    logits, _, concept_ids, node_type_ids, edge_index, edge_type = model(*input_data, detail=True)
                    predictions = logits.argmax(1)  # [bsize, ]
                    preds_ranked = (-logits).argsort(1)  # [bsize, n_choices]
                    for i, (qid, label, pred, _preds_ranked, cids, ntype, edges, etype) in enumerate(zip(qids, labels, predictions, preds_ranked, concept_ids, node_type_ids, edge_index, edge_type)):
                        acc = int(pred.item()==label.item())
                        print('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
                        f_preds.flush()
                        total_acc.append(acc)
        test_acc = float(sum(total_acc))/len(total_acc)

        print('-' * 71)
        print('test_acc {:7.4f}'.format(test_acc))
        print('-' * 71)


if __name__ == '__main__':
    main()
