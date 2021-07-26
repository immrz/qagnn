import json
import os
import time
import argparse

from modeling.modeling_qagnn import LM_QAGNN, LM_QAGNN_DataLoader
from multi_view.modeling_qagnn import Multiview_LM_QAGNN, Multiview_LM_QAGNN_DataLoader
from all_answer.modeling_qagnn import AllAns_LM_QAGNN, AllAns_LM_QAGNN_DataLoader


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def check_file(file):
    return os.path.isfile(file)


def export_config(config, path):
    param_dict = dict(vars(config))
    check_path(path)
    with open(path, 'w') as fout:
        json.dump(param_dict, fout, indent=4)


def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True


def test_data_loader_ms_per_batch(data_loader, max_steps=10000):
    start = time.time()
    n_batch = sum(1 for batch, _ in zip(data_loader, range(max_steps)))
    return (time.time() - start) * 1000 / n_batch


def pretty_args(args):
    """
    Return a nice string repr of the argparse.Namespace object.
    """
    s = 'Args:\n'
    for name, val in vars(args).items():
        s += f'{name.replace("_", " ").capitalize()}: {val}\n'
    return s + '\n'


def dataset_wrapper(args, device):
    """An interface for dataset construction according to the given args.
    """

    dataset_class = LM_QAGNN_DataLoader
    dataset_args = [args,
                    args.train_statements,
                    args.train_adj,
                    args.dev_statements,
                    args.dev_adj,
                    args.test_statements,
                    args.test_adj]
    dataset_kwargs = {'batch_size': args.batch_size,
                      'eval_batch_size': args.eval_batch_size,
                      'device': device,
                      'model_name': args.encoder,
                      'max_node_num': args.max_node_num,
                      'max_seq_length': args.max_seq_len,
                      'is_inhouse': args.inhouse,
                      'inhouse_train_qids_path': args.inhouse_train_qids,
                      'subsample': args.subsample,
                      'use_cache': args.use_cache}

    # zero or exactly one option
    assert sum([args.views is not None, args.all_ans]) <= 1

    if args.views is not None:
        dataset_class = Multiview_LM_QAGNN_DataLoader
        dataset_kwargs.update({'num_view': args.num_view,
                               'num_mask_view': args.num_mask_view,
                               'mask_view_prob': args.mask_view_prob,
                               'view_only_train': args.view_only_train,
                               'view_shuffle': 'shuffle' in args.views})

    elif args.all_ans:
        dataset_class = AllAns_LM_QAGNN_DataLoader

    return dataset_class(*dataset_args, **dataset_kwargs)


def model_wrapper(args, device, cp_emb):
    """An interface for model construction according to the given args.
    """

    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))

    model_class = LM_QAGNN
    model_args = [args, args.encoder]
    model_kwargs = {'k': args.k,
                    'n_ntype': 4,
                    'n_etype': args.num_relation,
                    'n_concept': concept_num,
                    'concept_dim': args.gnn_dim,
                    'concept_in_dim': concept_dim,
                    'n_attention_head': args.attn_head_num,
                    'fc_dim': args.fc_dim,
                    'n_fc_layer': args.fc_layer_num,
                    'p_emb': args.dropouti,
                    'p_gnn': args.dropoutg,
                    'p_fc': args.dropoutf,
                    'pretrained_concept_emb': cp_emb,
                    'freeze_ent_emb': args.freeze_ent_emb,
                    'init_range': args.init_range,
                    'encoder_config': {}}

    # zero or exactly one option
    assert sum([args.views is not None, args.all_ans]) <= 1

    if args.views is not None:
        model_class = Multiview_LM_QAGNN
        model_kwargs.update({'views': args.views})

    elif args.all_ans:
        model_class = AllAns_LM_QAGNN

    return model_class(*model_args, **model_kwargs)
