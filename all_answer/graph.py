import torch
import networkx as nx
import json
from tqdm import tqdm
import numpy as np
import pickle
from scipy.sparse import coo_matrix
from multiprocessing import Pool
from collections import OrderedDict

from utils.conceptnet import merged_relations
from transformers import RobertaTokenizer, RobertaForMaskedLM
import argparse
import os


concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_cpnet(cpnet_graph_path):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def concepts2adj(node_ids):
    global id2relation
    cids = np.array(node_ids, dtype=np.int32)
    n_rel = len(id2relation)
    n_node = cids.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet.has_edge(s_c, t_c):
                for e_attr in cpnet[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    # cids += 1  # note!!! index 0 is reserved for padding
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cids


class RobertaForMaskedLMwithLoss(RobertaForMaskedLM):

    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None):
        assert attention_mask is not None
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0]  # hidden_states of final layer (batch_size, sequence_length, hidden_size)
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
            # (masked_lm_loss), prediction_scores, sequence_output, (hidden_states), (attentions)
        return outputs


print('loading pre-trained LM...')
TOKENIZER = RobertaTokenizer.from_pretrained('roberta-large')
LM_MODEL = RobertaForMaskedLMwithLoss.from_pretrained('roberta-large')
LM_MODEL.cuda()
LM_MODEL.eval()
print('loading done')


def get_LM_score(cids, question):
    cids = cids[:]
    cids.insert(0, -1)  # QAcontext node
    sents, scores = [], []
    for cid in cids:
        if cid == -1:
            sent = question.lower()
        else:
            sent = '{} {}.'.format(question.lower(), ' '.join(id2concept[cid].split('_')))
        sent = TOKENIZER.encode(sent, add_special_tokens=True)
        sents.append(sent)
    n_cids = len(cids)
    cur_idx = 0
    batch_size = 50
    while cur_idx < n_cids:
        # Prepare batch
        input_ids = sents[cur_idx: cur_idx+batch_size]
        max_len = max([len(seq) for seq in input_ids])
        for j, seq in enumerate(input_ids):
            seq += [TOKENIZER.pad_token_id] * (max_len-len(seq))
            input_ids[j] = seq
        input_ids = torch.tensor(input_ids).cuda()  # [B, seqlen]
        mask = (input_ids != 1).long()  # [B, seq_len]
        # Get LM score
        with torch.no_grad():
            outputs = LM_MODEL(input_ids, attention_mask=mask, masked_lm_labels=input_ids)
            loss = outputs[0]  # [B, ]
            _scores = list(-loss.detach().cpu().numpy())  # list of float
        scores += _scores
        cur_idx += batch_size
    assert len(sents) == len(scores) == len(cids)
    cid2score = OrderedDict(sorted(list(zip(cids, scores)), key=lambda x: -x[1]))  # score: from high to low
    return cid2score


def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1(data):
    qc_ids, ac_ids, question = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    return (sorted(qc_ids), sorted(ac_ids), question, sorted(extra_nodes))


def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(data):
    qc_ids, ac_ids, question, extra_nodes = data
    cid2score = get_LM_score(qc_ids+ac_ids+extra_nodes, question)
    return (qc_ids, ac_ids, question, extra_nodes, cid2score)


def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part3(data):
    qc_ids, ac_ids, question, extra_nodes, cid2score = data
    schema_graph = qc_ids + ac_ids + sorted(extra_nodes, key=lambda x: -cid2score[x])  # score: from high to low
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return {'adj': adj, 'concepts': concepts, 'qmask': qmask, 'amask': amask, 'cid2score': cid2score}


def generate_adj_data_from_grounded_concepts__use_LM(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
        (5) cid2score that maps a concept id to its relevance score given the QA context
    to the output path in python pickle format
        (6) ac_alloc: boolean array indicating which answer choice the `ac` belongs to; shape (n_choice, n_node)

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    print(f'generating adj data for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    qa_data = []
    ac_alloc = []

    statement_path = grounded_path.replace('grounded', 'statement')
    with open(grounded_path, 'r', encoding='utf-8') as fin_ground, open(statement_path, 'r', encoding='utf-8') as fin_state:
        lines_ground = fin_ground.readlines()
        lines_state = fin_state.readlines()
        assert len(lines_ground) % len(lines_state) == 0
        n_choices = len(lines_ground) // len(lines_state)

        for j, line in enumerate(lines_state):
            # for each question
            QAcontext = json.loads(line)['question']['stem']
            q_ids, a_ids, choice_ids = set(), set(), []

            for i in range(n_choices):
                # for each answer choice
                dic = json.loads(lines_ground[i + j*n_choices])

                QAcontext += " {}".format(dic['ans'])
                q_ids |= set([concept2id[c] for c in dic['qc']])
                tmp = set([concept2id[c] for c in dic['ac']])
                a_ids |= tmp

                choice_ids.append(tmp)

            q_ids = q_ids - a_ids
            qa_data.append((q_ids, a_ids, QAcontext))
            ac_alloc.append(choice_ids)

    with Pool(num_processes) as p:
        res1 = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1, qa_data), total=len(qa_data)))

    res2 = []
    for j, _data in enumerate(res1):
        if j % 100 == 0:
            print(j)
        res2.append(concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(_data))

    with Pool(num_processes) as p:
        res3 = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair__use_LM__Part3, res2), total=len(res2)))

    assert len(res3) == len(ac_alloc)
    for j, _data in enumerate(res3):
        choice_ids = ac_alloc[j]  # list of set
        concepts = _data['concepts']
        mask = np.zeros((len(choice_ids), concepts.shape[0]), dtype=np.bool)
        for i in range(len(choice_ids)):
            ids = np.array(sorted(choice_ids[i]))
            index = (concepts.reshape(-1, 1) == ids).nonzero()[0]
            mask[i, index] = True
        assert np.sum(mask.any(axis=0)) == _data['amask'].sum()
        _data['ac_alloc'] = mask

    # res is a list of responses
    with open(output_path, 'wb') as fout:
        pickle.dump(res3, fout)

    print(f'adj data saved to {output_path}')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--grounded_path', type=str, required=True)
    parser.add_argument('--output_root', type=str, default='./data')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--cpnet_graph_path', type=str, default='cpnet/conceptnet.en.pruned.graph')
    parser.add_argument('--cpnet_vocab_path', type=str, default='cpnet/concept.txt')
    parser.add_argument('-p', '--num_processes', type=int, default=16)
    args = parser.parse_args()

    if 'AMLT_DATA_DIR' in os.environ:
        args.data_root = os.environ['AMLT_DATA_DIR']
        args.output_root = os.environ['AMLT_OUTPUT_DIR']

    generate_adj_data_from_grounded_concepts__use_LM(os.path.join(args.data_root, args.grounded_path),
                                                     os.path.join(args.data_root, args.cpnet_graph_path),
                                                     os.path.join(args.data_root, args.cpnet_vocab_path),
                                                     os.path.join(args.output_root, args.output_path),
                                                     args.num_processes)
