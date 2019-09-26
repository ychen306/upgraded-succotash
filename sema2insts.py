'''
Graph neural model(s) that takes a graph representation of SMT formula and predicts instructions that can implement the formula
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from expr_sampler import sample_expr, semas
from dgl.nn.pytorch.conv import GatedGraphConv, GINConv
from dgl import DGLGraph
from tqdm import tqdm

import z3
from z3_utils import get_z3_app

from z3_exprs import num_ops, num_bitwidths, max_num_params, serialize_expr

class OpEmb(nn.Module):
  '''
  (Op, Params) -> <vector>

  Params = expr.params() + (expr.size(),)
  '''
  def __init__(self, emb_size=128):
    super().__init__()
    self.emb_size = emb_size
    self.op_embs = nn.Embedding(num_ops, emb_size)
    self.param_embs = nn.Embedding(num_bitwidths, emb_size)
    self.mlp = nn.Sequential(
        nn.Linear(emb_size * (1 + max_num_params), emb_size),
        nn.ReLU(),
        nn.Linear(emb_size, emb_size))
  
  def forward(self, op, params):
    '''
    op : N x emb_size
    params : list of N x emb_size
    '''
    op_emb = self.op_embs(op)
    param_embs = [self.param_embs(p) for p in params]
    return self.mlp(torch.cat([op_emb] + param_embs, dim=1))

class Sema2Insts(nn.Module):
  def __init__(self, num_insts, emb_size=128, num_layers=3):
    super().__init__()
    self.num_insts = num_insts
    self.op_emb = OpEmb(emb_size)
    self.edge_func = nn.Sequential(
        nn.Linear(emb_size, emb_size),
        nn.ReLU()
        )
    self.conv_layers = nn.ModuleList()
    for _ in range(num_layers):
      self.conv_layers.append(
          GINConv(apply_func=nn.Sequential(
              nn.Linear(emb_size, emb_size),
              nn.ReLU()),
            aggregator_type='mean'))
    self.mlp = nn.Sequential(
        nn.Linear(emb_size * num_layers, emb_size),
        nn.ReLU(),
        nn.Linear(emb_size, emb_size),
        nn.ReLU(),
        nn.Linear(emb_size, emb_size),
        nn.ReLU(),
        nn.Linear(emb_size, num_insts))

  def forward(self, g, ops, params):
    states = self.op_emb(ops, params)
    conv_outs = []
    for conv in self.conv_layers:
      states = conv(g, states)
      conv_outs.append(states.sum(dim=0))
    return torch.sigmoid(self.mlp(torch.cat(conv_outs)))

def expr2graph(serialized_expr):
  '''
  <z3 expr> -> <graph>, <features>
  '''
  edges, ops, params = serialized_expr
  srcs, dsts = zip(*edges)
  g = DGLGraph()
  g.add_nodes(len(ops))
  g.add_edges(srcs, dsts)
  return g, torch.LongTensor(ops), [torch.LongTensor(p) for p in params]

def mean(xs):
  return sum(xs) / len(xs)

def validate(model, data):
  '''
  run model through data and get average precision and recall
  '''
  precisions = []
  recalls = []
  with torch.no_grad():
    for g, ops, params, inst_ids, _ in data:
      pred = model(g, ops, params)
      recall, prec = get_precision_and_recall(pred, inst_ids)
      recalls.append(recall)
      precisions.append(prec)
  return mean(precisions), mean(recalls)

def train(model, data, validator, batch_size=4, epochs=10):
  ids = list(range(len(data)))
  dl = DataLoader(ids, batch_size=batch_size, shuffle=True)
  criterion = nn.BCELoss(reduction='none')
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

  for epoch in range(epochs):
    pbar = tqdm(dl)
    for batch in pbar:
      preds = []
      targets = []
      recalls = []
      precisions = []
      weights = []
      for i in batch:
        g, ops, params, inst_ids, w = data[i]
        weights.append(w)
        pred = model(g, ops, params)
        preds.append(pred)
        recall, prec = get_precision_and_recall(pred, inst_ids)
        recalls.append(recall)
        precisions.append(prec)
        targets.append(inst_ids)
      preds = torch.stack(preds)
      targets = torch.stack(targets)
      loss = criterion(preds, targets).reshape(-1).dot(torch.cat(weights))
      pbar.set_description('loss: %.5f, recall: %.5f, precision: %.5f' % (loss, mean(recalls), mean(precisions)))

      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
      optimizer.step()

    torch.save(model.state_dict(), 'sema2insts.model')
    precision, recall = validator(model)
    print('Epoch:', epoch,
        'precision:', precision,
        'recall:', recall)

def unpack(ids, num_elems):
  x = torch.zeros(num_elems)
  for i in ids:
    x[i] = 1
  return x

insts2ids = { inst : i for i, inst in enumerate(semas) }
num_insts = len(insts2ids)

def load_data(expr_generator, n):
  '''
  expr_generator : () -> (<serialized expr>, <insts used to produce the expr>)
  '''
  data = []
  print('generating expressions')
  for _ in tqdm(range(n)):
    e, insts = expr_generator()
    g, ops, params = expr2graph(e)
    inst_ids = unpack([insts2ids[i] for i in insts], num_insts)
    weights = torch.tensor([num_insts/len(insts) if x==1 else 1 for x in inst_ids])
    weights.div_(weights.sum())
    data.append((g, ops, params, inst_ids, weights))
  return data

def get_precision_and_recall(predicted, actual):
  '''
  return recall and precision
  '''
  tp = ((predicted > 0.5) & (actual > 0.5)).sum()
  tn = ((predicted < 0.5) & (actual < 0.5)).sum()
  fp = ((predicted > 0.5) & (actual < 0.5)).sum()
  fn = ((predicted < 0.5) & (actual > 0.5)).sum()
  if int(tp + fn) == 0:
    recall = 1
  else:
    recall = float(tp) / float(tp + fn)
  if int(tp + fp) == 0:
    precision = 1
  else:
    precision = float(tp) / float(tp + fp)
  return recall, precision


def gen_expr():
  while True:
    e, insts = sample_expr(2)
    e = z3.simplify(e)
    if not (z3.is_bv_value(e) or
        z3.is_true(e) or
        z3.is_false(e) or
        get_z3_app(e) == z3.Z3_OP_UNINTERPRETED):
      return serialize_expr(e), insts

num_train = 1000000
num_test = 1000

num_train = 1000
num_test = 100

epochs = 20
gen_new_expr = False

if __name__ == '__main__':
  import json

  model = Sema2Insts(num_insts)

  # FIXME: check that we have enough expressions in `exprs.json'
  with open('exprs.json') as f:
    print('Loading serialized expressions')
    n = num_train + num_test
    pbar = iter(tqdm(range(n)))
    serialized_data = []
    for line in f:
      serialized_data.append(json.loads(line))
      next(pbar)
      if len(serialized_data) >= n:
        break
  
  load_one_expr = lambda : serialized_data.pop()
  data = load_data(load_one_expr, num_train)
  test_data = load_data(load_one_expr, num_test)
  validator = lambda model: validate(model, test_data)
  train(model, data, validator=validator, epochs=epochs)
