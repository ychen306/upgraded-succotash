'''
Graph neural model(s) that takes a graph representation of SMT formula and predicts instructions that can implement the formula
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.distributions import Multinomial
from expr_sampler import sample_expr, semas
from dgl.nn.pytorch.conv import GatedGraphConv, GINConv
from dgl import DGLGraph
from tqdm import tqdm
import json

import z3
from z3_utils import get_z3_app

from z3_exprs import num_ops, max_num_params, serialize_expr, max_bitwidth

class OpEmb(nn.Module):
  '''
  (Op, Params) -> <vector>

  Params = expr.params() + (expr.size(),)
  '''
  def __init__(self, emb_size=128):
    super().__init__()
    self.emb_size = emb_size
    self.op_embs = nn.Embedding(num_ops, emb_size)
    self.param_embs = nn.Embedding(max_bitwidth, emb_size)
    self.mlp = nn.Sequential(
        nn.Linear(emb_size * (1 + max_num_params), emb_size),
        nn.LeakyReLU(),
        nn.Linear(emb_size, emb_size))
  
  def forward(self, op, params):
    '''
    op : N x emb_size
    params : list of N x emb_size
    '''
    op_emb = self.op_embs(op)
    param_embs = [self.param_embs(p) for p in params]
    return self.mlp(torch.cat([op_emb] + param_embs, dim=1))

class SemaEmb(nn.Module):
  def __init__(self, emb_size=128, num_layers=3):
    super().__init__()
    self.op_emb = OpEmb(emb_size)
    self.edge_func = nn.Sequential(
        nn.Linear(emb_size, emb_size),
        nn.LeakyReLU()
        )
    self.conv_layers = nn.ModuleList()
    for _ in range(num_layers):
      self.conv_layers.append(
          GINConv(apply_func=nn.Sequential(
              nn.Linear(emb_size, emb_size),
              nn.LeakyReLU()),
            aggregator_type='mean'))
    self.mlp = nn.Sequential(
        nn.Linear(emb_size * num_layers, emb_size),
        nn.LeakyReLU(),
        nn.Linear(emb_size, emb_size))
    self.mlp_inv = nn.Sequential(
        nn.Linear(emb_size * num_layers, emb_size),
        nn.LeakyReLU(),
        nn.Linear(emb_size, emb_size))
    self.combine = nn.Linear(emb_size * 2, emb_size)

  def forward(self, g, g_inv, ops, params):
    states = self.op_emb(ops, params)
    states_inv = self.op_emb(ops, params)
    conv_outs = []
    conv_outs_inv = []
    for conv in self.conv_layers:
      states = conv(g, states)
      conv_outs.append(states.sum(dim=0))
    for conv in self.conv_layers:
      states_inv = conv(g_inv, states_inv)
      conv_outs_inv.append(states.sum(dim=0))
    g_emb = self.mlp(torch.cat(conv_outs))
    g_inv_emb = self.mlp_inv(torch.cat(conv_outs_inv))
    return self.combine(torch.cat([g_emb, g_inv_emb])), self.combine(torch.cat([states, states_inv], dim=1))

class Sema2Insts(nn.Module):
  def __init__(self, num_insts, emb_size=128, num_layers=3):
    super().__init__()
    self.sema_emb = SemaEmb(emb_size, num_layers)
    self.mlp = nn.Sequential(
        nn.Linear(emb_size, emb_size),
        nn.LeakyReLU(),
        nn.Linear(emb_size, num_insts))

  def forward(self, g, g_inv, ops, params):
    # drop the node embeddings
    emb, _ = self.sema_emb(g, g_inv, ops, params)
    return self.mlp(emb)

def expr2graph(serialized_expr):
  '''
  <z3 expr> -> <graph>, <features>
  '''
  edges, ops, params, expr_ids = serialized_expr
  srcs, dsts = zip(*edges)
  g = DGLGraph()
  g.add_nodes(len(ops))
  g.add_edges(srcs, dsts)

  g_inv = DGLGraph()
  g_inv.add_nodes(len(ops))
  g_inv.add_edges(dsts, srcs)

  return g, g_inv, torch.LongTensor(ops), [torch.LongTensor(p) for p in params], torch.LongTensor(expr_ids)

def mean(xs):
  return float(sum(xs)) / float(len(xs))

def validate(model, data):
  '''
  run model through data and get average precision and recall
  '''
  recalls = []
  with torch.no_grad():
    for g, g_inv, ops, params, inst_ids in data:
      pred = model(g, g_inv, ops, params)
      recall = get_recalled(pred.softmax(dim=0), inst_ids)
      recalls.append(recall)
  return mean(recalls)

def train(model, data, validator, num_insts, batch_size=4, epochs=10):
  ids = list(range(len(data)))
  dl = DataLoader(ids, batch_size=batch_size, shuffle=True)
  #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  optimizer = optim.Adam(model.parameters(), lr=1e-4)

  for epoch in range(epochs):
    pbar = tqdm(dl)
    for batch in pbar:
      recalls = []
      log_probs = []
      for i in batch:
        g, g_inv, ops, params, inst_ids = data[i]
        logits = model(g, g_inv, ops, params)
        probs = logits.softmax(dim=0)
        inst_dist = Multinomial(int(inst_ids.sum()), probs)
        log_probs.append(inst_dist.log_prob(inst_ids.float()))
        sample = torch.zeros(num_insts)
        recalled = get_recalled(probs, inst_ids)
        recalls.append(recalled)

      loss = -torch.stack(log_probs).mean()
      pbar.set_description('loss: %.5f, recall: %.5f' % (loss, mean(recalls)))

      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
      optimizer.step()

    torch.save(model.state_dict(), 'sema2insts.model')
    recall = validator(model)
    print('Epoch:', epoch, 'recall:', recall)

def unpack(ids, num_elems):
  x = torch.zeros(num_elems)
  for i in ids:
    x[i] = 1
  return x

inst_pool = []
with open('instantiated-insts.json') as f:
  for inst, imm8 in json.load(f):
    inst_pool.append((inst, imm8))

num_insts = len(inst_pool)
insts2ids = { inst : i for i, inst in enumerate(inst_pool) }

def load_data(expr_generator, n):
  '''
  expr_generator : () -> (<serialized expr>, <insts used to produce the expr>)
  '''
  data = []
  print('generating expressions')
  for _ in range(n):
    e, insts = expr_generator()
    g, g_inv, ops, params, _ = expr2graph(e)
    inst_ids = unpack([insts2ids[inst,imm8] for inst, imm8 in insts], num_insts)
    data.append((g, g_inv, ops, params, inst_ids.long()))
  return data

def get_recalled(predicted, actual):
  _, top20 = predicted.topk(30)
  return actual[top20].sum() / actual.sum()

def sema2insts(model, e, accept=lambda p: p > 0.5):
  g, g_inv, ops, params = expr2graph(serialize_expr(e))
  pred = model(g, ops, params)
  return [(ids2insts[i], float(p)) for i, p in enumerate(pred.reshape(-1)) if accept(p)]

num_train = 400000
num_test = 1000

#num_train = 1000
#num_test = 100

epochs = 20
gen_new_expr = False

if __name__ == '__main__':
  import json

  model = Sema2Insts(num_insts)

  # FIXME: check that we have enough expressions in `exprs.json'
  with open('exprs.json') as f:
    print('Loading serialized expressions')
    n = num_train + num_test
    serialized_data = []
    for line in f:
      serialized_data.append(json.loads(line))
      if len(serialized_data) >= n:
        break
  
  load_one_expr = lambda : serialized_data.pop()
  data = load_data(load_one_expr, num_train)
  test_data = load_data(load_one_expr, num_test)
  validator = lambda model: validate(model, test_data)
  train(model, data, num_insts=num_insts, validator=validator, epochs=epochs)
