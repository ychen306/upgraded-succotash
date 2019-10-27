import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.distributions import Categorical, Bernoulli
from expr_sampler import semas, sigs, get_usable_insts, categories, sample_expr
from sema2insts import SemaEmb, insts2ids, ids2insts, expr2graph, num_insts
from z3_exprs import serialize_expr
import z3
from z3_utils import *
from functools import reduce
from tqdm import tqdm
from collections import namedtuple
import random

from synth_utils import get_usable_insts

class InstCoder(nn.Module):
  def __init__(self, num_insts, max_args, emb_size):
    super().__init__()
    self.arg_embs = nn.ModuleList()
    for _ in range(max_args):
      self.arg_embs.append(nn.Linear(emb_size, emb_size))
    self.emit_inst = nn.Linear(emb_size, num_insts)
    self.emb_size = emb_size
    # <state x target> -> <action>
    self.emit_action = nn.Sequential(
        nn.Linear(emb_size + emb_size, emb_size),
        nn.LeakyReLU(),
        nn.Linear(emb_size, emb_size),
        nn.LeakyReLU(),
        nn.Linear(emb_size, emb_size))
    self.emit_imm8 = nn.Sequential(
        nn.Linear(emb_size, emb_size),
        nn.LeakyReLU(),
        nn.Linear(emb_size, 8))

  def forward(self, target, value_embs):
    state = torch.stack(value_embs, dim=1).sum(dim=1)
    obs = torch.cat([state, target], dim=1)
    action_emb = self.emit_action(obs)
    inst_prob = torch.softmax(self.emit_inst(action_emb), dim=1)
    arg_probs = [
        torch.softmax(
          # select/favor values "similar" to our arg embedding
          torch.stack(value_embs, dim=1).bmm(arg_fn(action_emb).unsqueeze(2)).squeeze(2), dim=1)
        for arg_fn in self.arg_embs]
    imm8_bit_probs = torch.sigmoid(self.emit_imm8(action_emb))
    return inst_prob, arg_probs, imm8_bit_probs

class InstPool:
  def __init__(self, semas, sigs):
    self.semas = semas
    self.sigs = sigs
    self.insts = list(semas.keys())

def gen_test_inputs(input_vars, num_test_cases=20):
  test_cases = []
  for _ in range(num_test_cases):
    test_case = []
    for x in input_vars:
      val = z3.BitVecVal(random.randint(0, 1<<x.size()), x.size())
      test_case.append((x, val))
    test_cases.append(test_case)
  return test_cases

class InstSample:
  def __init__(self, max_args):
    self.inst = None
    self.args = [None] * max_args
    self.imm8 = None
  
  def set_inst(self, inst):
    self.inst = inst

  def set_arg(self, arg_idx, arg):
    self.args[arg_idx] = arg

  def set_imm8(self, imm8):
    self.imm8 = imm8

def normalize(probs, mask):
  if probs[mask].sum() < 1e-12:
    probs[mask] = 1.0
  denorm = probs.sum()
  probs.div_(denorm)

class InstDist:
  def __init__(self, inst_probs, arg_probs, imm8_bit_probs, max_args):
    self.inst_probs = inst_probs
    self.arg_probs = arg_probs
    self.imm8_bit_probs = imm8_bit_probs
    self.max_args = max_args

  def sample(self, inst_pool, values):
    '''
    () -> <value>, <inst sample>
    '''
    usable_insts = get_usable_insts(inst_pool.insts, sigs, values)
    usable_inst_ids = torch.LongTensor([insts2ids[inst] for inst in usable_insts])

    sample = InstSample(self.max_args)

    # mask off nonusable insts
    filtered_inst_probs = torch.zeros(self.inst_probs.shape).reshape(-1)
    filtered_inst_probs[usable_inst_ids] = self.inst_probs.reshape(-1)[usable_inst_ids].detach()
    normalize(filtered_inst_probs, usable_inst_ids)
    if len(usable_inst_ids) == 1:
      [inst_id] = usable_inst_ids
    else:
      inst_dist = Categorical(filtered_inst_probs)
      # sample the instruction
      try:
        inst_id = inst_dist.sample()
      except:
        print('BAD INST PROBS:', self.inst_probs.reshape(-1)[usable_inst_ids], len(usable_inst_ids))
        exit(1)

    sample.set_inst(inst_id)
    inst = ids2insts[inst_id.item()]

    input_vals, output_vals = inst_pool.semas[inst]
    input_types, _ = inst_pool.sigs[inst]
    arg_pos = 0
    # sample the arguments
    args = []

    new_values = []
    for ty, param in zip(input_types, input_vals):
      if ty.is_constant:
        # sample from the immediate distribution
        imm8_dist = Bernoulli(self.imm8_bit_probs.reshape(-1))
        bits = imm8_dist.sample()
        # the first bit is the higher order bit, ...
        imm8 = reduce(lambda a,b:a|b, (int(b) << i for i, b in enumerate(bits.long())))
        arg = z3.BitVecVal(imm8, param.size())
        sample.set_imm8(bits)
      else:
        # sample from available values
        value_ids = torch.LongTensor([i for i, v in enumerate(values) if v.size() == param.size()])
        assert len(value_ids) > 0
        filtered_arg_probs = torch.zeros(self.arg_probs[arg_pos].shape).reshape(-1)
        filtered_arg_probs[value_ids] = self.arg_probs[arg_pos].reshape(-1)[value_ids]
        if len(value_ids) == 1:
          [value_id] = value_ids
        else:
          normalize(filtered_arg_probs, value_ids)
          arg_dist = Categorical(filtered_arg_probs)
          try:
            value_id = arg_dist.sample()
          except:
            print('BAD ARG PROBS:', self.arg_probs[arg_pos].reshape(-1)[value_ids], len(value_ids))
            exit(1)
        arg = values[value_id]
        sample.set_arg(arg_pos, value_id)
        arg_pos += 1
      args.append((param,arg))

    # instantiate/evaluate the full instruction
    for out in output_vals:
      out_evaluated = eval_z3_expr(out, args)
      new_values.append(out_evaluated)

    return new_values, sample

  def log_prob(self, inst_sample):
    inst_dist = Categorical(self.inst_probs.reshape(-1))
    arg_dists = [Categorical(arg_probs.reshape(-1)) for arg_probs in self.arg_probs]
    imm8_dist = Bernoulli(self.imm8_bit_probs.reshape(-1))
    assert inst_sample.inst is not None

    log_prob = 0
    log_prob += inst_dist.log_prob(inst_sample.inst)

    for arg, arg_dist in zip(inst_sample.args, arg_dists):
      if arg is not None:
        log_prob += arg_dist.log_prob(arg)
    
    if inst_sample.imm8 is not None:
      log_prob += imm8_dist.log_prob(inst_sample.imm8).sum()

    return log_prob

  def entropy(self):
    inst_dist = Categorical(self.inst_probs.reshape(-1))
    arg_dists = [Categorical(arg_probs.reshape(-1)) for arg_probs in self.arg_probs]
    imm8_dist = Bernoulli(self.imm8_bit_probs.reshape(-1))
    return (inst_dist.entropy() +
        sum(arg_dist.entropy() for arg_dist in arg_dists) +
        imm8_dist.entropy().sum())


class Synthesizer(nn.Module):
  def __init__(self, inst_pool, max_args=4, emb_size=128):
    super().__init__()
    self.inst_coder = InstCoder(len(inst_pool.insts), max_args, emb_size)
    self.sema2vec = SemaEmb(emb_size=emb_size)
    self.inst_pool = inst_pool
    self.max_args = max_args

  def encode_exprs(self, exprs):
    g, g_inv, ops, params, expr_ids = expr2graph(serialize_expr(*exprs))
    _, expr_embs = self.sema2vec(g, g_inv, ops, params)
    return [emb.unsqueeze(0) for emb in expr_embs[expr_ids]]

  def synthesize(self, target, steps, liveins):
    values = liveins[:]
    inst_samples = []
    log_probs = []
    obs = []
    test_cases = gen_test_inputs(liveins)
    outs = []
    for _ in range(steps):
      obs.append(values[:])
      inst_dist = self.get_inst_dist(target, values)
      new_values, inst_sample = inst_dist.sample(self.inst_pool, values)
      outs.append(new_values[0])
      inst_samples.append(inst_sample)
      log_probs.append(inst_dist.log_prob(inst_sample))
      if equivalent(new_values[0], target, test_cases):
        # let reward/perf be the inverse of the instruction length
        return obs, inst_samples, 1/len(inst_samples), outs, torch.tensor(log_probs)
      values.extend(new_values)

    # synthesis failed
    return obs, inst_samples, 0, outs, torch.tensor(log_probs)

  def get_inst_dist(self, target, values):
    target_emb, *value_embs = self.encode_exprs([target] + values)
    inst_probs, arg_probs, imm8_bit_probs = self.inst_coder(target_emb, value_embs)
    return InstDist(inst_probs, arg_probs, imm8_bit_probs, max_args=self.max_args)


def is_trival_expression(e):
  return (z3.is_bv_value(e) or
      z3.is_true(e) or
      z3.is_false(e) or
      get_z3_app(e) == z3.Z3_OP_UNINTERPRETED)

def gen_synth_problem():
  while True:
    try:
      e, _, inputs = sample_expr(3)
    except: # not in the mood to fix this shit
      continue
    e = z3.simplify(e)
    if not is_trival_expression(e):
      return e, inputs

# FIXME: vectorize this
def get_loss_for_target(synthesizer, target, obs, insts, orig_log_probs, perf):
  '''
  retrace a trajectory (of instructions) and get the trace for this (potentially hypothetical) target
  '''

  perf *= 2

  log_probs = []
  baselines = []
  value_loss = []
  entropies = [] 
  for values, inst_sample in zip(obs, insts):
    inst_dist = synthesizer.get_inst_dist(target, values)
    log_probs.append(inst_dist.log_prob(inst_sample))
    entropies.append(inst_dist.entropy())
  log_probs = torch.stack(log_probs).reshape(-1)
  # imp sampl. weights = prodcut p(new_target)/p(orig_target)
  weights = torch.exp(
      torch.cumsum(log_probs, dim=0) -
      torch.cumsum(orig_log_probs, dim=0))
  weights.div_(weights.sum()+1e-12)
  policy_loss = (-log_probs * weights.detach() * perf).sum()
  # we want higher entropy
  entropy_loss = -0.01 * torch.stack(entropies).mean()
  return policy_loss + entropy_loss

def train(optimizer, synthesizer, steps=5, batch_size=32, epoch=50000):
  pbar = tqdm(range(epoch))
  for i in pbar:
    losses = []
    num_synthesized = 0
    target, inputs = gen_synth_problem()
    for _ in range(batch_size):
      obs, insts, perf, outs, orig_log_probs = synthesizer.synthesize(target, steps, inputs)
      losses.append(get_loss_for_target(synthesizer, target, obs, insts, orig_log_probs, perf))
      if perf > 0:
        num_synthesized += 1
      for i, result in enumerate(outs):
        if not is_trival_expression(result):
          losses.append(
              get_loss_for_target(synthesizer, result, 
                obs[:i+1], insts[:i+1], orig_log_probs[:i+1], 1/(i+1)))

    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pbar.set_description("loss: %.4f, num synth'd: %d/%d" % (float(loss), num_synthesized, batch_size))
    if i % 5 == 0:
      torch.save(synthesizer.state_dict(), 'synth.model')

if __name__ == '__main__':
  '''small tests to make sure shapes match up'''

  synthesizer = Synthesizer(InstPool(sigs=sigs, semas=semas))
  try:
    synthesizer.load_state_dict(torch.load('synth.model'))
  except:
    print('Failed to reload model state')
  optimizer = optim.Adam(synthesizer.parameters(), lr=5e-5)
  train(optimizer, synthesizer)
