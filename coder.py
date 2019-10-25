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
from collections import namedtuple, defaultdict
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

to_mask = lambda x: 1.0 if x is not None else 0.0

class InstSample:
  def __init__(self, max_args):
    self.inst = torch.tensor(0.0)
    self.args = [torch.tensor(0.0)] * max_args
    self.imm8 = torch.zeros(8)
    self.inst_mask = torch.zeros(1)
    self.arg_masks = [torch.zeros(1) for _ in range(max_args)]
    self.imm8_mask = torch.zeros(1)
  
  def set_inst(self, inst):
    self.inst = inst
    self.inst_mask = torch.ones(1)

  def set_arg(self, arg_idx, arg):
    self.args[arg_idx] = arg
    self.args[arg_idx] = torch.ones(1)

  def set_imm8(self, imm8):
    self.imm8 = imm8
    self.imm8 = troch.ones(1)

  def get_masks(self):
    return self.inst_mask, self.arg_masks, self.imm8_mask

class BatchedInstSample:
  def __init__(self, samples, max_args):
    self.inst_mask = torch.tensor([to_mask(s.inst) for s in samples])
    self.args_masks = [
        torch.tensor([to_mask(s.args[i]) for s in samples])
        for i in range(max_args)]
    self.imm8_mask = torch.tensor([to_mask(s.imm8) for s in samples])

    self.inst = torch.stack([s.inst.squeeze(0) if s.inst else torch.tensor(0.0) for s in samples])
    self.args = [
        torch.stack([s.args[i].squeeze(0) if s.args[i] else torch.tensor(0.0) for s in samples])
        for i in range(max_args)
        ]
    self.imm8 = torch.stack([s.imm8.squeeze(0) if s.imm8 is not None else torch.zeros(8) for s in samples])

  def get_masks(self):
    return self.inst_mask, self.args_masks, self.imm8_mask

def normalize(probs, mask):
  probs[mask].add_(1e-10)
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
        filtered_arg_probs[value_ids] = self.arg_probs[arg_pos].reshape(-1)[value_ids].detach()
        if len(value_ids) == 1:
          [value_id] = value_ids
        else:
          normalize(filtered_arg_probs, value_ids)
          arg_dist = Categorical(filtered_arg_probs)
          try:
            value_id = arg_dist.sample()
          except Exception as e:
            print(value_ids, filtered_arg_probs, self.arg_probs[arg_pos])
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
    inst_dist = Categorical(self.inst_probs)
    arg_dists = [Categorical(arg_probs) for arg_probs in self.arg_probs]
    imm8_dist = Bernoulli(self.imm8_bit_probs)
    assert inst_sample.inst is not None

    log_prob = 0
    log_prob += inst_dist.log_prob(inst_sample.inst)

    _, arg_masks, imm8_mask = inst_sample.get_masks()

    for arg, arg_dist, arg_mask in zip(inst_sample.args, arg_dists, arg_masks):
      try:
        log_prob += arg_dist.log_prob(arg) * arg_mask
      except:
        print('LOG_PROB FAILED:', arg.shape, arg_dist.probs.shape)
        exit(1)
    
    if len(inst_sample.imm8.shape) == 1:
      log_prob += imm8_dist.log_prob(inst_sample.imm8).sum() * imm8_mask
    else:
      log_prob += imm8_dist.log_prob(inst_sample.imm8).sum(dim=1) * imm8_mask

    return log_prob

  def entropy(self):
    inst_dist = Categorical(self.inst_probs)
    arg_dists = [Categorical(arg_probs) for arg_probs in self.arg_probs]
    imm8_dist = Bernoulli(self.imm8_bit_probs)
    return (inst_dist.entropy() + 
        sum(arg_dist.entropy() for arg_dist in arg_dists) +
          imm8_dist.entropy().sum(dim=1))

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
      log_probs.append(inst_dist.log_prob(inst_sample).squeeze(0))
      if equivalent(new_values[0], target, test_cases):
        # let reward/perf be the inverse of the instruction length
        return obs, inst_samples, 1/len(inst_samples), outs, torch.tensor(log_probs)
      values.extend(new_values)

    # synthesis failed
    return obs, inst_samples, 0, outs, torch.tensor(log_probs)

  def get_inst_dist(self, target, values):
    target_emb, *value_embs = self.encode_exprs([target] + values)
    return self.get_inst_dist_for_embs(target_emb, value_embs)

  def get_inst_dist_for_embs(self, target_emb, value_embs):
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
def get_loss_for_target(synthesizer, target_batch, obs_batch, insts_batch, orig_log_probs_batch, perf_batch):
  '''
  retrace a trajectory (of instructions) and get the trace for this (potentially hypothetical) target
  '''
  log_probs = []
  value_loss = []
  entropies = []

  episode_len = len(obs_batch[0])
  # shape: episode x tensor
  target_emb_batch = []
  # shape: episode x <num vals> x tensor
  value_embs_batch = []
  for i in range(episode_len):
    cur_target_embs = [] 
    num_values = len(obs_batch[0][i])
    cur_value_embs = [[] for _ in range(num_values)]
    for target, obs in zip(target_batch, obs_batch):
      values = obs[i]
      target_emb, *value_embs = synthesizer.encode_exprs([target] + values)
      cur_target_embs.append(target_emb)
      for j, value_emb in enumerate(value_embs):
        cur_value_embs[j].append(value_emb)
    target_emb_batch.append(torch.stack(cur_target_embs).squeeze(1))
    for j in range(num_values):
      cur_value_embs[j] = torch.stack(cur_value_embs[j]).squeeze(1)
    value_embs_batch.append(cur_value_embs)

  orig_log_probs = torch.stack(orig_log_probs_batch).t()
  assert orig_log_probs.shape[0] == episode_len

  insts_batch_new = []
  for i in range(episode_len):
    cur_insts = []
    for insts in insts_batch:
      cur_insts.append(insts[i])
    insts_batch_new.append(BatchedInstSample(cur_insts, max_args=synthesizer.max_args))
  insts_batch = insts_batch_new

  for target_emb, value_embs, inst_sample in zip(target_emb_batch, value_embs_batch, insts_batch):
    inst_dist = synthesizer.get_inst_dist_for_embs(target_emb, value_embs)
    log_probs.append(inst_dist.log_prob(inst_sample))
    entropies.append(inst_dist.entropy())

  log_probs = torch.stack(log_probs)
  assert log_probs.shape[0] == episode_len
  # imp sample weights = prodcut p(new_target)/p(orig_target)
  weights = torch.exp(
      torch.cumsum(log_probs, dim=1) -
      torch.cumsum(orig_log_probs, dim=1))
  weights = F.normalize(weights + 1e-12, p=1, dim=1).detach()
  assert weights.shape[0] == episode_len
  for i, perf in enumerate(perf_batch):
    weights[:, i].mul_(perf * 2)
  policy_loss = -(log_probs * Variable(weights)).sum()
  # we want higher entropy
  entropy_loss = -0.01 * torch.stack(entropies).mean(dim=1)
  return policy_loss + entropy_loss

def train(optimizer, synthesizer, steps=5, batch_size=32, epoch=50000):
  pbar = tqdm(range(epoch))
  for i in pbar:
    losses = []
    num_synthesized = 0
    target, inputs = gen_synth_problem()
    batch = defaultdict(list)
    num_episodes = 0
    for _ in range(batch_size):
      obs, insts, perf, outs, orig_log_probs = synthesizer.synthesize(target, steps, inputs)
      batch[len(insts)].append((target, obs, insts, orig_log_probs, 1/len(insts)))
      num_episodes += 1
      for i, result in enumerate(outs):
        if not is_trival_expression(result):
          # train with hindsight
          batch[i+1].append((result, obs[:i+1], insts[:i+1], orig_log_probs[:i+1], i/(i+1)))
          num_episodes += 1
      if perf > 0:
        num_synthesized += 1

    loss = 0
    for episodes in batch.values():
      target_batch = []
      obs_batch = []
      insts_batch = []
      orig_log_probs_batch = []
      perf_batch = []
      for target, obs, insts, orig_log_probs, perf in episodes:
        target_batch.append(target)
        obs_batch.append(obs)
        insts_batch.append(insts)
        orig_log_probs_batch.append(orig_log_probs)
        perf_batch.append(perf)

      loss += get_loss_for_target(synthesizer,
          target_batch, obs_batch, insts_batch,
          orig_log_probs_batch, perf_batch).sum()

    loss = loss / num_episodes
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
    print('Failed reload model state')
  optimizer = optim.Adam(synthesizer.parameters(), lr=5e-5)
  train(optimizer, synthesizer)
