from gen_enumerator import emit_everything, make_fully_connected_graph, ConcreteInst
from expr_sampler import sigs, gen_expr
from tempfile import NamedTemporaryFile
import subprocess
from sema2insts import Sema2Insts, expr2graph, unpack
from tqdm import tqdm
from torch.distributions import Bernoulli, Multinomial
import torch
import torch.nn.functional as F
import torch.optim as optim
from io import StringIO
import multiprocessing

import requests
import json
import os

import z3
from z3_exprs import serialize_expr
from z3_utils import serialize_z3_expr

llvm_insts = [inst for inst in sigs.keys() if inst.startswith('llvm')]

job_queue = multiprocessing.Queue()
result_queue = multiprocessing.Queue()

def synthesize(insts, target, liveins, timeout=5):
  g, nodes = make_fully_connected_graph(
      liveins=liveins,
      insts=[ConcreteInst(inst, imm8=imm8) for inst, imm8 in insts],
      num_levels=4)
  with NamedTemporaryFile(mode='w', suffix='.c') as f, NamedTemporaryFile() as exe:
    emit_everything(target, g, nodes, f)
    f.flush()
    subprocess.check_output('cc %s insts.o -o %s -I. 2>/dev/null' % (f.name, exe.name), shell=True)
    p = subprocess.Popen(['timeout', str(timeout), exe.name], stdout=subprocess.PIPE)
    return p.stdout.read()

def check_synth_batched(insts_batch, target, liveins, timeout=5):
  batch_size = len(insts_batch)
  c_files = [NamedTemporaryFile(mode='w', suffix='.c') for _ in range(batch_size)]
  exe_files = [NamedTemporaryFile(delete=False) for _ in range(batch_size)]

  compilations = []
  for insts, f, exe in zip(insts_batch, c_files, exe_files):
    g, nodes = make_fully_connected_graph(
        liveins=liveins,
        insts=[ConcreteInst(inst, imm8=imm8) for inst, imm8 in insts],
        num_levels=4)
    try:
      emit_everything(target, g, nodes, f)
    except:
      compilations.append(None)
      continue
    f.flush()
    exe.close()
    compilations.append(subprocess.Popen('gcc %s insts.o -o %s -I. -no-pie 2>/dev/null' % (f.name, exe.name), shell=True))

  synth_jobs = []
  for compilation, exe in zip(compilations, exe_files):
    if compilation is None:
      synth_jobs.append(None)
      continue
    compilation.communicate()
    synth_jobs.append(subprocess.Popen(['timeout', str(timeout), exe.name], stdout=subprocess.PIPE))

  results = []
  for i, synth_job in enumerate(synth_jobs):
    if synth_job is None:
      results.append(0)
    else:
      out = synth_job.stdout.readline()
      synthesized = len(out) > 1
      results.append(1.0 if synthesized else 0.0)
      synth_job.kill()
    c_files[i].close()
    exe_files[i].close()
    if os.path.exists(exe_files[i].name):
      os.remove(exe_files[i].name)

  return results

class ServerSelector:
  def __init__(self, servers):
    self.servers = servers
    self.counter = 0

  def select(self):
    server = self.servers[self.counter]
    # bump the counter
    self.counter = (self.counter + 1) % len(self.servers)
    return server

def worker(job_queue, result_queue):
  while True:
    job = job_queue.get(True)
    job_id = job.pop('job_id')
    server = job.pop('server')
    resp = requests.post(server, data=json.dumps(job))
    result_queue.put((job_id, resp.json()))

if __name__ == '__main__':
  import sys

  config_f = sys.argv[1]

  with open(config_f) as f:
    config = json.load(f)
    servers = config['servers']
    num_threads = int(config['num_threads'])
    temp_dir = config['temp_dir']
    dir = config['dir']
  server_selector = ServerSelector(servers)

  pool = multiprocessing.Pool(len(servers), worker, (job_queue, result_queue))

  inst_pool = []
  with open('instantiated-insts.json') as f:
    for inst, imm8 in json.load(f):
      inst_pool.append((inst, imm8))

  print('Num insts:', len(inst_pool))

  epoch = 50000
  batch_size = 4
  max_insts = 20
  beta = 0.05
  num_rollouts = 760
  num_insts = len(inst_pool)
  insts2ids = { inst : i for i, inst in enumerate(inst_pool) }

  model = Sema2Insts(num_insts)
  try:
    model.load_state_dict(torch.load('synth.model'))
  except:
    print('Failed to reload model state')

  optimizer = optim.Adam(model.parameters())

  pbar = tqdm(list(range(epoch)))
  num_solved = 0
  losses = []
  for i in pbar:
    reqs = []

    target, target_serialized, _, liveins = gen_expr()
    liveins = [(x.decl().name(), x.size()) for x in liveins]
    g, g_inv, ops, params, _ = expr2graph(target_serialized)
    inst_ids = unpack([insts2ids[i] for i in inst_pool], num_insts)
    inst_probs = model(g, g_inv, ops, params).softmax(dim=0)
    inst_dist = Multinomial(max_insts, inst_probs)

    solved = False
    rollout_losses = []

    trials = 0
    inflight_jobs = []
    successes = 0
    job_to_log_probs = {}
    while trials < num_rollouts:
      log_probs = []
      insts_batch = []

      # sample as many samples as one server can handle
      for _ in range(num_threads):
        selected = torch.multinomial(inst_probs, max_insts, replacement=False)
        sample = torch.zeros(num_insts)
        sample[selected] = 1
        log_probs.append(inst_dist.log_prob(sample).sum())
        selected_insts = [inst_pool[i] for i in selected]
        insts_batch.append(selected_insts)

      trials += num_threads

      # enqueue a batch of enumeration job
      job_id = len(job_to_log_probs)
      job_to_log_probs[job_id] = torch.stack(log_probs)
      job = {
          'insts_batch': insts_batch,
          'liveins': liveins,
          'target': serialize_z3_expr(target),
          'timeout': str(5),
          'server': server_selector.select(),
          'job_id': job_id
          }
      job_queue.put(job)

    while len(job_to_log_probs) > 0:
      job_id, results = result_queue.get(True)
      log_prob = job_to_log_probs.pop(job_id)
      rollout_losses.append(-(log_prob * torch.FloatTensor(results)).mean())
      successes += sum(results)

    num_solved += int(successes > 0)
    loss = torch.stack(rollout_losses).mean()
    losses.append(loss)
    pbar.set_description('loss: %.4f, num_synthesized: %d/%d, # solved: %.4f' % (
      float(loss), int(successes), trials, num_solved/(i+1)))

    if len(losses) == batch_size:
      loss = torch.stack(losses).mean()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      losses = []

      torch.save(model.state_dict(), 'synth.model')
