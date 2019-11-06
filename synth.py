from gen_enumerator import emit_everything, make_fully_connected_graph, ConcreteInst
from expr_sampler import sigs, gen_expr
from tempfile import NamedTemporaryFile
import subprocess
from sema2insts import Sema2Insts, expr2graph, unpack, insts2ids
from tqdm import tqdm
from torch.distributions import Bernoulli, Multinomial
import torch
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing

import z3
from z3_exprs import serialize_expr

llvm_insts = [inst for inst in sigs.keys() if inst.startswith('llvm')]

def synthesize(insts, target, liveins, timeout=5):
  g, nodes = make_fully_connected_graph(
      liveins=liveins,
      insts=[ConcreteInst(inst, imm8=None) for inst in insts],
      num_levels=4)
  with NamedTemporaryFile(mode='w', suffix='.c') as f, NamedTemporaryFile() as exe:
    emit_everything(target, g, nodes, f)
    f.flush()
    subprocess.check_output('cc %s insts.o -o %s -I. 2>/dev/null' % (f.name, exe.name), shell=True)
    p = subprocess.Popen(['timeout', str(timeout), exe.name], stdout=subprocess.PIPE)
    return p.stdout.read()

def check_synth_batched(insts_batch, target, liveins):
  batch_size = len(insts_batch)
  c_files = [NamedTemporaryFile(mode='w', suffix='.c') for _ in range(batch_size)]
  exe_files = [NamedTemporaryFile() for _ in range(batch_size)]

  compilations = []
  for insts, f, exe in zip(insts_batch, c_files, exe_files):
    g, nodes = make_fully_connected_graph(
        liveins=liveins,
        insts=[ConcreteInst(inst, imm8=None) for inst in insts],
        num_levels=4)
    try:
      emit_everything(target, g, nodes, f)
    except:
      compilations.append(None)
      continue
    f.flush()
    compilations.append(subprocess.Popen('cc %s insts.o -o %s -I. 2>/dev/null' % (f.name, exe.name), shell=True))

  synth_jobs = []
  for compilation, exe in zip(compilations, exe_files):
    if compilation is None:
      synth_jobs.append(None)
      continue
    compilation.communicate()
    synth_jobs.append(subprocess.Popen(['timeout', '5', exe.name], stdout=subprocess.PIPE))

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

  return torch.tensor(results)

if __name__ == '__main__':
  epoch = 50000
  batch_size = 4
  num_threads = 16
  max_insts = 20
  beta = 0.05
  num_rollouts = 64
  num_insts = len(llvm_insts)

  num_rollouts = 12
  num_threads = 12
  batch_size = 1

  model = Sema2Insts(num_insts)
  try:
    model.load_state_dict(torch.load('synth.model'))
  except:
    print('Failed to reload model state')

  #optimizer = optim.Adam(model.parameters(), lr=2.5e-5)
  optimizer = optim.Adam(model.parameters())

  pbar = tqdm(list(range(epoch)))
  num_solved = 0
  losses = []
  for i in pbar:
    #target, target_serialized, _, liveins = gen_expr()

    x, y = z3.BitVecs('x y', 64)
    liveins = [x,y]
    target = z3.If(x >= y, x, y)  * y
    target_serialized = serialize_expr(target)


    liveins = [(x.decl().name(), x.size()) for x in liveins]
    g, g_inv, ops, params, _ = expr2graph(target_serialized)
    inst_ids = unpack([insts2ids[i] for i in llvm_insts], num_insts)
    inst_probs = model(g, g_inv, ops, params).softmax(dim=0)
    inst_dist = Multinomial(max_insts, inst_probs)
    insts_batch = []
    log_probs = []
    solved = False
    rollout_losses = []
    for _ in range(num_rollouts//num_threads):
      for _ in range(num_threads):
        selected = torch.multinomial(inst_probs, max_insts, replacement=False)
        sample = torch.zeros(num_insts)
        sample[selected] = 1
        log_probs.append(inst_dist.log_prob(sample).sum())
        selected_insts = [llvm_insts[i] for i, selected in enumerate(sample) if selected > 0]
        insts_batch.append(selected_insts)
      results = check_synth_batched(insts_batch, target, liveins)
      rollout_losses.append(-(torch.stack(log_probs) * results).mean())
      solved = solved or results.sum() > 0
    num_solved += int(solved)
    loss = torch.stack(rollout_losses).mean()
    losses.append(loss)
    pbar.set_description('loss: %.4f, num_syntheized: %d/%d, # solved: %.4f' % (
      float(loss), int(results.sum()), num_rollouts, num_solved/(i+1)))

    if len(losses) == batch_size:
      loss = torch.stack(losses).mean()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      losses = []

      torch.save(model.state_dict(), 'synth.model')
