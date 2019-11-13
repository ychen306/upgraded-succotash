'''
Sample a random expresison dag, input, and dump the I/O examples together with used instructions
'''
from collections import namedtuple, defaultdict
from intrinsic_types import intrinsic_types
from semas import semas
from specs import specs
import random
import z3
from z3_utils import askey, eval_z3_expr
from z3_exprs import *
import json

from synth_utils import get_usable_insts

InputType = namedtuple('InputType', ['bitwidth', 'is_constant'])

instantiated_insts = []
with open('instantiated-insts.json') as f:
  for inst, imm8 in json.load(f):
    instantiated_insts.append((inst, imm8))

def get_ctype_bitwidth(typename):
  if typename.endswith('*'):
    typename = typename[:-1].strip()
  return intrinsic_types[typename].bitwidth

def get_intrinsic_signature(spec):
  '''
  given spec, return ([<input type>], [<output size>])
  '''
  input_types = []
  output_sizes = []
  inst_form = spec.inst_form.split(', ')
  no_imm8 = 'imm8' not in (param.name for param in spec.params)
  for i, param in enumerate(spec.params):
    bitwidth = get_ctype_bitwidth(param.type)

    if ((no_imm8 and i < len(inst_form) and inst_form[i] == 'imm') or
        param.name == 'imm8'):
      input_types.append(InputType(bitwidth=bitwidth, is_constant=True))
      continue

    if param.type.endswith('*'):
      output_sizes.append(bitwidth)
    else:
      input_types.append(InputType(bitwidth=bitwidth, is_constant=False))
  if spec.rettype != 'void':
    out_bitwidth = get_ctype_bitwidth(spec.rettype)
    output_sizes = [out_bitwidth,] + output_sizes
  return tuple(input_types), tuple(output_sizes)

def categorize_by_output(sigs):
  '''
  given signature, which maps instruction -> signature
  return a map: <output sig> -> <instruction>]
  '''
  categories = defaultdict(list)
  for inst, (_, out_sig) in sigs.items():
    categories[out_sig].append(inst)
  return categories

# FIXME: make this run faster
def sample_expr_with_inputs(out_sig, rounds, sigs, semas, categories, live_ins):
  available_values = live_ins[:]

  vals2insts = {}

  # mapping inst -> sensible imm8s
  imm8s = defaultdict(list)
  for inst, imm8 in instantiated_insts:
    imm8s[inst].append(imm8)

  def sample_one_inst(out_sig):
    '''
    sample one instruction that returns `out_sig`
    '''
    usable_insts = get_usable_insts(categories[out_sig], sigs, available_values)
    if len(usable_insts) == 0:
      return []
    inst = random.choice(usable_insts)
    input_types, _ = sigs[inst]
    input_vals, output_vals = semas[inst]
    args = []

    selected_imm8 = None
    # sample arguments
    for ty, param in zip(input_types, input_vals):
      if ty.is_constant:
        # sample an immediate
        selected_imm8 = random.choice(imm8s[inst])
        arg = z3.BitVecVal(int(selected_imm8), param.size())
      else:
        # sample from available values
        arg = random.choice([v for v in available_values if v.size() == param.size()])
      args.append((param, arg))

    # evaluate inst with sampled arguments
    evaluated = []
    for out in output_vals:
      out_evaluated = eval_z3_expr(out, args)
      vals2insts[askey(out_evaluated)] = (inst, selected_imm8), args
      evaluated.append(out_evaluated)
    return evaluated

  for _ in range(rounds):
    for out_sig2 in categories:
      available_values.extend(sample_one_inst(out_sig2))
  
  used_insts = []
  visited_vals = set()
  def get_used_insts(val):
    val_key = askey(val)
    if val_key in visited_vals or val_key not in vals2insts:
      return
    visited_vals.add(val_key)
    inst, args = vals2insts[val_key]
    for _, arg in args:
      get_used_insts(arg)
    used_insts.append(inst)

  outs = sample_one_inst(out_sig)
  for out in outs:
    get_used_insts(out) 
  return outs[0], used_insts

sigs = {}
for inst, (in_vals, out_vals) in semas.items():
  if inst in specs:
    sigs[inst] = get_intrinsic_signature(specs[inst])
  else:
    input_types = tuple(InputType(x.size(), is_constant=False) for x in in_vals)
    output_sizes = tuple(y.size() for y in out_vals)
    sigs[inst] = input_types, output_sizes

categories = categorize_by_output(sigs)

def sample_expr(rounds):
  bitwidths = [8, 16, 32, 64, 128, 256, 512]
  num_inputs = int(random.gauss(4, 1)+1)
  inputs = []
  for i in range(num_inputs):
    bw = random.choice(bitwidths)
    inputs.append(z3.BitVec('x_%d_%d' % (i, bw), bw))
  out_size = random.choice(bitwidths)
  e, insts = sample_expr_with_inputs((out_size,), rounds, sigs, semas, categories, live_ins=inputs)
  return e, insts, inputs

def gen_expr(rounds=5):
  while True:
    try:
      e, insts, liveins = sample_expr(rounds)
    except:
      continue
    e = z3.simplify(e)
    if not (z3.is_bv_value(e) or
        z3.is_true(e) or
        z3.is_false(e) or
        get_z3_app(e) == z3.Z3_OP_UNINTERPRETED):
      return e, serialize_expr(e), insts, liveins

def gen_serialized_expr(*args):
  _, serialize_expr, insts, _ = gen_expr(rounds=5)
  return serialize_expr, insts

if __name__ == '__main__':
  from multiprocessing.pool import Pool
  from z3_exprs import serialize_expr
  from tqdm import tqdm
  import json

  pool = Pool(128)

  #target, _, insts, _ = gen_expr(instantiated_insts)
  #print(z3.simplify(target))
  #print(insts)
  #exit()

  num_exprs = 500000
  pbar = iter(tqdm(range(num_exprs)))

  with open('exprs.json', 'w') as outf:
    for e, insts in pool.imap_unordered(gen_serialized_expr, range(num_exprs)):
      outf.write(json.dumps((e, insts))+'\n')
      next(pbar)
