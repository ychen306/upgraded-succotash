from collections import defaultdict, namedtuple
import z3
from io import StringIO
from z3_utils import *
from expr_sampler import semas, sigs, get_usable_insts, categories, sample_expr
from tqdm import tqdm

counter = 0
def gen_temp():
  global counter
  counter += 1
  return 'tmp-sygus%d' % counter

VarDecl = namedtuple('VarDecl', ['var', 'type', 'defn'])

def z3expr_to_str(e, quantified_vars):
  # mapping expr -> var, type
  decls = {}
  topo_order = []
  def translate_expr(e):
    key = askey(e)
    if key in decls:
      return decls[key]
    
    args = e.children()
    if len(args) == 0:
      if e.decl().kind() == z3.Z3_OP_UNINTERPRETED and key not in quantified_vars:
        # replace unquantified (i.e. impliticly universally quantified) vars
        # with default
        e = get_default_val(e)
      decl = VarDecl(gen_temp(), e.sort(), e.sexpr())
      decls[key] = decl
      topo_order.append(key)
      return decl

    translated_args = [translate_expr(arg).var for arg in args]
    op = get_z3_app(e)
    if op in z3op_names:
      op_name = z3op_names[op]
    else:
      if op not in (z3.Z3_OP_EXTRACT, z3.Z3_OP_SIGN_EXT, z3.Z3_OP_ZERO_EXT):
        print('WTF???', e.decl())
      assert op in (z3.Z3_OP_EXTRACT, z3.Z3_OP_SIGN_EXT, z3.Z3_OP_ZERO_EXT)
      params = ' '.join(str(p) for p in e.params())
      op_name = '(_ %s %s)' % ({
          z3.Z3_OP_EXTRACT: 'extract',
          z3.Z3_OP_SIGN_EXT: 'sign_extend',
          z3.Z3_OP_ZERO_EXT: 'zero_extend',
          }[op], params)
    decl = VarDecl(gen_temp(), e.sort(), '(%s %s)' % (op_name, ' '.join(translated_args)))
    topo_order.append(key)
    decls[key] = decl
    return decl

  translate_expr(e)
  buf = StringIO()
  for var in topo_order:
    decl = decls[var] 
    buf.write('(let ((%s %s %s))\n' % (decl.var, decl.type.sexpr(), decl.defn))
  buf.write(decls[topo_order[-1]].var)
  buf.write(')' * len(topo_order))
  buf.write('\n')
  s = buf.getvalue()
  buf.close()
  return s

def fix_imm(bitwidth, imm8):
  # extend the bitwidth of an immediate in case the intrinsic screws up
  # e.g. saying the type of 'imm8' is int
  if bitwidth > 8:
    return '(concat #x%s imm8)' % ('0' * ((bitwidth-8)//4))
  return imm8

# output a "synth-fun" for a function that returns `out_sig`
def gen_one_inst(name, out_sig, sigs, categories, available_values, outf):
  '''
  available_values : [<name>, <bit-width>]
  '''
  # mapping <bitwidth> -> [<val>]
  bw2vals = defaultdict(list)
  for val, bw in available_values:
    bw2vals[bw].append(val)

  outf.write('(synth-fun %s' % name)
  outf.write('(') # open arg list
  for val, bitwidth in available_values:
    outf.write('(%s (BitVec %d))' % (val, bitwidth))
  # every instruction also gets to use a constant
  outf.write('(imm8 (BitVec 8))')
  outf.write(')') # end arg list

  # ret type
  out_bitwidth = sum(out_sig)
  outf.write(' (BitVec %d) ' % out_bitwidth)

  # now declare the syntax
  outf.write('\n((Start (BitVec %d) ( ' % out_bitwidth)
  # declare available instructions
  for inst in categories[out_sig]:
    in_types, _ = sigs[inst]
    # we can only use this instruction if we can supply legal arguments
    if any(x.bitwidth not in bw2vals and not x.is_constant for x in in_types):
      continue
    arg_list = ' '.join(
      ('Arg%d' % x.bitwidth) if not x.is_constant else fix_imm(x.bitwidth, 'imm8')
      for x in in_types)
    outf.write('(%s %s)' % (inst, arg_list))
  outf.write('))') # close Start

  # declare available arguments
  for bw, vals in bw2vals.items():
    outf.write('\n(Arg%d (BitVec %d) (%s))' % (bw, bw, ' '.join(vals)))

  outf.write(')') # close syntax
  outf.write(')\n') # close synth-fun


def can_use_inst(inst, sigs, available_bitwidths):
  inputs, _ = sigs[inst]
  return all(x.bitwidth in available_bitwidths for x in inputs)

def get_default_val(x):
  if z3.is_bv(x):
    return z3.BitVecVal(0, x.size())
  if z3.is_bool(x):
    return z3.BitVecVal(False)
  assert False, "unsupported type"

def define_func(inst, sema, outf):
  inputs, outputs = sema
  outf.write('(define-fun %s (' % inst)

  # declare inputs
  for x in inputs:
    outf.write('(%s (BitVec %d))' % (x.decl().name(), x.size()))
  outf.write(')') # close papram. list

  # declare output type
  outf.write(' (BitVec %d)\n' % sum(y.size() for y in outputs))


  # define the func body
  if len(outputs) == 1:
    out = outputs[0]
  else:
    out = z3.Concat(outputs)
  # some of our semantics contain universally quantified free vars,
  # assuming the semantics is correct, we replace these variables
  # with some default values.
  outf.write(z3expr_to_str(out, set(map(askey, inputs))))

  outf.write(')\n')

def gen_immediate(outf):
  '''
  ask the solver to synthesize one constant
  '''
  imm = gen_temp()
  outf.write("(synth-fun %s () (BitVec 8))\n" % imm)
  return imm

# FIXME: infer live_ins directly from target's semantics
def gen_sygus(target, round, out_sig, sigs, semas, categories, live_ins, outf):
  outf.write('(set-logic BV)\n')

  # TODO: do this offline
  for inst, sema in semas.items():
    define_func(inst, sema, outf)
  
  available_values = live_ins[:]
  insts_declared = 0
  out_sigs = [] 
  available_values_per_inst = []
  immediates = []
  for _ in range(round):
    available_bitwidths = set(bw for _, bw in available_values)
    for out_sig2, insts in categories.items():
      if any(can_use_inst(inst, sigs, available_bitwidths) for inst in insts):
        available_values_per_inst.append(available_values[:])
        immediates.append(gen_immediate(outf))
        gen_one_inst('i%d' % insts_declared, out_sig2, sigs, categories, available_values, outf)
        for i, bw in enumerate(out_sig2):
          available_values.append(('r%d_%d' % (insts_declared, i), bw))
        out_sigs.append(out_sig2)
        insts_declared += 1

  immediates.append(gen_immediate(outf))
  out_sigs.append(out_sig)
  available_values_per_inst.append(available_values)
  gen_one_inst('i%d' % insts_declared, out_sig, sigs, categories, available_values, outf)

  # generate the wrapper
  outf.write('(define-fun soln (')
  # inputs
  for x, bw in live_ins:
    outf.write('(%s (BitVec %d))' % (x, bw))
  outf.write(')') # close param list
  # output
  outf.write(' (BitVec %d)\n' % sum(out_sig))
  # chain all the instructions together
  for i, (sig, imm) in enumerate(zip(out_sigs, immediates)):
    avail = available_values_per_inst[i]
    avail.append((imm, 8))
    call_inst = '(i%d %s)' % (i, ' '.join(x for x, _ in avail))
    inst_out = gen_temp()
    outf.write('(let ((%s (BitVec %d) %s))\n' % (inst_out, sum(sig), call_inst))
    # unpack the outputs
    outf.write('(let (')
    for j, bw in enumerate(sig):
      outf.write('(r%d_%d (BitVec %d) ((_ extract %d %d) %s))\n' % (
        i, j, bw, sum(sig[:j+1])-1, sum(sig[:j]), inst_out))
    outf.write(')\n') # close decl list
  outf.write(inst_out)
  outf.write('))' * len(out_sigs)) # close all the lets
  outf.write(')\n') # close define-fun

  # declare synthesis contraints
  define_func('target', target, outf)
  # declare the live-ins
  for x, bw in live_ins:
    outf.write('(declare-var %s (BitVec %d))\n' % (x, bw))
  outf.write('(constraint (= (soln {0}) (target {0})))\n'.format(' '.join(x for x, _ in live_ins)))
  outf.write('(check-synth)\n')

if __name__ == '__main__':
  import sys
  import torch
  
  from sema2insts import sema2insts, Sema2Insts, num_insts, semas
  
  model = Sema2Insts(num_insts)
  model.load_state_dict(torch.load('sema2insts.model'))
  model.eval()
  
  sema = semas['_mm512_mask_shuffle_i32x4'][1][0]
  with open('t.sy', 'w') as outf:
    while True:
      expr, used_insts, inputs = sample_expr(1)
      expr = z3.simplify(expr)
      if not (z3.is_bv_value(expr) or
          z3.is_true(expr) or
          z3.is_false(expr) or
          get_z3_app(expr) == z3.Z3_OP_UNINTERPRETED):
        break
    live_ins = [(x.decl().name(), x.size()) for x in inputs]
    possible_insts = set(inst for inst, _ in sema2insts(model, expr)).union(set(used_insts))
    print('Num possible insts:', len(possible_insts))
    filtered_semas = {inst: sema for inst, sema in semas.items() if inst in possible_insts}
    assert len(filtered_semas) == len(possible_insts)
    filtered_sigs = {inst: sig for inst, sig in sigs.items() if inst in possible_insts}
    filtered_cats = {}
    for sig, insts in categories.items():
      filtered_cats[sig] = [inst for inst in insts if inst in possible_insts]
    target_sema = inputs, (expr,)
    gen_sygus(target_sema, 1, (expr.size(),),
        filtered_sigs, filtered_semas, filtered_cats, live_ins, outf)
