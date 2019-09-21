from collections import defaultdict, namedtuple
from semas import semas
import z3
from io import StringIO
from z3_utils import *

counter = 0
def gen_temp():
  global counter
  counter += 1
  return 'tmp%d' % counter

VarDecl = namedtuple('VarDecl', ['var', 'type', 'defn'])

def z3expr_to_str(e):
  # mapping expr -> var, type
  decls = {}
  topo_order = []
  def translate_expr(e):
    key = askey(e)
    if key in decls:
      return decls[key]
    
    args = e.children()
    if len(args) == 0:
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

# output a "synth-fun" for a function that returns `out_sig`
def gen_one_inst(name, out_sig, semas, categories, available_values, outf):
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
  outf.write(')') # end arg list

  # ret type
  out_bitwidth = sum(out_sig)
  outf.write(' (BitVec %d) ' % out_bitwidth)

  # now declare the syntax
  outf.write('\n((Start (BitVec %d) ( ' % out_bitwidth)
  # declare available instructions
  for inst in categories[out_sig]:
    args, _ = semas[inst]
    # we can only use this instruction if we can supply legal arguments
    if any(arg.size() not in bw2vals for arg in args):
      continue
    arg_list = ' '.join('Arg%d' % arg.size() for arg in args)
    outf.write('(%s %s)' % (inst, arg_list))
  outf.write('))') # close Start

  # declare available arguments
  for bw, vals in bw2vals.items():
    outf.write('\n(Arg%d (BitVec %d) (%s))' % (bw, bw, ' '.join(vals)))

  outf.write(')') # close syntax
  outf.write(')\n') # close synth-fun


def can_use_inst(inst, semas, available_bitwidths):
  inputs, _ = semas[inst]
  return all(x.size() in available_bitwidths for x in inputs)

def define_func(inst, semas, outf):
  inputs, outputs = semas
  outf.write('(define-fun %s (' % inst)

  # declare inputs
  for x in inputs:
    outf.write('(%s (BitVec %d))' % (x.decl().name(), x.size()))
  outf.write(')') # close papram. list

  # declare output type
  outf.write(' (BitVec %d)\n' % sum(y.size() for y in outputs))

  # define the func body
  if len(outputs) == 1:
    outf.write(z3expr_to_str(outputs[0]))
  else:
    outf.write(z3expr_to_str(z3.Concat(outputs)))

  outf.write(')\n')

def gen_sygus(round, out_sig, semas, categories, live_ins, outf):
  for inst, sema in semas.items():
    define_func(inst, sema, outf)
  
  available_values = live_ins[:]
  insts_declared = 0
  sigs = [] 
  available_values_per_inst = []
  for _ in range(round):
    available_bitwidths = set(bw for _, bw in available_values)
    for sig, insts in categories.items():
      if any(can_use_inst(inst, semas, available_bitwidths) for inst in insts):
        available_values_per_inst.append(available_values[:])
        gen_one_inst('i%d' % insts_declared, sig, semas, categories, available_values, outf)
        for i, bw in enumerate(sig):
          available_values.append(('r%d_%d' % (insts_declared, i), bw))
        sigs.append(sig)
        insts_declared += 1
  sigs.append(out_sig)
  available_values_per_inst.append(available_values)
  gen_one_inst('i%d' % insts_declared, out_sig, semas, categories, available_values, outf)

  # generate the wrapper
  outf.write('(define-fun soln (')
  # inputs
  for x, bw in live_ins:
    outf.write('(%s (BitVec %d))' % (x, bw))
  outf.write(')') # close param list
  # output
  outf.write(' (BitVec %d)\n' % sum(out_sig))
  # chain all the instructions together
  for i, sig in enumerate(sigs):
    avail = available_values_per_inst[i]
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
  outf.write('))' * len(sigs)) # close all the lets
  outf.write(')\n') # close define-fun

categories = categorize_insts(semas)
if __name__ == '__main__':
  import sys
  gen_sygus(2, (512,), semas, categories, [('x', 512), ('y', 512), ('k', 16)], sys.stdout)
