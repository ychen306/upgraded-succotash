from fp_sema import set_precision

set_precision(False)

from spec_serializer import load_spec
from llvm_sema import *
import itertools
import z3

# mapping <intrinsic name> -> [<in vars>], [<out formulas>]
semas = {}

# get semantics of intrinsics
with open('intrinsics.sema') as sema_f:
  while True:
    break
    intrin_name = next(sema_f, None)
    if intrin_name is None:
      break

    spec = next(sema_f)
    if 'fp' in spec:
      continue

    #if 'mask' not in intrin_name:
    #  continue
    #if 'add_epi32' not in intrin_name:
    #  continue
    #if intrin_name.strip() == '_ktest_mask8_u8':
    #  f = load_spec(spec)[1][0]
    #  z3.simplify(f)
    #  print(f)
    #  print('=============')
    #  print(f.sexpr())
    #  exit(1)

    sema = load_spec(spec)
    semas[intrin_name.strip()] = sema
    _, ys = sema

# get semantics of llvm binary instructions
# note that we ignore vectorized instructions 
# (because we want to lower all IR vector instructions to intrinsics)
bitwidths = [1, 8, 16, 32, 64]
for op, impl in binary_ops.items():
  if op not in float_ops:
    possible_bitwidths = bitwidths
  else:
    continue
    possible_bitwidths = [32, 64]
  for bw in possible_bitwidths:
    if bw == 1:
      continue
    op_name = get_llvm_op_name(op, bw)
    x = z3.BitVec(op_name + '_x', bw)
    y = z3.BitVec(op_name + '_y', bw)
    semas[op_name] = (x,y), (impl(x,y),)

## get semantics of select
for bw in bitwidths:
  if bw == 1:
    continue
  op_name = get_llvm_op_name('Select', bw)
  c = z3.BitVec(op_name + '_c', 1)
  x = z3.BitVec(op_name + '_x', bw)
  y = z3.BitVec(op_name + '_y', bw)
  semas[op_name] = (c,x,y), (z3.If(c == 1, x, y),)

def get_trunc_sema(bw_in, bw_out):
  trunc_name = get_trunc_name(bw_in, bw_out)
  x = z3.BitVec(trunc_name + '_x', bw_in)
  sema = ((x,), (z3.Extract(bw_out-1, 0, x),))
  return trunc_name, sema

def get_sext_sema(bw_in, bw_out):
  sext_name = get_sext_name(bw_in, bw_out)
  x = z3.BitVec(sext_name + '_x', bw_in)
  sema = ((x,), (z3.SignExt(bw_out-bw_in, x),))
  return sext_name, sema

def get_zext_sema(bw_in, bw_out):
  zext_name = get_zext_name(bw_in, bw_out)
  x = z3.BitVec(zext_name + '_x', bw_in)
  sema = ((x,), (z3.ZeroExt(bw_out-bw_in, x),))
  return zext_name, sema

# get semantics of Trunc/SExt/ZExt
for bw_in, bw_out in itertools.product(bitwidths, bitwidths):
  if bw_in == bw_out:
    continue 
  if bw_in < bw_out:
    if bw_in > 1:
      sext_name, sext_sema = get_sext_sema(bw_in, bw_out)
      semas[sext_name] = sext_sema
    zext_name, zext_sema = get_zext_sema(bw_in, bw_out)
    semas[zext_name] = zext_sema
  elif bw_out > 1:
    trunc_name, trunc_sema = get_trunc_sema(bw_in, bw_out)
    semas[trunc_name] = trunc_sema

# return a map <ret type> -> [ insts ]
def categorize_insts(semas):
  categories = defaultdict(list)
  for inst, (_, outs) in semas.items():
    out_sig = tuple(out.size() for out in outs)
    categories[out_sig].append(inst)
  return categories
