import torch
import operator
from sema2insts import Sema2Insts, expr2graph
import json
import random
from time import time

import z3
from z3_exprs import serialize_expr
from synth import synthesize, sigs, check_synth_batched

llvm_insts = [inst for inst in sigs.keys() if inst.startswith('llvm')]
inst_pool = []
with open('instantiated-insts.json') as f:
  for inst, imm8 in json.load(f):
    inst_pool.append((inst, imm8))

num_insts = len(inst_pool)
model = Sema2Insts(num_insts)
model.load_state_dict(torch.load('sema2insts.model'))
model.eval()

bvmax = lambda a,b : z3.If(a>=b, a, b)

x, y, z = z3.BitVecs('x y z', 32)
liveins = [('x', 32), ('y', 32)]

def p19_impl(x, m, k):
  o1 = x >> k;
  o2 = x ^ o1;
  o3 = o2 & m;
  o4 = o3 << k;
  o5 = o4 ^ o3;
  return o5 ^ x; 

one = z3.BitVecVal(1, 32)
zero = z3.BitVecVal(0, 32)

p01 = x & (x-1) # 0.496
p02 = x & (x+1) # 0.496
p03 = x & (-x) # 0.4899
p04 = x ^ (x-1) # 0.728
p05 = x | (x-1) # 1.068
p06 = x | (x+1) # 0.613
p07 = (~x) & (x+1) # 0.57
p08 = (~x) & (x-1) # 0.64
p09 = ((x >> 31) ^ x) - (x >> 31) # 2.6092
p10 = z3.If((x & y) <= (x ^ y), one, zero) # 1.24
p11 = z3.If(x & (~y) > y, one, zero) # no solution
p12 = z3.If(x & (~y) <= y, one, zero) # 4.07898
p13 = (x >>31) | (-x) >> 31 # 0.36
p14 = (x&y) + ((x^y) >> 1) # no solution
p15 = (x|y) - ((x^y) >> 1) # no solution
p16 = (x^y) & (-z3.If(z3.UGE(x,y), one, zero)) ^ y # no solution
p17 = (((x-1) | x) + 1) & x # 1.32
p18 = z3.If(z3.And(((x-1)&x)==0, x!=0), one, zero) # no solution
# p19 = timeout?

o1 = -x
o2 = x & o1
o3 = x + o2
o4 = x ^ o2
o5 = o4 >> 2
o6 = o5 / o2
p20 = o3 | o6 # no solution


#target = bvmax(x, y)  * y

liveins = [('x', 32)]

#liveins = [('x', 32), ('y', 32)]
liveins = [('x', 32), ('y', 32), ('z', 32)]
target = p19_impl(x, y, z)

target = p20

target = z3.simplify(target)

#ranges = {'z': (0, 32), 'x': (-10,10) }
#counter_examples = {'x': [0] }
counter_examples = {'x': [1<<3, 0b101 << 3, 1<<4, 1<<31, 1<<15, 1<<11, 0b11 << 23, 0b101 << 15] }
counter_examples = {}
counter_examples = {'z': list(range(32)) }

target_serialized = serialize_expr(target)
g, g_inv, ops, params, _ = expr2graph(target_serialized)
inst_probs = model(g, g_inv, ops, params).softmax(dim=0)
_, ids = inst_probs.topk(30)
insts = [inst_pool[i] for i in ids]
insts = [(inst, None) for inst in sigs if '32' in inst and 'llvm' in inst and 'Trunc' not in inst and 
    'Ext' not in inst
]
print(insts)
random.shuffle(insts)
begin = time()
out = synthesize(insts, target, liveins, timeout=60 * 60, num_levels=7, test_inputs=counter_examples)
end = time()

print(out)
print('Time elapsed:', end-begin)
