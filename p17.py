import torch
import operator
import json
import random
from time import time

import z3
from synth import synthesize, sigs, check_synth_batched

insts = [
    ('bvnot32', None),
    ('llvm_Xor_32', None), # bvxor
    ('llvm_And_32', None), # bvand
    ('llvm_Or_32', None), # bvor
    ('bvneg', None),
    ('llvm_Add_32', None), # bvadd
    ('llvm_Mul_32', None), # bvmul
    ('llvm_UDiv_32', None), # bvudiv
    ('llvm_URem_32', None), # bvurem
    ('llvm_LShr_32', None), # bvlshr
    ('llvm_AShr_32', None), # bvashr
    ('llvm_Shl_32', None), # bvshl
    ('llvm_SDiv_32', None), # bvsdiv
    ('llvm_SRem_32', None), # bvsrem
    ('llvm_Sub_32', None), # bvsub
    ]

x = z3.BitVec('x', 32)
liveins = [('x', 32)]
target = (((x-1) | x) + 1) & x

random.shuffle(insts)
counter_examples = {}

begin = time()
out = synthesize(insts, target, liveins, timeout=3600, num_levels=4, test_inputs=counter_examples,
    constants = [(0, 32), (1, 32), (0x1f, 32), (0xFFFFFFFF, 32)])
end = time()

print(out)
print('Time elapsed:', end-begin)
