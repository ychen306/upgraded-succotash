import torch
from coder import Synthesizer, InstPool
import operator
from sema2insts import Sema2Insts, expr2graph

import z3
from z3_exprs import serialize_expr
from synth import synthesize, sigs

llvm_insts = [inst for inst in sigs.keys() if inst.startswith('llvm')]
num_insts = len(llvm_insts)
model = Sema2Insts(num_insts)
model.load_state_dict(torch.load('synth.model'))
model.eval()


x, y = z3.BitVecs('x y', 32)
liveins = [('x',32), ('y',32)]
target = z3.simplify((x + y))
target_serialized = serialize_expr(target)
g, g_inv, ops, params, _ = expr2graph(target_serialized)
inst_probs = model(g, g_inv, ops, params).softmax(dim=0)
inst_indices = torch.multinomial(inst_probs, 20)
insts = [llvm_insts[i] for i in inst_indices] + ['llvm_Add_64']
print(insts)
print(synthesize(insts, target, liveins))
