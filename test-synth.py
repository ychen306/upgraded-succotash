import torch
from coder import Synthesizer, InstPool
import operator
from sema2insts import Sema2Insts, expr2graph

import z3
from z3_exprs import serialize_expr
from synth import synthesize, sigs, check_synth_batched

llvm_insts = [inst for inst in sigs.keys() if inst.startswith('llvm')]

num_insts = len(llvm_insts)
model = Sema2Insts(num_insts)
model.load_state_dict(torch.load('synth.model'))
model.eval()

bvmax = lambda a,b : z3.If(a>=b, a, b)

x, y = z3.BitVecs('x y', 64)
liveins = [('x', 64), ('y', 64)]
target = bvmax(x, y)  * y


target_serialized = serialize_expr(target)
g, g_inv, ops, params, _ = expr2graph(target_serialized)
inst_probs = model(g, g_inv, ops, params).softmax(dim=0)
samples = [torch.multinomial(inst_probs, 20) for _ in range(64)]
insts_batch = [[llvm_insts[i] for i in inst_indices] for inst_indices in samples]
print('SYNTHESIZING')
synthesized = check_synth_batched(insts_batch, target, liveins)
print(synthesized)
exit()
insts = insts_batch[synthesized.topk(1).indices[0]]
print(insts)
print(synthesize(insts, target, liveins).decode('utf-8'))
