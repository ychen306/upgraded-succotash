from gen_enumerator import *
import json

def emit_insts_lib(out, h_out):
  insts = []
  with open('instantiated-insts.json') as f:
    for inst, imm8 in json.load(f):
      insts.append(ConcreteInst(inst, imm8))
  _, nodes = make_fully_connected_graph(
      liveins=[('x', 64), ('y', 64)],
      constants=[],
      insts=insts,
      num_levels=4)

  emit_includes(out)
  emit_inst_runners(nodes, out, h_out)

with open('insts.c', 'w') as out, open('insts.h', 'w') as h_out:
  emit_insts_lib(out, h_out)
