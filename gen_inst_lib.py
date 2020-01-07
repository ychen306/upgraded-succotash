from gen_enumerator import *
import json

def emit_insts_lib(out, h_out):
  insts = [
      ConcreteInst('ehad', None),
      ConcreteInst('arba', None),
      ConcreteInst('shesh', None),
      ConcreteInst('smol', None),
      ConcreteInst('im', None),
      ConcreteInst('bvnot', None),
      ConcreteInst('bvnot32', None),
      ConcreteInst('bvneg', None),
      ]

  with open('instantiated-insts.json') as f:
    for inst, imm8 in json.load(f):
      insts.append(ConcreteInst(inst, imm8))

  emit_includes(out)
  emit_inst_runners(insts, out, h_out)

with open('insts.c', 'w') as out, open('insts.h', 'w') as h_out:
  emit_insts_lib(out, h_out)
