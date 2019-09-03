from llvm_instruction import Instruction

def parse_inst(inst_def):
  defn, init = inst_def.split(':')
  op, bitwidth, vectorlen, op_a, op_b, op_c = defn.split(',')
  if init == '':
    init_list = None
  else:
    init_list = [int(x) for x in init.split(',')]

  return Instruction(op, int(bitwidth),
      int(vectorlen),
      int(op_a),
      int(op_b),
      int(op_c),
      init_list)

def parse_dag(lines):
  # mapping <inst id> -> <parsed inst>
  dag = {}
  for line in lines:
    line = line.split('#')[0] # chop off the comments
    i, inst_def = line.split(',', 1)
    dag[int(i)] = parse_inst(inst_def)
  return dag
