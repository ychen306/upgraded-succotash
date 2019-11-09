import z3

s = z3.Solver()
def prove(f):
  s.push()
  s.add(z3.Not(f))
  stat = s.check()
  s.pop()
  return stat == z3.unsat

def instantiate_with_imm8(inst, semas, sigs):
  '''
  inst -> [ConcreteInst]
  '''
  input_types, _ = sigs[inst]
  input_vals, output_vals = semas[inst]
  args = []

  has_imm8 = any(ty.is_constant for ty in input_types)
  if not has_imm8:
    return [(inst, None)]

  if len(output_vals) > 1:
    return [(inst, str(imm8)) for imm8 in range(256)]

  imm8_param = None
  params = []
  for ty, param in zip(input_types, input_vals):
    if ty.is_constant:
      imm8_param = param
    else:
      params.append(param)

  if len(params) > 2:
    return [(inst, str(imm8)) for imm8 in range(256)]

  assert imm8_param is not None

  [y] = output_vals
  specialized_outputs = []
  concrete_insts = []

  for imm8 in range(256):
    # partially specialize this instruction with the immediate
    specialized_y = z3.simplify(z3.substitute(y, (imm8_param, z3.BitVecVal(imm8, imm8_param.size()))))
    redundant = False
    for y2 in specialized_outputs:
      if prove(specialized_y == y2):
        redundant = True
        break
      elif len(params) == 2:
        x1, x2 = params
        if prove(specialized_y == z3.simplify(z3.substitute(y2, (x1,x2), (x2,x1)))):
          redundant = True
          break
    if not redundant:
      specialized_outputs.append(specialized_y)
      concrete_insts.append((inst, str(imm8)))
  return concrete_insts

if __name__ == '__main__':
  from expr_sampler import sigs, semas
  from tqdm import tqdm
  import json
  
  concrete_insts = []
  insts = list(sigs.keys())
  for inst in tqdm(insts):
    concrete_insts.extend(instantiate_with_imm8(inst, semas=semas, sigs=sigs))
  with open('instantiated-insts.json', 'w') as f:
    json.dump(concrete_insts, f)
