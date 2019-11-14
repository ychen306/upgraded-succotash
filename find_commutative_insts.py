import z3
from collections import defaultdict
import itertools

def prove(f):
  s = z3.Solver()
  s.add(z3.Not(f))
  stat = s.check()
  return stat == z3.unsat

def get_commutative_params(inst, semas, sigs):
  '''
  inst -> [pair of param idxs]
  '''
  input_types, _ = sigs[inst]
  input_vals, output_vals = semas[inst]

  # mapping bitwidths -> (<params of the same bitwidths>, <parm idx>)
  bw2params = defaultdict(list)

  if len(output_vals) != 1:
    return []

  [y] = output_vals

  for i, (ty, param) in enumerate(zip(input_types, input_vals)):
    if not ty.is_constant:
      bw2params[param.size()].append((i, param))

  commutative_params = []
  for params in bw2params.values():
    if len(params) < 2:
      continue
    for (i, p1), (j, p2) in itertools.combinations(params, 2):
      y_swapped = z3.substitute(y, (p1, p2), (p2, p1))
      if prove(y_swapped == y):
        commutative_params.append((i,j))
  return commutative_params

if __name__ == '__main__':
  from expr_sampler import sigs, semas
  from tqdm import tqdm
  import json
  
  # mapping inst -> [<pair of param idxs that commute>]
  commutative_params = {}
  insts = list(sigs.keys())
  for inst in tqdm(insts):
    commutative_params[inst] = get_commutative_params(inst, semas, sigs)
  with open('commutative-params.json', 'w') as f:
    json.dump(commutative_params, f)
