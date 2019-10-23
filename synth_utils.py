from collections import defaultdict

def get_usable_insts(insts, sigs, available_values):
  '''
  Return the subset of `insts` that we can use, given available values
  '''
  # mapping <bitwidth> -> [<val>]
  bw2vals = defaultdict(list)
  for v in available_values:
    bw2vals[v.size()].append(v)

  usable_insts = []
  for inst in insts:
    in_types, _ = sigs[inst]
    if all(ty.bitwidth in bw2vals or ty.is_constant for ty in in_types):
      usable_insts.append(inst)
  return usable_insts
