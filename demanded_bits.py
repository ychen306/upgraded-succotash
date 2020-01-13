import z3

class RangeSet:
  def __init__(self):
    self.ranges = []
    self.cur_range = None

  def add(self, x):
    if self.cur_range is None:
      self.cur_range = (x, x)
      return

    lo, hi = self.cur_range
    assert x > hi
    if x == hi+1:
      self.cur_range = lo, hi+1
      return

    self.ranges.append(self.cur_range)
    self.cur_range = (x, x)

  def finalize(self):
    if self.cur_range is not None:
      self.ranges.append(self.cur_range)
      self.cur_range = None

  def __repr__(self):
    self.finalize()
    return ', '.join(repr(r) for r in self.ranges)

  def to_list(self):
    self.finalize()
    return self.ranges

def get_may_demanded_bits(x, y):
  '''
  Let `x' be a live-in of `y'.
  For each bits `i' of `y', find out bits of `x' that
  *could* affect `y[i]'
  '''
  s = z3.Solver()
  demanded_bits = {}
  for i in range(y.size()):
    demanded = RangeSet()
    for j in range(x.size()):
      s.push()
      x_prime = x ^ (~(1 << j))
      y_prime = z3.substitute(y, (x, x_prime))
      s.add(z3.Extract(i,i,y) != z3.Extract(i,i,y_prime))
      if s.check() != z3.unsat:
        demanded.add(j)
      s.pop()

    demanded_bits[i] = demanded.to_list()
  return demanded_bits

def get_must_demanded_bits(x, xs, y):
  s = z3.Solver()
  demanded_bits = {}
  for i in range(y.size()):
    demanded = RangeSet()
    for j in range(x.size()):
      s.push()
      x_prime = x ^ (~(1 << j))
      y_prime = z3.substitute(y, (x, x_prime))
      s.add(z3.ForAll(xs, 
        z3.Extract(i,i,y) != z3.Extract(i,i,y_prime)))
      if s.check() != z3.unsat:
        demanded.add(j)
      s.pop()

    demanded_bits[i] = demanded.to_list()
  return demanded_bits


from expr_sampler import sigs, semas

def get_may_demanded_bits_for_inst(concrete_inst):
  inst, imm8 = concrete_inst
  xs, ys = semas[inst]
  input_types, _ = sigs[inst]
  y = ys[0]
  for x, ty in zip(xs, input_types):
    if ty.is_constant:
      y = z3.substitute(y, (x, z3.BitVecVal(int(imm8), x.size())))
      break
  result = {'inst': (inst, imm8), 'demanded': []}
  for x in xs:
    result['demanded'].append(get_may_demanded_bits(x, y))
  return result

def get_must_demanded_bits_for_inst(concrete_inst):
  inst, imm8 = concrete_inst
  xs, ys = semas[inst]
  input_types, _ = sigs[inst]
  y = ys[0]
  for x, ty in zip(xs, input_types):
    if ty.is_constant:
      y = z3.substitute(y, (x, z3.BitVecVal(int(imm8), x.size())))
      break
  result = {'inst': (inst, imm8), 'demanded': []}
  for x in xs:
    result['demanded'].append(get_must_demanded_bits(x, xs, y))
  return result

if __name__ == '__main__':
  import json
  from tqdm import tqdm
  from multiprocessing import Pool
  
  pool = Pool(8)

  with open('instantiated-insts.json') as f:
    demanded_bits = []
    instantiated_insts = json.load(f)

  results = pool.imap_unordered(get_may_demanded_bits_for_inst, instantiated_insts)
  pbar = tqdm(iter(results), total=len(instantiated_insts))
  for result in pbar:
    demanded_bits.append(result)

  with open('demanded-bits.json', 'w') as out:
    json.dump(demanded_bits, out)
