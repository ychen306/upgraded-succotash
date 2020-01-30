'''
Lift smt formula to an IR similar to LLVM (minus control flow)
'''

from collections import namedtuple, defaultdict
import z3_utils
import z3

# "IR"
Instruction = namedtuple('Instruction', ['op', 'bitwidth', 'args'])
Constant = namedtuple('Constant', ['value', 'bitwidth'])

def trunc(x, size):
  return Instruction(op='Trunc', bitwidth=size, args=[x])

class Slice:
  def __init__(self, base, lo, hi):
    '''
    lo is inclusive and hi is exclusive
    '''
    self.base = base
    self.lo = lo
    self.hi = hi
    self.hash_key = base, lo, hi

  def __hash__(self):
    return hash(self.hash_key)

  def __eq__(self, other):
    return self.hash_key == other.hash_key

  def overlaps(self, other):
    return (
        self.base == other.base and
        self.size() + other.size() > self.union(other).size())

  def union(self, other):
    assert self.base == other.base
    lo = min(self.lo, other.lo)
    hi = max(self.hi, other.hi)
    return Slice(self.base, lo, hi)

  def size(self):
    return self.hi - self.lo

  def __repr__(self):
    return f'{self.base}[{self.lo}:{self.hi}]'

ops = {
    'Add', 'Sub', 'Mul', 'SDiv', 'SRem',
    'UDiv', 'URem', 'Shl', 'LShr', 'AShr',
    'And', 'Or', 'Xor',

    'FAdd', 'FSub', 'FMul', 'FDiv', 'FRem',

    'Foeq', 'Fone', 'Fogt', 'Foge', 'Folt', 'Fole',

    'Eq', 'Ne', 'Ugt', 'Uge', 'Ult', 'Ule', 'Sgt', 'Sge', 'Slt', 'Sle',
    }

op_table = {
    z3.Z3_OP_AND: 'And',
    z3.Z3_OP_OR: 'Or',
    z3.Z3_OP_XOR: 'Xor',
    #z3.Z3_OP_FALSE
    #z3.Z3_OP_TRUE
    z3.Z3_OP_NOT: 'Not',
    z3.Z3_OP_ITE: 'Select',
    z3.Z3_OP_BAND : 'And',
    z3.Z3_OP_BOR : 'Or',
    z3.Z3_OP_BXOR : 'Xor',
    z3.Z3_OP_SIGN_EXT: 'SExt',
    z3.Z3_OP_SIGN_EXT: 'ZExt',
    #z3.Z3_OP_BNOT
    #z3.Z3_OP_BNEG
    #z3.Z3_OP_CONCAT
    z3.Z3_OP_ULT : 'Ult',
    z3.Z3_OP_ULEQ : 'Ule',
    z3.Z3_OP_SLT : 'Slt',
    z3.Z3_OP_SLEQ : 'Sle',
    z3.Z3_OP_UGT : 'Ugt',
    z3.Z3_OP_UGEQ : 'Uge',
    z3.Z3_OP_SGT : 'Sgt',
    z3.Z3_OP_SGEQ : 'Sge',
    z3.Z3_OP_BADD : 'Add',
    z3.Z3_OP_BMUL : 'Mul',
    z3.Z3_OP_BUDIV : 'UDiv',
    z3.Z3_OP_BSDIV : 'SDiv',
    z3.Z3_OP_BUREM : 'URem', 
    #z3.Z3_OP_BSREM
    z3.Z3_OP_BSMOD : 'SRem',
    z3.Z3_OP_BSHL : 'Shl',
    z3.Z3_OP_BLSHR : 'LShr',
    z3.Z3_OP_BASHR : 'AShr',
    z3.Z3_OP_BSUB : 'Sub',
    z3.Z3_OP_EQ : 'Eq',
    #z3.Z3_OP_DISTINCT : 'distinct',

    z3.Z3_OP_BSDIV_I:  'SDiv',
    z3.Z3_OP_BUDIV_I:  'UDiv',
    #z3.Z3_OP_BSREM_I
    z3.Z3_OP_BUREM_I:  'URem',
    z3.Z3_OP_BSMOD_I:  'SRem',
    }

# translation table from uninterp. func to our ir (basically LLVM)
float_ops = {
    'add': 'FAdd',
    'sub': 'FSub',
    'mul': 'FMul',
    'div': 'FDiv',
    'lt': 'Folt',
    'le': 'Fole',
    'gt': 'Fogt',
    'ge': 'Foge',
    'ne': 'Fone',
    }

def is_simple_extraction(ext):
  '''
  check if `ext` is an extract on a variable
  '''
  [x] = ext.children()
  return (
      z3.is_app_of(x, z3.Z3_OP_UNINTERPRETED) and
      len(x.children()) == 0)

def partition_slices(slices):
  partition = set()
  for s in slices:
    for s2 in partition:
      if s.overlaps(s2):
        partition.remove(s2)
        partition.add(s.union(s2))
    else:
      partition.add(s)
  return partition

class ExtractionHistory:
  '''
  this class records all of the extraction
  done on a set of live-in bitvector,
  '''
  def __init__(self):
    # list of extracted slices
    self.extracted_slices = defaultdict(list)
    self.id_counter = 0

  def record(self, ext):
    assert is_simple_extraction(ext)
    [x] = ext.children()
    hi, lo = ext.params()
    s = Slice(x, lo, hi+1)
    self.extracted_slices[x].append(s)
    return s

  def translate_slices(self):
    '''
    return a map <slice> -> <ir>
    '''
    translated = {}
    for slices in self.extracted_slices.values():
      partition = partition_slices(slices)
      for s in slices:
        for root_slice in partition:
          if s.overlaps(root_slice):
            lo = s.lo - root_slice.lo
            hi = s.hi - root_slice.lo
            if s == root_slice:
              translated[s] = root_slice
            elif s.lo == 0:
              # truncation
              translated[s] = trunc(root_slice, s.size())
            else: # s.lo > 0
              # shift right + truncation
              shift = Instruction(
                  op='LShr',
                  bitwidth=root_slice.size(),
                  args=[root_slice])
              translated[s] = trunc(shift, s.size())
            break
    return translated


def recover_sub(f):
  '''
  z3 simplifies `a + b` to `a + 0b111..11 * b`,
  but we want to turn this into subtraction
  '''
  if not z3.is_app_of(f, z3.Z3_OP_BADD):
    return f
  args = f.children()
  if len(args) != 2:
    return f
  a, b = args
  if not z3.is_app_of(b, z3.Z3_OP_BMUL):
    if z3.is_app_of(a, z3.Z3_OP_BMUL):
      b, a = a, b
    else:
      return f
  b_args = b.children()
  if len(b_args) != 2:
    return f
  b1, b2 = b_args
  if z3.is_true(z3.simplify(b1 == z3.BitVecVal(-1, b1.size()))):
    return a - b2
  return f


class Translator:
  def __init__(self):
    self.extraction_history = ExtractionHistory()
    self.z3op_translators = {
        z3.Z3_OP_TRUE: self.translate_true, 
        z3.Z3_OP_FALSE: self.translate_false,
        z3.Z3_OP_BNOT: self.translate_not,
        z3.Z3_OP_BNEG: self.translate_neg,
        z3.Z3_OP_EXTRACT: self.translate_extract,
        z3.Z3_OP_CONCAT: self.translate_concat,
        z3.Z3_OP_UNINTERPRETED: self.translate_uninterpreted,
        z3.Z3_OP_BNUM: self.translate_constant,
        }
    # mapping <formula> -> <ir node id>
    self.translated = {}
    # translated IR
    self.ir = {}
    self.id_counter = 0

  def new_id(self):
    new_id = self.id_counter
    self.id_counter += 1
    return new_id

  def translate_constant(self, c):
    return Constant(value=c.as_long(), bitwidth=c.size())

  def translate_formula(self, f):
    '''
    entry point
    '''
    if not z3.is_app_of(f, z3.Z3_OP_CONCAT):
      outs = [self.translate(f)]
    else:
      # output is a concat, probably vector code
      outs = [self.translate(elem_f) for elem_f in f.children()]
    # translate the slices
    slice2ir = self.extraction_history.translate_slices()
    for node_id, node in self.ir.items():
      if isinstance(node, Slice):
        self.ir[node_id] = slice2ir[node]
    return outs, self.ir

  def translate(self, f):
    if f in self.translated:
      return self.translated[f]
    f = recover_sub(f)
    node_id = self.new_id()
    z3op = z3_utils.get_z3_app(f)

    if z3op in self.z3op_translators:
      # see if there's a specialized translator
      node = self.z3op_translators[z3op](f)
    else:
      # FIXME: need to use a preprocessing pass to shrink the bitwidth
      # in the `desc -> smt' pass to make things simple, we make the required
      # bitwidth unnecessarily large 
      # (e.g., to sum 8 8-bit values we need 64 bit addition!)
      # We should shrink them to make it realistic
      op = op_table[z3op]
      assert z3.is_bv(f) or z3.is_bool(f)
      bitwidth = f.size() if z3.is_bv(f) else 1
      node = Instruction(
          op=op, bitwidth=bitwidth, 
          args=[self.translate(arg) for arg in f.children()])

    self.translated[f] = node_id
    self.ir[node_id] = node
    return node_id

  def translate_true(*_):
    return Constant(z3.BitVecVal(1, 1))
  
  def translate_false(*_):
    return Constant(z3.BitVecVal(0, 1))
  
  def translate_not(self, f):
    [x] = f.children()
    # not x == xor -1, x
    return self.translate((-1) ^ x)
  
  def translate_neg(self, f):
    [x] = f.children()
    # not x == sub 0, x
    return self.translate(0-x)
  
  def translate_extract(self, ext):
    if is_simple_extraction(ext):
      s = self.extraction_history.record(ext)
      return s
    [x] = ext.children()
    assert x.size() <= 64,\
        "extraction too complex to model in scalar code"
    _, lo = ext.params()
    if lo > 0:
      return trunc(self.translate(z3.LShR(x, lo)), ext.size())
    return trunc(self.translate(x), ext.size())

  def translate_concat(self, concat):
    args = concat.children()
    a, b = args
    assert z3.is_bv(a) and a.as_long() == 0,\
        "only support using concat for zero extension"
    return Instruction(
        op='ZExt', bitwidth=concat.size(), args=[self.translate(b)])

  def translate_uninterpreted(self, f):
    args = f.children()
    if len(args) == 0:
      # live-in
      return f
    func = f.decl().name()
    assert func.startswith('fp_')
    _, op, _ = func.split('_')
    return Instruction(
        op=float_ops[op], bitwidth=f.size(),
        args=[self.translate(arg) for arg in f.children()])

if __name__ == '__main__':
    from semas import semas
    from pprint import pprint
    from tqdm import tqdm

    #translator = Translator()
    #y = semas['_mm_mulhi_pu16'][1][0]
    #outs, dag = translator.translate_formula(y)
    #pprint(outs)
    #pprint(dag)
    #exit()

    pbar = tqdm(iter(semas.items()), total=len(semas))
    num_tried = 0
    num_translated = 0
    for inst, sema in pbar:
      translator = Translator()
      y = sema[1][0]
      num_tried += 1
      print(inst)
      translator.translate_formula(y)
      #try:
      #  outs, dag = translator.translate_formula(y)
      #  num_translated += 1
      #except:
      #  pass
      pbar.set_description('translated/tried: %d/%d' % (
        num_translated, num_tried))
