import operator
from fp_sema import *
import z3

'''
    Opaque,

    Add,
    Sub,
    Mul,
    SDiv,
    SRem,
    UDiv,
    URem,
    Shl,
    LShr,
    AShr,
    And,
    Or,
    Xor,
    SExt,
    ZExt,
    Trunc,
    PtrToInt,
    BitCast,

    // Floating point arith
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,

    // Floating point icmp
    Foeq,
    Fone,
    Fogt,
    Foge,
    Folt,
    Fole,
    Fueq,
    Fune,
    Fugt,
    Fuge,
    Fult,
    Fule,
    Ftrue,

    // ICmp
    Eq,
    Ne,
    Ugt,
    Uge,
    Ult,
    Ule,
    Sgt,
    Sge,
    Slt,
    Sle,

    Select,

    // vector insts
    InsertElement,
    ExtractElement,
    ShuffleVector,

    Constant,
    FConstant
'''

unsupported_opcode = {'Fueq',
    'Fune',
    'Fugt',
    'Fuge',
    'Fult',
    'Fule',
    'Ftrue'}

def is_supported(dag):
  return all(inst.op not in unsupported_opcode
      for inst in dag.values())

def to_bitvec(f):
  def impl(*args):
    b = f(*args)
    return z3.If(b, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1))
  return impl

def get_llvm_op_name(op, out_bw):
  return 'llvm_%s_%d' % (op, out_bw)

def get_trunc_name(bw_in, bw_out):
  return get_llvm_op_name('Trunc%d' % bw_in, bw_out)

def get_sext_name(bw_in, bw_out):
  return get_llvm_op_name('SExt%d' % bw_in, bw_out)

def get_zext_name(bw_in, bw_out):
  return get_llvm_op_name('ZExt%d' % bw_in, bw_out)

def shift_op(op):
  def impl(a, b):
    mask = z3.BitVecVal((1 << a.size()) - 1, b.size())
    return op(a, b & mask)
  return impl

# Not included:
#    SExt,
#    ZExt,
#    Trunc,
#    Select,
#    
#    InsertElement,
#    ExtractElement,
#    ShuffleVector,
binary_ops = {
    'Add': operator.add,
    'Sub': operator.sub,
    'Mul': operator.mul,
    'SDiv': lambda a,b: a / b,
    'SRem': lambda a,b: a % b,
    'UDiv': z3.UDiv,
    'URem': z3.URem,
    'Shl': shift_op(operator.lshift),
    'LShr': shift_op(z3.LShR),
    'AShr': shift_op(operator.rshift),
    'And': lambda a, b: a & b,
    'Or': operator.or_,
    'Xor': operator.xor,

    'FAdd': binary_float_op('add'),
    'FSub': binary_float_op('sub'),
    'FMul': binary_float_op('mul'),
    'FDiv': binary_float_op('div'),
    'FRem': binary_float_op('rem'),

    'Foeq': binary_float_cmp('eq'),
    'Fone': binary_float_cmp('ne'),
    'Fogt': binary_float_cmp('gt'),
    'Foge': binary_float_cmp('ge'),
    'Folt': binary_float_cmp('lt'),
    'Fole': binary_float_cmp('le'),

    'Eq': to_bitvec(operator.eq),
    'Ne': to_bitvec(operator.ne),
    'Ugt': to_bitvec(z3.UGT),
    'Uge': to_bitvec(z3.UGE),
    'Ult': to_bitvec(z3.ULT),
    'Ule': to_bitvec(z3.ULE),
    'Sgt': to_bitvec(operator.gt),
    'Sge': to_bitvec(operator.ge),
    'Slt': to_bitvec(operator.lt),
    'Sle': to_bitvec(operator.le),
    }

binary_syntaxes = {
    'Add': '+',
    'Sub': '-',
    'Mul': '*',
    'SDiv': '/',
    'SRem': '%',
    'UDiv': '/',
    'URem': '%',
    'Shl': '<<',
    'LShr': '>>',
    'AShr': '>>',
    'And': '&',
    'Or': '|',
    'Xor': '^',

    'FAdd': '+',
    'FSub': '-',
    'FMul': '*',
    'FDiv': '/',
    'FRem': '%',

    'Foeq': '==',
    'Fone': '!=',
    'Fogt': '>',
    'Foge': '>=',
    'Folt': '<',
    'Fole': '<=',

    'Eq': '==',
    'Ne': '!=',
    'Ugt': '>',
    'Uge': '>=',
    'Ult': '<',
    'Ule': '<=',
    'Sgt': '>',
    'Sge': '>=',
    'Slt': '<',
    'Sle': '<=',
    }

signed_binary_ops = {
    'SDiv', 'SRem', 'AShr', 'Sgt', 'Sge', 'Slt', 'Sle',
    }

divisions = {
    'SDiv', 'UDiv', 'URem', 'SRem',
    }

comparisons = {
    'Foeq',
    'Fone',
    'Fogt',
    'Foge',
    'Folt',
    'Fole',

    'Eq',
    'Ne',
    'Ugt',
    'Uge',
    'Ult',
    'Ule',
    'Sgt',
    'Sge',
    'Slt',
    'Sle',
    }


float_ops = {
    'FAdd',
    'FSub',
    'FMul',
    'FDiv',
    'FRem',

    'Foeq',
    'Fone',
    'Fogt',
    'Foge',
    'Folt',
    'Fole',
    }

def unpack_vector(v, elem_bitwidth):
  assert v.size() % elem_bitwidth == 0
  return [z3.Extract(i+elem_bitwidth-1, i, v)
      for i in range(0, v.size(), elem_bitwidth)]

def fix_bitwidth(x, bitwidth):
  return z3.Extract(bitwidth-1, 0, z3.ZeroExt(bitwidth, x))

def build_mask(idx, elem_bitwidth, total_bitwidth):
  lower_bits = fix_bitwidth(idx, total_bitwidth) * elem_bitwidth
  return z3.BitVecVal((1 << elem_bitwidth) - 1, total_bitwidth) << lower_bits

def extract_element(src, idx, bitwidth):
  idx = z3.simplify(idx)

  # fast path to generate easy-on-the-eyes formula
  if z3.is_bv_value(idx):
    begin = idx.as_long() * bitwidth
    end = begin + bitwidth - 1
    return z3.Extract(end, begin, src)

  total_bitwidth = src.size()
  lower_bits = fix_bitwidth(idx, total_bitwidth) * bitwidth
  mask = build_mask(idx, bitwidth, total_bitwidth)
  return fix_bitwidth((src & ~mask) >> lower_bits, bitwidth)

def revconcat(xs):
  if len(xs) == 1:
    return xs[0]
  return z3.Concat(list(reversed(xs)))

def get_sema(dag):
  # mapping inst_id -> symbolic value
  vals = {}

  def get_inst_bitwidth(inst_id):
    return dag[inst_id].bitwidth

  def interpret(inst_id):
    if inst_id in vals:
      return vals[inst_id]
    inst = dag[inst_id]

    if inst.op == 'Opaque':
      total_bitwidth = inst.bitwidth * inst.vectorlen
      return z3.BitVec('x%d' % inst_id, total_bitwidth)

    if inst.op in binary_ops:
      # syntax : OP A B
      impl = binary_ops[inst.op]
      # the semantic of the vectorized instructions in binary_ops
      # are simple: apply the operation element-wise
      a = interpret(inst.op_a)
      b = interpret(inst.op_b)
      a_elems = unpack_vector(a, get_inst_bitwidth(inst.op_a))
      b_elems = unpack_vector(b, get_inst_bitwidth(inst.op_b))
      return revconcat([
        impl(aa, bb)
        for aa, bb in zip(a_elems, b_elems)])

    if inst.op in ('PtrToInt', 'BitCast'):
      # these instructions don't modify values
      return interpret(inst.op_a)

    if inst.op == 'Select':
      # syntax : select cond, then, else
      cond = interpret(inst.op_a)
      a = interpret(inst.op_b)
      b = interpret(inst.op_c)
      preds = unpack_vector(cond, 1)
      a_elems = unpack_vector(a, inst.bitwidth)
      b_elems = unpack_vector(b, inst.bitwidth)
      return revconcat([
        z3.If(pred == 1, aa, bb)
        for pred, aa, bb in zip(preds, a_elems, b_elems)])

    if inst.op in ('Trunc', 'SExt', 'ZExt'):
      if inst.op == 'Trunc':
        fix_bitwidth_ = lambda x: z3.Extract(inst.bitwidth-1, 0, x)
      elif inst.op == 'SExt':
        fix_bitwidth_ = lambda x: z3.SignExt(inst.bitwidth-x.size(), x)
      else: # inst.op == 'ZExt'
        fix_bitwidth_ = lambda x: z3.ZeroExt(inst.bitwidth-x.size(), x)
      orig_bitwidth = dag[inst.op_a].bitwidth
      xs = interpret(inst.op_a)
      return revconcat([
        fix_bitwidth_(x)
        for x in unpack_vector(xs, orig_bitwidth)])

    if inst.op == 'InsertElement':
      # syntax: insertelement src, elem, idx
      src = interpret(inst.op_a)
      elem = interpret(inst.op_b)
      idx = interpret(inst.op_c)
      total_bitwidth = inst.bitwidth * inst.vectorlen
      lower_bits = fix_bitwidth(idx, total_bitwidth) * inst.bitwidth
      mask = build_mask(idx, inst.bitwidth, total_bitwidth)
      return (src & ~mask) | (fix_bitwidth(elem, total_bitwidth) << lower_bits)

    if inst.op == 'ExtractElement':
      # syntax: extractelement src, idx
      src = interpret(inst.op_a)
      idx = interpret(inst.op_b)
      assert inst.vectorlen == 1
      return extract_element(src, idx, inst.bitwidth)

    if inst.op == 'ShuffleVector':
      # syntax : shufflevector src1, src2, idxs
      src1 = interpret(inst.op_a)
      src2 = interpret(inst.op_b)
      idxs = unpack_vector(interpret(inst.op_c), 32)
      src = z3.Concat(src2, src1)
      return revconcat([
        extract_element(src, i, inst.bitwidth)
        for i in idxs
        ])

    if inst.op == 'Constant':
      assert inst.init_list is not None
      return revconcat([
        z3.BitVecVal(c, inst.bitwidth)
        for c in inst.init_list
        ])

    assert inst.op == 'FConstant'
    return revconcat([
      fp_literal(c, inst.bitwidth)
      for c in inst.init_list
      ])

  for i in dag:
    vals[i] = z3.simplify(interpret(i))
  return vals

if __name__ == '__main__':
  from dag_parser import parse_dag
  from pprint import pprint
  dag_text = '''0,Add,32,4,1,2,3:
1,Constant,32,1,-1,-1,-1:42
2,Constant,32,1,-1,-1,-1:0'''
  dag = parse_dag(dag_text.split('\n'))
  pprint(get_sema(dag))
