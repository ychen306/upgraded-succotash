from sema_ast import *
from collections import namedtuple
import operator
from bitstring import Bits, BitArray, CreationError
import math

'''
TODO: refactor all the "if signed... Bits(..) else Bits(..) "
  by using helper funcs like create_signed_bits...
'''

from bit_util import *
from intrinsic_types import (
    intrinsic_types, max_vl,
    IntegerType, FloatType, DoubleType,
    is_float)


def get_default_value(type):
  if type.is_float:
    num_elems = type.bitwidth // 32
    return float_vec_to_bits([.1] * num_elems, float_size=32)
  elif type.is_double:
    num_elems = type.bitwidth // 64
    return float_vec_to_bits([.1] * num_elems, float_size=64)
  return Bits(type.bitwidth)

def as_many(val):
  if type(val) == list:
    return val
  return [val]

class Environment:
  def __init__(self, func_defs=None):
    self.vars = {}
    if func_defs is None:
      func_defs = {}
    self.func_defs = func_defs
    # mapping expr -> signess
    self.reconfigured_binary_exprs = {}

  def configure_binary_expr_signedness(self, configs):
    self.reconfigured_binary_exprs = dict(configs)

  def get_binary_expr_signedness(self, expr):
    return self.reconfigured_binary_exprs.get(expr.expr_id, None)

  def new_env(self):
    return Environment(self.func_defs)

  def def_func(self, func, func_def):
    self.func_defs[func] = func_def

  def get_func(self, func):
    return self.func_defs[func]

  def define(self, name, type, value=None):
    assert name not in self.vars
    if value is None:
      value = get_default_value(type)
    self.vars[name] = type, value

  def undef(self, name):
    del self.vars[name]

  def has(self, name):
    return name in self.vars

  def get_type(self, name):
    ty, _ = self.vars[name]
    return ty

  def set_type(self, name, ty):
    _, val = self.vars[name]
    self.vars[name] = ty, val

  def get_value(self, name):
    _, val = self.vars[name]
    return val

  def set_value(self, name, value):
    type = self.get_type(name)
    self.vars[name] = type, value

class Slice:
  def __init__(self, var, lo_idx, hi_idx, stride=1):
    self.var = var
    self.lo_idx = lo_idx
    self.hi_idx = hi_idx
    self.zero_extending = True
    self.stride = stride

  def set_stride(self, stride):
    return Slice(self.var, self.lo_idx, self.hi_idx, stride)

  def slice(self, lo, hi):
    lo = lo * self.stride
    hi = (hi+1) * self.stride - 1
    return Slice(self.var, lo+self.lo_idx, hi+self.lo_idx, self.stride)

  def __repr__(self):
    return '%s[%d:%d]' % (self.var, self.lo_idx, self.hi_idx)

  def mark_sign_extend(self):
    self.zero_extending = False

  def update(self, rhs, env):
    '''
    rhs : integer
    '''
    bitwidth = env.get_type(self.var).bitwidth

    hi_idx = min(self.hi_idx, bitwidth-1)

    if self.lo_idx > hi_idx:
      return

    if rhs == None: # undefined
      return

    old_val = env.get_value(self.var)
    old_bits = BitArray(uint=old_val.uint, length=old_val.length)

    update_width = hi_idx - self.lo_idx + 1

    if update_width < rhs.length:
      # trunc rhs
      rhs = slice_bits(rhs, 0, update_width)

    extend = zero_extend if self.zero_extending else sign_extend
    if not (self.lo_idx >= 0 and
        self.lo_idx <= hi_idx and
        hi_idx < bitwidth):
      print(self.lo_idx, hi_idx, bitwidth)
    assert (self.lo_idx >= 0 and
        self.lo_idx <= hi_idx and
        hi_idx < bitwidth)
    update_bits(old_bits, self.lo_idx, hi_idx+1, extend(rhs, hi_idx-self.lo_idx+1))
    new_val = Bits(uint=old_bits.uint, length=old_bits.length)
    env.set_value(self.var, new_val)

  def get_value(self, env):
    bitwidth = self.hi_idx - self.lo_idx + 1
    total_bitwidth = env.get_type(self.var).bitwidth
    if not (self.lo_idx >= 0 and
        self.lo_idx <= self.hi_idx and
        self.hi_idx < total_bitwidth):
      return None
    #if not (self.lo_idx >= 0 and
    #    self.lo_idx <= self.hi_idx and
    #    self.hi_idx < total_bitwidth):
    #  print(self.var, self.lo_idx, self.hi_idx)
    #assert (self.lo_idx >= 0 and
    #    self.lo_idx <= self.hi_idx and
    #    self.hi_idx < total_bitwidth)
    val = slice_bits(env.get_value(self.var), self.lo_idx, self.hi_idx+1)
    val = Bits(uint=val.uint, length=bitwidth)
    return val

def unreachable():
  assert False

def get_value(v, env):
  if isinstance(v, Slice):
    return v.get_value(env)
  return v

def binary_float_cmp(op):
  def impl(a, b, _=None):
    return Bits(uint=op(a.uint, b.uint), length=1)
  return impl

def binary_op(op, trunc=False, signed=True, get_bitwidth=lambda a, b: a.length):
  def impl(a, b, signed_override=signed):
    bitwidth = get_bitwidth(a, b)
    mask = (1 << bitwidth)-1
    if signed_override:
      return Bits(int=op(a.int, b.int), length=bitwidth)
    else:
      c = op(a.uint, b.uint)
      if trunc:
        c &= mask
      return Bits(uint=c, length=bitwidth)
  return impl

def binary_sub(a, b, signed=False):
  if signed:
    return Bits(int=a.int-b.int, length=get_total_arg_width(a,b))

  bitwidth = get_total_arg_width(a,b)
  mask = (1 << bitwidth) - 1
  b_neg = ((1<<bitwidth) - b.uint)
  c = Bits(uint=(a.uint + b_neg) & mask, length=bitwidth)
  return c

def binary_neg(a):
  mask = (1 << a.length) - 1
  a_neg = ((1<<a.length) - a.uint)
  return Bits(uint=a_neg & mask, length=a.length)

def binary_complement(a):
  mask = (1 << a.length) - 1
  a_comp = ((1<<a.length) - a.uint - 1)
  return Bits(uint=a_comp & mask, length=a.length)

def binary_shift(op):
  return lambda a, b : op(a, b.uint)

def binary_lshift(a, b, _=None):
  bitwidth = max(a.length, min(max_vl, b.uint+1))
  mask = (1 << bitwidth)-1
  if b.uint <= max_vl:
    c = a.uint << b.uint
    c &= mask
  else:
    c = 0
  return Bits(uint=c, length=bitwidth)

def unary_op(op, signed=True):
  def impl(a):
    bitwidth = a.length
    if signed:
      return Bits(int=op(a.int), length=bitwidth)
    else:
      return Bits(uint=op(a.uint), length=bitwidth)
  return impl

def unary_float_op(op):
  def impl(a):
    bitwidth = a.length
    return Bits(float=op(a.float), length=bitwidth)
  return impl

def binary_float_op(op):
  def impl(a, b, _=None):
    bitwidth = a.length
    return Bits(float=op(a.float,b.float), length=bitwidth)
  return impl

def rol(bits, a):
  bit_array = BitArray(uint=bits.uint, length=bits.length)
  bit_array.rol(a)
  return BitArray(uint=bit_array.uint, length=bits.length)

get_max_arg_width = lambda a, b: max(a.length, b.length)
get_total_arg_width = lambda a, b: a.length + b.length

# mapping <op, is_float?> -> impl
binary_op_impls = {
    ('+', True): binary_float_op(operator.add),
    ('-', True): binary_float_op(operator.sub),
    ('*', True): binary_float_op(operator.mul),
    ('/', True): binary_float_op(operator.truediv),
    ('<', True): binary_float_cmp(operator.lt),
    ('<=', True): binary_float_cmp(operator.le),
    ('>', True): binary_float_cmp(operator.gt),
    ('>=', True): binary_float_cmp(operator.ge),
    ('!=', True): binary_op(operator.ne),
    ('>>', True): binary_shift(operator.rshift),
    #('<<', True): binary_shift(operator.lshift),
    ('<<', True): binary_lshift,

    ('AND', True): binary_op(operator.and_, signed=False),
    ('OR', True): binary_op(operator.or_, signed=False, get_bitwidth=get_max_arg_width),
    ('XOR', True): binary_op(operator.xor, signed=False),

    # FIXME: what about the signednes?????
    ('*', False): binary_op(operator.mul, get_bitwidth=get_total_arg_width, signed=False),
    ('+', False): binary_op(operator.add, get_bitwidth=get_total_arg_width, signed=False),
    #('-', False): lambda a, b: a + (~b+1),
    ('-', False): binary_sub,
    ('>', False): binary_op(operator.gt, signed=False),
    ('>=', False): binary_op(operator.ge),
    ('<', False): binary_op(operator.lt),
    ('<=', False): binary_op(operator.le),
    ('%', False): binary_op(operator.mod, signed=True),
    ('<<', False): binary_lshift,
    ('<<<', False): binary_shift(rol), # is this correct??
    ('>>', False): binary_shift(operator.rshift), # what about the signedness???

    ('AND', False): binary_op(operator.and_, signed=False),
    ('&', False): binary_op(operator.and_, signed=False),
    ('|', False): binary_op(operator.or_, signed=False, get_bitwidth=get_max_arg_width),
    ('OR', False): binary_op(operator.or_, signed=False, get_bitwidth=get_max_arg_width),
    ('XOR', False): binary_op(operator.xor, signed=False),

    ('!=', False): binary_op(operator.ne),
    }

# mapping <op, is_float?> -> impl
unary_op_impls = {
    #('NOT', True): unary_op(operator.not_, signed=False),
    ('NOT', True): binary_complement,
    ('-', True): unary_float_op(operator.neg),

    #('NOT', False): unary_op(operator.not_, signed=False),
    ('NOT', False): binary_complement,
    ('-', False): binary_neg,
    ('~', False): binary_complement,
    }

def get_signed_max(bitwidth):
  return (1<<(bitwidth-1))-1

def get_signed_min(bitwidth):
  return -get_signed_max(bitwidth)-1

def get_unsigned_max(bitwidth):
  return (1<<bitwidth)-1

def get_unsigned_min(bitwidth):
  return 0

def gen_saturation_func(bitwidth, in_signed, out_signed):
  hi = get_signed_max(bitwidth) if out_signed else get_unsigned_max(bitwidth)
  lo = get_signed_min(bitwidth) if out_signed else get_unsigned_min(bitwidth)
  def saturate(args, env):
    [(slice_or_val, _)] = args
    bits = get_value(slice_or_val, env)
    val = bits.int if in_signed else bits.uint
    if val < lo:
      val = lo
    elif val > hi:
      val = hi
    if out_signed:
      return Bits(int=val, length=bitwidth), IntegerType(bitwidth)
    else:
      return Bits(uint=val, length=bitwidth), IntegerType(bitwidth)
  return saturate

def zero_extend(x, bitwidth):
  return Bits(uint=x.uint, length=bitwidth)

def sign_extend(x, bitwidth):
  return Bits(int=x.int, length=bitwidth)

def builtin_unary_func(op):
  def impl(args, _):
    [(slice_or_val, ty)] = args

    if not is_float(ty):
      bits = slice_or_val
      return Bits(int=op(bits.int), length=ty.bitwidth), ty

    # float
    return Bits(float=op(slice_or_val.float), length=ty.bitwidth), ty
  return impl

def builtin_int_to_float(signed=True, bitwidth=32):
  def impl(args, env):
    [(slice_or_val, ty)] = args
    val = slice_or_val.int if signed else slice_or_val.uint
    return Bits(float=val, length=bitwidth), FloatType(bitwidth)
  return impl

def builtin_int_to_double(args, env):
  [(slice_or_val, ty)] = args
  return Bits(float=slice_or_val.int, length=64), FloatType(64)

def builtin_float_to_int(args, env):
  [(slice_or_val, ty)] = args
  return Bits(int=int(slice_or_val.float+0.5), length=32), IntegerType(32)

def builtin_float_to_long(args, env):
  [(slice_or_val, ty)] = args
  return Bits(int=int(slice_or_val.float+0.5), length=64), IntegerType(64)

def builtin_float_to_int_trunc(args, env):
  [(slice_or_val, ty)] = args
  return Bits(int=int(slice_or_val.float), length=32), IntegerType(32)

def builtin_float_to_long_trunc(args, env):
  [(slice_or_val, ty)] = args
  return Bits(int=int(slice_or_val.float), length=64), IntegerType(64)

def builtin_double_to_float(args, env):
  [(slice_or_val, ty)] = args
  return Bits(int=int(slice_or_val.float), length=32), IntegerType(32)

def builtin_float_to_double(args, env):
  [(slice_or_val, ty)] = args
  return Bits(int=int(slice_or_val.float), length=64), IntegerType(64)

def builtin_binary_func(op):
  def impl(args, _):
    [(a, ty), (b, _)] = args

    if not is_float(ty):
      return Bits(int=op(a.int, b.int), length=ty.bitwidth), ty

    # float
    return Bits(float=op(a.float, b.float), length=ty.bitwidth), ty
  return impl

def builtin_abs(args, _):
  [(a, ty)] = args

  if not is_float(ty):
    return Bits(uint=abs(a.int), length=ty.bitwidth), ty

  # float
  return Bits(float=abs(a.float), length=ty.bitwidth), ty

def builtin_concat(args, _):
  [(a, a_ty), (b, b_ty)] = args
  assert not is_float(a_ty) and not is_float(b_ty)
  return a + b, IntegerType(a_ty.bitwidth+b_ty.bitwidth)

def builtin_popcount(args, _):
  [(a, _)] = args
  return Bits(int=a.bin.count('1'), length=32), IntegerType(32)

def builtin_select(args, _):
  '''
  select dword (32-bit) in a[...] by b
  '''
  [(a, a_ty), (b, _)] = args
  bit_idx = b.uint * 32
  selected = slice_bits(a, bit_idx, bit_idx+32).uint
  return Bits(uint=selected, length=32), a_ty._replace(bitwidth=32)

def builtin_zero_extend_to_512(args, _):
  [(a, a_ty)] = args
  return zero_extend(a, 512), a_ty._replace(bitwidth=512)

ignore = lambda args, env: args[0]

builtins = {
    'Saturate_Int16_To_Int8': gen_saturation_func(8, True, True),
    'Saturate_Int16_To_UnsignedInt8': gen_saturation_func(8, True, False),
    'Saturate_Int32_To_Int16': gen_saturation_func(16, True, True),
    'Saturate_Int32_To_Int8': gen_saturation_func(8, True, True),
    'Saturate_Int32_To_UnsignedInt16': gen_saturation_func(16, True, False),
    'Saturate_Int64_To_Int16': gen_saturation_func(16, True, True),
    'Saturate_Int64_To_Int32': gen_saturation_func(32, True, True),
    'Saturate_Int64_To_Int8': gen_saturation_func(8, True, True),
    'Saturate_To_Int16': gen_saturation_func(16, True, True),
    'Saturate_To_Int8': gen_saturation_func(8, True, True),
    'Saturate_To_UnsignedInt16': gen_saturation_func(16, True, False),
    'Saturate_To_UnsignedInt8': gen_saturation_func(8, True, False),
    'Saturate_UnsignedInt16_To_Int8': gen_saturation_func(8, False, True),
    'Saturate_UnsignedInt32_To_Int16': gen_saturation_func(16, False, True),
    'Saturate_UnsignedInt32_To_Int8': gen_saturation_func(8, False, True),
    'Saturate_UnsignedInt64_To_Int16': gen_saturation_func(16, False, True),
    'Saturate_UnsignedInt64_To_Int32': gen_saturation_func(32, False, True),
    'Saturate_UnsignedInt64_To_Int8': gen_saturation_func(8, False, True),
    'SIGNED_DWORD_SATURATE': gen_saturation_func(32, True, True),
    'Truncate_Int32_To_Int8': gen_saturation_func(8, True, True),
    'Truncate_Int64_To_Int8': gen_saturation_func(8, True, True),
    'Truncate_Int32_To_Int16': gen_saturation_func(16, True, True),
    'Truncate_Int64_To_Int32': gen_saturation_func(32, True, True),
    'Truncate_Int64_To_Int16': gen_saturation_func(32, True, True),
    'Truncate_Int16_To_Int8': gen_saturation_func(8, True, True),

    # FIXME: do this here rather than "lookahead" interpret_update
    'ZeroExtend': ignore,
    'SignExtend': ignore,
    'ZeroExtend64': ignore,
    'ZeroExtend_To_512': builtin_zero_extend_to_512,
    'Convert_Int32_To_FP32': builtin_int_to_float(signed=True, bitwidth=32),
    'Convert_Int64_To_FP32': builtin_int_to_float(signed=True, bitwidth=32),
    'Convert_Int32_To_FP64': builtin_int_to_double,
    'Convert_Int64_To_FP64': builtin_int_to_double,
    'Convert_FP32_To_Int32': builtin_float_to_int,
    'Convert_FP64_To_Int32': builtin_float_to_int,
    'Convert_FP32_To_Int64': builtin_float_to_long,
    'Convert_FP64_To_Int64': builtin_float_to_long,
    'Convert_FP32_To_Int32_Truncate': builtin_float_to_int_trunc,
    'Convert_FP64_To_Int32_Truncate': builtin_float_to_int_trunc,
    'Convert_FP64_To_Int64_Truncate': builtin_float_to_long_trunc,
    'Convert_FP64_To_FP32': builtin_double_to_float,
    'Convert_FP32_To_FP64': builtin_float_to_double,
    'Float64ToFloat32': builtin_double_to_float,
    'Float32ToFloat64': builtin_float_to_double,
    # FIXME: fix the signedness issue.
    'Convert_FP64_To_UnsignedInt32': builtin_float_to_int,
    'Convert_FP32_To_UnsignedInt32': builtin_float_to_int,
    'Convert_FP64_To_UnsignedInt64': builtin_float_to_long,
    'Convert_FP32_To_UnsignedInt64': builtin_float_to_long,
    'Convert_FP64_To_UnsignedInt32_Truncate': builtin_float_to_int_trunc,
    'Convert_FP32_To_UnsignedInt32_Truncate': builtin_float_to_int_trunc,
    'Convert_FP64_To_UnsignedInt64_Truncate': builtin_float_to_long_trunc,
    'Convert_FP32_To_Int64_Truncate': builtin_float_to_long_trunc,
    'Convert_FP32_To_UnsignedInt64_Truncate': builtin_float_to_long_trunc,
    'Convert_FP32_To_IntegerTruncate': builtin_float_to_long_trunc,
    'ConvertUnsignedIntegerTo_FP64': builtin_int_to_float(signed=False, bitwidth=64),
    'ConvertUnsignedInt32_To_FP32': builtin_int_to_float(signed=False, bitwidth=32),
    'Convert_UnsignedInt32_To_FP64': builtin_int_to_float(signed=False, bitwidth=64),
    'Convert_UnsignedInt64_To_FP64': builtin_int_to_float(signed=False, bitwidth=64),
    'Convert_UnsignedInt32_To_FP32': builtin_int_to_float(signed=False, bitwidth=32),
    'Convert_UnsignedInt64_To_FP32': builtin_int_to_float(signed=False, bitwidth=32),
    'ConvertUnsignedInt64_To_FP64': builtin_int_to_float(signed=False, bitwidth=64),
    'ConvertUnsignedInt64_To_FP32': builtin_int_to_float(signed=False, bitwidth=32),
    'Int32ToFloat64': builtin_int_to_float(signed=True, bitwidth=64),
    'UInt32ToFloat64': builtin_int_to_float(signed=False, bitwidth=64),

    'APPROXIMATE': ignore,
    'MIN': builtin_binary_func(min),
    'MAX': builtin_binary_func(max),
    'ABS': builtin_abs,
    'SQRT': builtin_unary_func(math.sqrt),
    'FLOOR': builtin_unary_func(math.floor),
    'CEIL': builtin_unary_func(math.ceil),
    'PopCount': builtin_popcount,
    'POPCNT': builtin_popcount,
    'select': builtin_select,

    # bitstring has overloaded plus for concatenation
    'concat': builtin_concat,
    }

# TODO: handle integer overflow here
def interpret_update(update, env):
  rhs, rhs_type = interpret_expr(update.rhs, env)

  if (type(update.rhs) == Call and
    type(update.rhs.func) == Var and
    update.rhs.func.name == 'SignExtend'):
    sign_extending = True
  else:
    sign_extending = False

  # TODO: refactor this shit out
  if type(update.lhs) == Var and not env.has(update.lhs.name):
    env.define(update.lhs.name, rhs_type)
    assert env.has(update.lhs.name)

  lhs, _ = interpret_expr(update.lhs, env)

  if sign_extending:
    lhs.mark_sign_extend()

  rhs_val = get_value(rhs, env)
  lhs.update(rhs_val, env)
  return rhs_val

def interpret_var(var, env):
  '''
  return a slice/reference, which can be update/deref later
  '''
  if var.name == 'undefined':
    return None, None
  type = env.get_type(var.name)
  slice = Slice(var.name, 0, type.bitwidth-1)
  return slice, type

def is_number(expr):
  return type(expr) == Number

def collect_chained_cmpeq(expr, chained):
  if type(expr) != BinaryExpr or expr.op != '==':
    chained.append(expr)
    return
  
  collect_chained_cmpeq(expr.a, chained)
  collect_chained_cmpeq(expr.b, chained)

def interpret_binary_expr(expr, env):
  # special case for expression like "a == b == c == d"
  if expr.op == '==':
    chained_operands = []
    collect_chained_cmpeq(expr, chained_operands)
    vals = [evaluate_expr(operand, env) for operand in chained_operands]
    v, _ = vals[0]
    equal = True
    for v2, _ in vals[1:]:
      if v2.uint != v.uint:
        equal = False
        break
    return Bits(uint=equal, length=1), IntegerType(1)

  a, a_type = evaluate_expr(expr.a, env)
  # special case for expressions that can be short-cirtuited
  if expr.op == 'AND':
    if not a.uint:
      return Bits(int=0, length=a.length), a_type

  b, b_type = evaluate_expr(expr.b, env)

  # automatically change bit widths 
  # TODO: make sure this crap is correct
  #if a_type.bitwidth < 16:
  #  a = sign_extend(a, 16)
  #  a_type = a_type._replace(bitwidth=16)
  #if b_type.bitwidth < 16:
  #  b = sign_extend(b, 16)
  #  b_type = b_type._replace(bitwidth=16)
  #assert a_type == b_type or is_number(expr.a) or is_number(expr.b) or expr.op in ('<<', '>>')

  impl_sig = expr.op, is_float(a_type)
  impl = binary_op_impls[impl_sig]
  # check the configuration for whether this expression is signed
  signedness = env.get_binary_expr_signedness(expr)
  if signedness is not None:
    result = impl(a, b, signedness)
  else:
    # if signedness is not specified, just use the default
    result = impl(a, b)
  return result, a_type._replace(bitwidth=result.length)

def interpret_unary_expr(expr, env):
  a, a_type = evaluate_expr(expr.a, env)
  impl_sig = expr.op, is_float(a_type)
  impl = unary_op_impls[impl_sig]
  return impl(a), a_type

def interpret_while(while_stmt, env):
  def keep_going():
    cond, _ = evaluate_expr(while_stmt.cond, env)
    return cond.int

  while keep_going():
    for stmt in while_stmt.body:
      interpret_stmt(stmt, env)

def interpret_for(for_stmt, env):
  iterator = for_stmt.iterator
  begin, _ = evaluate_expr(for_stmt.begin, env)
  end, _ = evaluate_expr(for_stmt.end, env)
  inc = lambda v: int_to_bits(v.int+1)
  dec = lambda v: int_to_bits(v.int-1)
  cont_inc = lambda : env.get_value(iterator).int <= end.int
  cont_dec = lambda : env.get_value(iterator).int >= end.int
  update_iterator = inc if for_stmt.inc else dec
  cont = cont_inc if for_stmt.inc else cont_dec
  env.define(iterator, IntegerType(32), value=get_value(begin, env))
  while cont():
    for stmt in for_stmt.body:
      interpret_stmt(stmt, env)
    new_iterator_value = update_iterator(env.get_value(iterator))
    env.set_value(iterator, new_iterator_value)
  env.undef(iterator)

def interpret_if(if_stmt, env):
  cond, _ = evaluate_expr(if_stmt.cond, env)
  stmts = if_stmt.then if cond.int else if_stmt.otherwise
  for stmt in stmts:
    interpret_stmt(stmt, env)

def interpret_select(select, env):
  cond, _ = evaluate_expr(select.cond, env)
  expr = select.then if cond.int else select.otherwise
  return interpret_expr(expr, env)

def interpret_bit_slice(bit_slice, env):
  lo, _ = evaluate_expr(bit_slice.lo, env)

  # special case for the magic variable 'MAX' 
  if (type(bit_slice.hi) == Var and
      bit_slice.hi.name == 'MAX'):
    hi = int_to_bits(max_vl-1)
  else:
    hi, _ = evaluate_expr(bit_slice.hi, env)

  # in case we have a variable implicitly declared
  # assume only integers can be implicitly declared
  if (type(bit_slice.bv) == Var and
      not env.has(bit_slice.bv.name)):
    env.define(bit_slice.bv.name, type=IntegerType(hi.uint+1))
  slice_src, ty = interpret_expr(bit_slice.bv, env)
  assert (isinstance(slice_src, Slice) or
      isinstance(slice_src, Bits))

  # compute update type with new bitwidth
  slice_width = hi.uint - lo.uint + 1
  if type(slice_src) == Slice:
    slice_width *= slice_src.stride
  if ty.is_float:
    new_type = FloatType(slice_width)
  elif ty.is_double:
    new_type = DoubleType(slice_width)
  else:
    new_type = IntegerType(slice_width)

  # in case the bits we are slicing from
  # is a result of a computation, not a variable
  if type(slice_src) == Bits:
    sliced = slice_bits(slice_src, lo.uint, hi.uint+1)
    return Bits(uint=sliced.uint, length=slice_width), new_type

  # adjust bitwidth in case the variable is implicitly defined
  # allow integer slice to have implicitly declared bitwidth
  if not is_float(ty):
    bitwidth = max(env.get_type(slice_src.var).bitwidth, hi.uint+1)
    env.set_type(slice_src.var, IntegerType(bitwidth))
    val = env.get_value(slice_src.var)
    new_val = zero_extend(val, bitwidth)
    env.set_value(slice_src.var, new_val)

  return slice_src.slice(lo.uint, hi.uint), new_type

def interpret_match(match_stmt, env):
  val, _ = evaluate_expr(match_stmt.val, env)
  cases = {}
  for case in match_stmt.cases:
    case_val, _ = evaluate_expr(case.val, env)
    if case_val.uint == val.uint:
      for stmt in case.stmts:
        if type(stmt) == Break:
          break
        interpret_stmt(stmt, env)
      break
  return

def interpret_call(call, env):
  assert type(call.func) == str

  # compute all the arguments
  args = [evaluate_expr(arg, env) for arg in call.args]

  # calling a builtin
  if call.func in builtins:
    # assume builtins don't modify the environment
    return builtins[call.func](args, env)

  # assume there is no indirect calls
  func_def = env.get_func(call.func)

  # Calling a user defined function
  # Pass the arguments first
  new_env = env.new_env()
  assert len(func_def.params) == len(args)
  for (arg, arg_type), param in zip(args, func_def.params):
    # make sure the argument bitwidths match 
    if type(param) == BitSlice:
      assert type(param.bv) == Var
      assert is_number(param.lo)
      assert is_number(param.hi)
      assert param.lo.val == 0
      param_name = param.bv.name
      param_width = param.hi.val+1
      assert arg_type.bitwidth == param_width
    else:
      assert type(param) == Var
      param_name = param.name
      param_width = arg.length
      assert arg_type.bitwidth == param_width
    new_env.define(param_name, arg_type, arg)

  # step over the function
  for stmt in func_def.body:
    if type(stmt) == Return:
      return evaluate_expr(stmt.val, new_env)
    interpret_stmt(stmt, new_env)

  # user defined function should return explicitly
  unreachable()

def interpret_number(n, _):
  if type(n.val) == int:
    try:
      return int_to_bits(n.val), IntegerType(32)
    except CreationError:
      return int_to_bits(n.val, 64), IntegerType(64)
  return float_to_bits(n.val), FloatType(32)

def interpret_func_def(func_def, env):
  env.def_func(func_def.name, func_def)

def interpret_lookup(lookup, env):
  '''
  essentially these expression returns a slice
  whose stride is specified by the property,
  which by defualt is `bit'.

  Some examples:

  a[127:0] is a slice of bits from 0 to 127
  a.byte[0] is a slice of bits from 0 to 7
  a.dword[0].bit[2] is ...
  '''
  if (type(lookup.obj) == Var and
      not env.has(lookup.obj.name)):
    # implicitly defined obj, we will refine the bitwidth later
    env.define(lookup.obj.name, type=IntegerType(1))
  strides = {
      'bit': 1,
      'byte': 8,
      'word': 16,
      'dword': 32,
      'qword': 64,
      }
  stride = strides[lookup.key]
  obj, ty = interpret_expr(lookup.obj, env)
  return obj.set_stride(stride), ty

# dispatch table for different ast nodes
interpreters = {
    While: interpret_while,
    For: interpret_for,
    Update: interpret_update,
    BinaryExpr: interpret_binary_expr,
    UnaryExpr: interpret_unary_expr,
    Call: interpret_call,
    Var: interpret_var,
    Number: interpret_number,
    BitSlice: interpret_bit_slice,
    If: interpret_if,
    Select: interpret_select,
    FuncDef: interpret_func_def,
    Match: interpret_match,
    Lookup: interpret_lookup,
    }

def interpret_stmt(stmt, env):
  interp = interpreters[type(stmt)]
  interp(stmt, env)

def interpret_expr(expr, env):
  '''
  return val, type
  '''
  interp = interpreters[type(expr)]
  value = interp(expr, env)
  return value

def evaluate_expr(expr, env):
  '''
  similar to interpret_expr, except we deref a slice to a concrete value
  '''
  slice_or_val, ty = interpret_expr(expr, env)
  return get_value(slice_or_val, env), ty

def interpret(spec, args=None):
  # bring the arguments into scope
  env = Environment()
  env.configure_binary_expr_signedness(spec.configs)
  if args is None:
    args = [None] * len(spec.params)
  out_params = []
  returns_void = False
  for arg, param in zip(args, spec.params):
    if param.type.endswith('*'):
      param_type = intrinsic_types[param.type[:-1].strip()]
      out_params.append(param.name)
    else:
      param_type = intrinsic_types[param.type]
    env.define(param.name, type=param_type, value=arg)
  if spec.rettype != 'void':
    env.define('dst', type=intrinsic_types[spec.rettype])
  else:
    returns_void = True

  for stmt in spec.spec:
    if type(stmt) == Return:
      assign_to_dst = Update(lhs=Var('dst'), rhs=stmt.val)
      interpret_stmt(assign_to_dst, env)
      break
    interpret_stmt(stmt, env)

  outputs = [env.get_value(out_param) for out_param in out_params]
  if returns_void:
    dst = None
  else:
    dst = env.get_value('dst')
  return dst, outputs

if __name__ == '__main__':

  from manual_parser import get_spec_from_xml

  import xml.etree.ElementTree as ET
  sema = '''
  <intrinsic tech='SSE2' vexEq='TRUE' rettype='__m128d' name='_mm_add_pd'>
        <type>Floating Point</type>
        <CPUID>SSE2</CPUID>
        <category>Arithmetic</category>
        <parameter varname='a' type='__m128d'/>
        <parameter varname='b' type='__m128d'/>
        <description>Add packed double-precision (64-bit) floating-point elements in "a" and "b", and store the results in "dst".</description>
        <operation>
FOR j := 0 to 1
        i := j*64
        dst[i+63:i] := a[i+63:i] + b[i+63:i]
ENDFOR
        </operation>
        <instruction name='addpd' form='xmm, xmm'/>
        <header>emmintrin.h</header>
</intrinsic>
  '''

  intrin_node = ET.fromstring(sema)
  spec = get_spec_from_xml(intrin_node)
  eager_map = lambda f, xs: list(map(f, xs))
  a = [1,2]
  b = [1,2]
  c = interpret(spec, eager_map(float_vec_to_bits, [a, b]))
  print(bits_to_float_vec(c[0]))
  c = interpret(spec, eager_map(float_vec_to_bits, [[4,5], [7,8]]))
  print(bits_to_float_vec(c[0]))
  c = interpret(spec, eager_map(float_vec_to_bits, [[2,3], [4,9]]))
  print(bits_to_float_vec(c[0]))

  sema = '''
  <intrinsic tech="Other" rettype='unsigned int' name='_mulx_u32'>
    <type>Integer</type>
    <CPUID>BMI2</CPUID>
    <category>Arithmetic</category>
    <parameter type='unsigned int' varname='a' />
    <parameter type='unsigned int' varname='b' />
    <parameter type='unsigned int*' varname='hi' />
    <description>Multiply unsigned 32-bit integers "a" and "b", store the low 32-bits of the result in "dst", and store the high 32-bits in "hi". This does not read or write arithmetic flags.</description>
    <operation>
dst[31:0] := (a * b)[31:0]
hi[31:0] := (a * b)[63:32]
    </operation>
    <instruction name='mulx' form='r32, r32, m32' />
    <header>immintrin.h</header>
</intrinsic>
  '''
  intrin_node = ET.fromstring(sema)
  spec = get_spec_from_xml(intrin_node)
  print(interpret(spec, [Bits(uint=2, length=32), Bits(uint=3, length=32)]))

  sema = '''
  <intrinsic tech="AVX-512" rettype="__m512i" name="_mm512_shuffle_epi8">
    <type>Integer</type>
    <CPUID>AVX512BW</CPUID>
    <category>Miscellaneous</category>
    <parameter varname="a" type="__m512i"/>
    <parameter varname="b" type="__m512i"/>
    <description>Shuffle packed 8-bit integers in "a" according to shuffle control mask in the corresponding 8-bit element of "b", and store the results in "dst".</description>
    <operation>
FOR j := 0 to 63
    i := j*8
    IF b[i+7] == 1
        dst[i+7:i] := 0
    ELSE
        index[5:0] := b[i+3:i] + (j &amp; 0x30)
        dst[i+7:i] := a[index*8+7:index*8]
    FI
ENDFOR
dst[MAX:512] := 0
    </operation>
    <instruction name="vpshufb"/>
    <header>immintrin.h</header>
</intrinsic>
  '''
  intrin_node = ET.fromstring(sema)
  spec = get_spec_from_xml(intrin_node)
  print(interpret(spec, [Bits(uint=2, length=32), Bits(uint=3, length=32)]))

  sema = '''
  <intrinsic tech="Other" rettype='unsigned int' name='_bextr_u32'>
    <type>Integer</type>
    <CPUID>BMI1</CPUID>
    <category>Bit Manipulation</category>
    <parameter type='unsigned int' varname='a' />
    <parameter type='unsigned int' varname='start' />
    <parameter type='unsigned int' varname='len' />
    <description>Extract contiguous bits from unsigned 32-bit integer "a", and store the result in "dst". Extract the number of bits specified by "len", starting at the bit specified by "start".</description>
    <operation>
tmp := ZeroExtend_To_512(a)
dst := ZeroExtend(tmp[start[7:0]+len[7:0]-1:start[7:0]])
    </operation>
    <instruction name='bextr' form='r32, r32, r32'/>
    <header>immintrin.h</header>
</intrinsic>
  '''
  intrin_node = ET.fromstring(sema)
  spec = get_spec_from_xml(intrin_node)
  print(interpret(spec, [Bits(uint=0b1011001, length=32), Bits(uint=2, length=32), Bits(uint=4, length=32)]))
