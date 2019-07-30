from ast import *
from collections import namedtuple
import operator
from bitstring import Bits
import math

# is is_pointer then this is a pointer to the type
ConcreteType = namedtuple('ConcreteType', ['bitwidth', 'is_float', 'is_double', 'is_pointer'])

IntegerType = lambda bw: ConcreteType(bw, False, False, False)
FloatType = lambda bw: ConcreteType(bw, True, False, False)
DoubleType = lambda bw: ConcreteType(bw, False, True, False)
PointerType = lambda ty: ty._replace(is_pointer=True)

# convert textual types like '_m512i' to ConcreteType
concrete_types = {
    '__m512i': IntegerType(512),
    '__m256i': IntegerType(256),
    '__m128i': IntegerType(128),
    '__m64': IntegerType(64),

    # single precision floats
    '__m512': FloatType(512),
    '__m256': FloatType(256),
    '__m128': FloatType(128),

    # double precision floats
    '__m512d': DoubleType(512),
    '__m256d': DoubleType(256),
    '__m128d': DoubleType(128),

    # masks
    '__mmask8': IntegerType(8),
    '__mmask16': IntegerType(8),
    '__mmask32': IntegerType(8),
    '__mmask64': IntegerType(8),

    'int': IntegerType(32),
    'uint': IntegerType(32),
    'unsigned int': IntegerType(32),
    '__int64': IntegerType(64),
    }

def get_default_value(type):
  if type.is_float:
    num_elems = type.bitwidth // 32
    return [1] * num_elems
  elif type.is_double:
    num_elems = type.bitwidth // 64
    return [1] * num_elems
  return Bits(type.bitwidth)
  
def as_one(slice_or_val):
  if type(slice_or_val) == list:
    assert len(slice_or_val) == 1
    return slice_or_val[0]
  return slice_or_val

def as_many(val):
  if type(val) == list:
    return val
  return [val]

class Environment:
  def __init__(self):
    self.vars = {}

  def define(self, name, type, value=None):
    assert name not in self.vars
    if value is None:
      value = get_default_value(type)
    self.vars[name] = type, value

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
  def slice(self, lo, hi):
    new_slice = self.__class__(self.var, lo+self.lo_idx, hi+self.lo_idx)
    return new_slice

  def __repr__(self):
    return '%s[%d:%d]' % (self.var, self.lo_idx, self.hi_idx)

class FloatSlice(Slice):
  def __init__(self, var, lo_idx, hi_idx):
    '''
    var : str,
    *_idx : int
    '''
    assert lo_idx % 32 == 0
    assert hi_idx % 32 == 31
    self.var = var
    self.lo_idx = lo_idx
    self.hi_idx = hi_idx

  def get_lo_idx(self):
    return self.lo_idx // 32

  def get_hi_idx(self):
    return (self.hi_idx + 1) // 32

  def update(self, rhs, env):
    '''
    rhs : [float]
    '''
    val = env.get_value(self.var)
    val[self.get_lo_idx() : self.get_hi_idx()] = as_many(rhs)
    env.set_value(self.var, val)

  def get_value(self, env):
    val = env.get_value(self.var)
    return val[self.get_lo_idx() : self.get_hi_idx()]

class DoubleSlice(FloatSlice):
  def __init__(self, var, lo_idx, hi_idx):
    '''
    var : str,
    *_idx : int
    '''
    assert lo_idx % 64 == 0
    assert hi_idx % 64 == 63
    self.var = var
    self.lo_idx = lo_idx
    self.hi_idx = hi_idx

  def get_lo_idx(self):
    return self.lo_idx // 64

  def get_hi_idx(self):
    return (self.hi_idx + 1) // 64

class IntegerSlice(Slice):
  def __init__(self, var, lo_idx, hi_idx):
    self.var = var
    self.lo_idx = lo_idx
    self.hi_idx = hi_idx
    self.zero_extending = True

  def mark_sign_extend(self):
    self.zero_extending = False

  def update(self, rhs, env):
    '''
    rhs : integer
    '''
    val = env.get_value(self.var)
    bitwidth = env.get_type(self.var).bitwidth
    extend = zero_extend if self.zero_extending else sign_extend
    assert (self.lo_idx >= 0 and
        self.lo_idx < self.hi_idx and
        self.hi_idx < bitwidth)
    val |= extend(rhs, bitwidth) << self.lo_idx
    env.set_value(self.var, val)

  def get_value(self, env):
    bitwidth = self.hi_idx - self.lo_idx + 1
    total_bitwidth = env.get_type(self.var).bitwidth
    val = env.get_value(self.var)
    val <<= total_bitwidth - self.hi_idx + 1
    val >>= (total_bitwidth - self.hi_idx + 1) + self.lo_idx
    # restrict the bitwidth
    val = Bits(uint=val.uint, length=bitwidth)
    return val

def unreachable():
  assert False

def get_value(v, env):
  if isinstance(v, Slice):
    return v.get_value(env)
  return v

def binary_op(op):
  def impl(a, b):
    bitwidth = a.length
    #return Bits(int=op(a.int, b.int), length=bitwidth)
    return Bits(int=op(a.int, b.int), length=64)
  return impl

def binary_shift(op):
  return lambda a, b: op(a, b.int)

def unary_op(op):
  def impl(a):
    bitwidth = a.length
    return Bits(int=op(a.int), length=bitwidth)
  return impl

def binary_float_op(op):
  def impl(a, b):
    bitwidth = a.length
    return Bits(float=op(a.float,b.float), length=bitwidth)
  return impl

# mappgin <op, is_float?> -> impl
binary_op_impls = {
    ('+', True): binary_float_op(operator.add),
    ('-', True): binary_float_op(operator.sub),
    ('*', True): binary_float_op(operator.mul),
    ('/', True): binary_float_op(operator.truediv),

    ('*', False): binary_op(operator.mul),
    ('+', False): binary_op(operator.add),
    ('-', False): binary_op(operator.sub),
    ('>', False): binary_op(operator.gt),
    ('<', False): binary_op(operator.gt),
    ('<<', False): binary_shift(operator.lshift),
    ('>>', False): binary_shift(operator.rshift),
    ('AND', False): binary_op(operator.and_),
    ('OR', False): binary_op(operator.or_),
    ('XOR', False): binary_op(operator.xor),
    ('==', False): binary_op(operator.eq),
    }

# mappgin <op, is_float?> -> impl
unary_op_impls = {
    ('NOT', False): unary_op(operator.not_),
    }


def int_to_bits(x, bitwidth=32):
  if x < 0:
    return Bits(int=x, length=bitwidth)
  return Bits(uint=x, length=bitwidth)

def get_signed_max(bitwidth):
  return (1<<(bitwidth-1))-1

def get_signed_min(bitwidth):
  return -get_signed_max(bitwidth)

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
    return Bits(int=val, length=bitwidth), IntegerType(bitwidth)
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
    return op(as_one(slice_or_val)), ty
  return impl

def builtin_int_to_float(args, env):
  [(slice_or_val, ty)] = args
  return slice_or_val.int, FloatType(32)

def builtin_binary_func(op):
  def impl(args, _):
    [(a, ty), (b, _)] = args

    if not is_float(ty):
      return Bits(int=op(a.int, b.int), length=ty.bitwidth), ty

    # float
    return op(as_one(a), as_one(b)), ty
  return impl

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

    'ZeroExtend': ignore,
    'SignExtend': ignore,
    'ABS': builtin_unary_func(abs),
    'Convert_Int32_To_FP32': builtin_int_to_float,
    'Convert_Int64_To_FP32': builtin_int_to_float,
    'SQRT': builtin_unary_func(math.sqrt),
    'APPROXIMATE': ignore,
    'MIN': builtin_binary_func(min),
    'MAX': builtin_binary_func(max),
    }

def is_float(type):
  return type.is_float or type.is_double

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

  lhs, _ = interpret_expr(update.lhs, env)

  if sign_extending:
    lhs.mark_sign_extend()

  if update.modifier is None:
    rhs_val = get_value(rhs, env)
    lhs.update(rhs_val, env)
    return rhs_val
  unreachable()

def interpret_var(var, env):
  '''
  don't return a value but a slice/reference which can be update/deref later
  '''
  type = env.get_type(var.name)
  slice = IntegerSlice(var.name, 0, type.bitwidth-1)
  return slice, type

def is_number(expr):
  return type(expr) == Number

def interpret_binary_expr(expr, env):
  a, a_type = evaluate_expr(expr.a, env)
  b, b_type = evaluate_expr(expr.b, env)
  assert a_type == b_type or is_number(expr.a) or is_number(expr.b) or expr.op in ('<<', '>>')
  impl_sig = expr.op, is_float(a_type)
  impl = binary_op_impls[impl_sig]
  return impl(as_one(a), as_one(b)), a_type

def interpret_unary_expr(expr, env):
  a, a_type = evaluate_expr(expr.a, env)
  impl_sig = expr.op, is_float(a_type)
  impl = unary_op_impls[impl_sig]
  return impl(as_one(a)), a_type

def interpret_for(for_stmt, env):
  iterator = for_stmt.iterator
  begin, _ = interpret_expr(for_stmt.begin, env)
  end, _ = interpret_expr(for_stmt.end, env)
  inc = lambda v: int_to_bits(v.int+1)
  dec = lambda v: int_to_bits(v.int-1)
  cont_inc = lambda : env.get_value(iterator).int <= end.int
  cont_dec = lambda : env.get_value(iterator).int >= end.int
  update_iterator = inc if for_stmt.inc else dec
  cont = cont_inc if for_stmt.inc else done_dec
  env.define(iterator, IntegerType(32), value=get_value(begin, env))
  while cont():
    for stmt in for_stmt.body:
      interpret_stmt(stmt, env)
    new_iterator_value = update_iterator(env.get_value(iterator))
    env.set_value(iterator, new_iterator_value)

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
    _, ty = interpret_expr(bit_slice.bv, env)
    hi = int_to_bits(ty.bitwidth-1)
  else:
    hi, _ = evaluate_expr(bit_slice.hi, env)

  # in case we have a variable implicitly declared
  # assume only integers can be implicitly declared
  if (type(bit_slice.bv) == Var and
      not env.has(bit_slice.bv.name)):
    env.define(bit_slice.bv.name, type=IntegerType(hi.int+1))
  slice_src, ty = interpret_expr(bit_slice.bv, env)
  assert (isinstance(slice_src, Slice) or
      isinstance(slice_src, Bits))

  # compute update type with new bitwidth
  slice_width = hi.int - lo.int + 1
  if ty.is_float:
    new_type = FloatType(slice_width)
  elif ty.is_double:
    new_type = DoubleType(slice_width)
  else:
    new_type = IntegerType(slice_width)

  # in case the bits we are slicing from
  # is a result of a computation, not a variable
  if type(slice_src) == Bits:
    sliced = slice_src[lo.int:hi.int]
    return Bits(uint=sliced.uint, length=slice_width), new_type

  # adjust bitwidth in case the variable is implicitly defined
  # allow integer slice to have implicitly declared bitwidth
  if type(slice_src) == IntegerSlice:
    bitwidth = max(ty.bitwidth, hi.int+1)
    env.set_type(slice_src.var, IntegerType(bitwidth))
    val = env.get_value(slice_src.var)
    env.set_value(slice_src.var, zero_extend(val, bitwidth))

  return slice_src.slice(lo.int, hi.int), new_type

def interpret_match(match_stmt, env):
  val, _ = evaluate_expr(match_stmt.val, env)
  cases = {}
  for case in match_stmt.cases:
    case_val, _ = evaluate_expr(case.val, env)
    if case_val.int == val.int:
      for stmt in case.stmts:
        if type(stmt) == Break:
          break
        interpret_stmt(stmt, env)
      break
  return

def interpret_call(call, env):
  assert type(call.func) == str
  args = [evaluate_expr(arg, env) for arg in call.args]
  if call.func in builtins:
    # assume builtins don't modify the environment
    return builtins[call.func](args, env)

  # assume there is no indirect calls
  func_def = env.get_value(call.func)

  # Calling a user defined function
  new_env = Environment()
  assert len(func_def.params) == len(args)
  for (arg, arg_type), param in zip(args, func_def.params):
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
      param_width = arg
      assert arg_type.bitwidth == param_width
    new_env.define(param_name, arg_type, arg)

  for stmt in func_def.body:
    if type(stmt) == Return:
      return evaluate_expr(stmt.val, env)
    interpret_stmt(stmt, env)

  # user defined function should return explicitly
  unreachable()

def interpret_number(n, _):
  if type(n.val) == int:
     return int_to_bits(n.val), IntegerType(32)
  return n.val, DoubleType(64)

def interpret_func_def(func_def, env):
  env.define(func_def.name, type=None, value=func_def)

# dispatch table for different ast nodes
interpreters = {
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
  similar to interpret_expr, except we force evaluation of a slice to a concrete value
  '''
  slice_or_val, type = interpret_expr(expr, env)
  return get_value(slice_or_val, env), type

# TODO: before execution, step over (statically) all assignment to determine bitwidth of all variables and define them ahead of time
def interpret(spec, args=None):
  # bring arguments into scope
  env = Environment()
  if args is None:
    args = [None] * len(spec.params)
  for arg, param in zip(args, spec.params):
    param_type = concrete_types[param.type]
    env.define(param.name, type=param_type, value=arg)
  env.define('dst', type=concrete_types[spec.rettype])

  for stmt in spec.spec:
    interpret_stmt(stmt, env)

  return env.get_value('dst')

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
  a = [1,2]
  b = [1,2]
  c = interpret(spec, [a, b])
  print(c)
  c = interpret(spec, [[4,5], [7,8]])
  print(c)
  c = interpret(spec, [[2,3], [4,9]])
  print(c)
