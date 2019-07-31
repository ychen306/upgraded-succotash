from ast import *
from collections import namedtuple
import operator
from bitstring import Bits, BitArray, CreationError
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

    'float': FloatType(32),
    'double': FloatType(64),
    'int': IntegerType(32),
    'const int': IntegerType(32),
    'uint': IntegerType(32),
    'unsigned int': IntegerType(32),
    'unsigned char': IntegerType(8),
    '__int64': IntegerType(64),
    }

def get_default_value(type):
  if type.is_float:
    num_elems = type.bitwidth // 32
    return float_vec_to_bits([1] * num_elems, float_size=32)
  elif type.is_double:
    num_elems = type.bitwidth // 64
    return float_vec_to_bits([1] * num_elems, float_size=64)
  return Bits(type.bitwidth)

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
  def __init__(self, var, lo_idx, hi_idx):
    self.var = var
    self.lo_idx = lo_idx
    self.hi_idx = hi_idx
    self.zero_extending = True

  def slice(self, lo, hi):
    new_slice = self.__class__(self.var, lo+self.lo_idx, hi+self.lo_idx)
    return new_slice

  def __repr__(self):
    return '%s[%d:%d]' % (self.var, self.lo_idx, self.hi_idx)

  def mark_sign_extend(self):
    self.zero_extending = False

  def update(self, rhs, env):
    '''
    rhs : integer
    '''
    val = env.get_value(self.var)
    bits = BitArray(uint=val.uint, length=val.length)
    bitwidth = env.get_type(self.var).bitwidth
    extend = zero_extend if self.zero_extending else sign_extend
    assert (self.lo_idx >= 0 and
        self.lo_idx < self.hi_idx and
        self.hi_idx < bitwidth)
    bits[self.lo_idx:self.hi_idx+1] = extend(rhs, self.hi_idx-self.lo_idx+1)
    new_val = Bits(uint=bits.uint, length=bits.length)
    env.set_value(self.var, new_val)

  def get_value(self, env):
    bitwidth = self.hi_idx - self.lo_idx + 1
    total_bitwidth = env.get_type(self.var).bitwidth
    val = env.get_value(self.var)[self.lo_idx:self.hi_idx+1]
    # restrict the bitwidth
    val = Bits(uint=val.uint, length=bitwidth)
    return val

def unreachable():
  assert False

def get_value(v, env):
  if isinstance(v, Slice):
    return v.get_value(env)
  return v

def binary_op(op, signed=True):
  def impl(a, b):
    bitwidth = a.length
    #return Bits(int=op(a.int, b.int), length=bitwidth)
    if signed:
      return Bits(int=op(a.int, b.int), length=64)
    else:
      return Bits(int=op(a.uint, b.uint), length=64)
  return impl

def binary_shift(op):
  return lambda a, b: op(a, b.int)

def unary_op(op, signed=True):
  def impl(a):
    bitwidth = a.length
    if signed:
      return Bits(int=op(a.int), length=bitwidth)
    else:
      return Bits(int=op(a.uint), length=bitwidth)
  return impl

def binary_float_op(op):
  def impl(a, b):
    bitwidth = a.length
    return Bits(float=op(a.float,b.float), length=bitwidth)
  return impl

# mapping <op, is_float?> -> impl
binary_op_impls = {
    ('+', True): binary_float_op(operator.add),
    ('-', True): binary_float_op(operator.sub),
    ('*', True): binary_float_op(operator.mul),
    ('/', True): binary_float_op(operator.truediv),
    ('<', True): binary_float_op(operator.lt),
    ('<=', True): binary_float_op(operator.le),
    ('>', True): binary_float_op(operator.gt),
    ('>=', True): binary_float_op(operator.ge),
    ('!=', True): binary_op(operator.ne),
    ('>>', True): binary_shift(operator.rshift),

    ('AND', True): binary_op(operator.and_),
    ('OR', True): binary_op(operator.or_, signed=False),
    ('XOR', True): binary_op(operator.xor, signed=False),
    ('==', True): binary_op(operator.eq),

    ('*', False): binary_op(operator.mul),
    ('+', False): binary_op(operator.add),
    ('-', False): binary_op(operator.sub),
    ('>', False): binary_op(operator.gt),
    ('<', False): binary_op(operator.gt),
    ('%', False): binary_op(operator.imod),
    ('<<', False): binary_shift(operator.lshift),
    ('>>', False): binary_shift(operator.rshift),
    ('AND', False): binary_op(operator.and_, signed=False),
    ('OR', False): binary_op(operator.or_, signed=False),
    ('XOR', False): binary_op(operator.xor, signed=False),
    ('==', False): binary_op(operator.eq),
    }

# mapping <op, is_float?> -> impl
unary_op_impls = {
    ('NOT', False): unary_op(operator.not_, signed=False),
    ('NOT', True): unary_op(operator.not_, signed=False),
    }


def int_to_bits(x, bitwidth=32):
  if x < 0:
    return Bits(int=x, length=bitwidth)
  return Bits(uint=x, length=bitwidth)

def float_to_bits(x, bitwidth=32):
  return Bits(float=x, length=bitwidth)

def float_vec_to_bits(vec, float_size=64):
  bitwidth = len(vec) * float_size
  bits = BitArray(uint=0, length=bitwidth)
  for i, x in enumerate(vec):
    bits[i*float_size:(i+1)*float_size] = float_to_bits(x, float_size)
  return Bits(uint=bits.uint, length=bitwidth)

def bits_to_float_vec(bits, float_size=64):
  vec = []
  for i in range(0, bits.length, float_size):
    vec.append(bits[i:i+float_size].float)
  return vec

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
    return Bits(float=op(slice_or_val.float), length=ty.bitwidth), ty
  return impl

def builtin_int_to_float(args, env):
  [(slice_or_val, ty)] = args
  return Bits(float=slice_or_val.int, length=32), FloatType(32)

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
    'Convert_Int32_To_FP32': builtin_int_to_float,
    'Convert_Int64_To_FP32': builtin_int_to_float,
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

    'APPROXIMATE': ignore,
    'MIN': builtin_binary_func(min),
    'MAX': builtin_binary_func(max),
    'ABS': builtin_unary_func(abs),
    'SQRT': builtin_unary_func(math.sqrt),
    'FLOOR': builtin_unary_func(math.floor),
    'CEIL': builtin_unary_func(math.ceil),
    }

def is_float(type):
  return type.is_float or type.is_double

# TODO: handle integer overflow here
def interpret_update(update, env):
  print(update)
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
  slice = Slice(var.name, 0, type.bitwidth-1)
  return slice, type

def is_number(expr):
  return type(expr) == Number

def interpret_binary_expr(expr, env):
  a, a_type = evaluate_expr(expr.a, env)
  b, b_type = evaluate_expr(expr.b, env)
  # automatically change bit widths 

  # TODO: make sure this crap is correct
  if a_type.bitwidth < 16:
    a = sign_extend(a, 16)
    a_type = a_type._replace(bitwidth=16)
  if b_type.bitwidth < 16:
    b = sign_extend(b, 16)
    b_type = b_type._replace(bitwidth=16)
  #assert a_type == b_type or is_number(expr.a) or is_number(expr.b) or expr.op in ('<<', '>>')

  impl_sig = expr.op, is_float(a_type)
  impl = binary_op_impls[impl_sig]
  return impl(a, b), a_type

def interpret_unary_expr(expr, env):
  a, a_type = evaluate_expr(expr.a, env)
  impl_sig = expr.op, is_float(a_type)
  impl = unary_op_impls[impl_sig]
  return impl(a), a_type

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
  if not is_float(ty):
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
      param_width = arg.length
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
    try:
      return int_to_bits(n.val), IntegerType(32)
    except CreationError:
      return int_to_bits(n.val, 64), IntegerType(64)
  return float_to_bits(n.val), FloatType(32)

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
    if type(stmt) == Return:
      assign_to_dst = Update(lhs=Var('dst'), rhs=stmt.val, modifier=None)
      interpret_stmt(assign_to_dst, env)
      break
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
  eager_map = lambda f, xs: list(map(f, xs))
  a = [1,2]
  b = [1,2]
  c = interpret(spec, eager_map(float_vec_to_bits, [a, b]))
  print(bits_to_float_vec(c))
  c = interpret(spec, eager_map(float_vec_to_bits, [[4,5], [7,8]]))
  print(bits_to_float_vec(c))
  c = interpret(spec, eager_map(float_vec_to_bits, [[2,3], [4,9]]))
  print(bits_to_float_vec(c))
