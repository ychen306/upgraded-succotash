'''
Code generators for each intrinsics/llvm scalar ops
'''

from collections import namedtuple
from llvm_sema import *
from specs import specs
import itertools

# mapping <instrinsic name> -> <inputs, outputs> -> <string representing computation>
expr_generators = {}

include_bv_ops = True

def get_type_name(bitwidth, signed):
  if bitwidth < 8:
    assert bitwidth == 1
    bitwidth = 32
  return '%sint%d_t' % ('' if signed else 'u', bitwidth)

def index_into(xs, i, ty):
  return '(({ty} *){xs})[{i}]'.format(ty=ty, xs=xs, i=i)

def address_of(x):
  return '&' + x;

if include_bv_ops:
  uint64_t = 'uint64_t'
  uint32_t = 'uint32_t'

  def ehad(i, args, results, *_):
    [x] = args
    [y] = results
    return '{y} = {x} >> 1;'.format(
        x=index_into(x, i, uint64_t), y=index_into(y, i, uint64_t))

  def arba(i, args, results, *_):
    [x] = args
    [y] = results
    return '{y} = {x} >> 4;'.format(
        x=index_into(x, i, uint64_t), y=index_into(y, i, uint64_t))

  def shesh(i, args, results, *_):
    [x] = args
    [y] = results
    return '{y} = {x} >> 16;'.format(
        x=index_into(x, i, uint64_t), y=index_into(y, i, uint64_t))

  def smol(i, args, results, *_):
    [x] = args
    [y] = results
    return '{y} = {x} << 1;'.format(
        x=index_into(x, i, uint64_t), y=index_into(y, i, uint64_t))

  def smol(i, args, results, *_):
    [x] = args
    [y] = results
    return '{y} = {x} << 1;'.format(
        x=index_into(x, i, uint64_t), y=index_into(y, i, uint64_t))

  def im(i, args, results, *_):
    x, y, z = args
    [out] = results
    return '{out} = ({x} == 1) ? {y} : {z};'.format(
        out=index_into(out, i, uint64_t),
        x=index_into(x,i,uint64_t), y=index_into(y,i,uint64_t),
        z=index_into(z,i,uint64_t))

  def bvnot(i, args, results, *_):
    [x] = args
    [y] = results
    return '{y} = ~{x};'.format(
        x=index_into(x, i, uint64_t), y=index_into(y, i, uint64_t))

  def bvnot32(i, args, results, *_):
    [x] = args
    [y] = results
    return '{y} = ~{x};'.format(
        x=index_into(x, i, uint32_t), y=index_into(y, i, uint32_t))

  def bvneg(i, args, results, *_):
    [x] = args
    [y] = results
    return '{y} = -{x};'.format(
        x=index_into(x, i, uint32_t), y=index_into(y, i, uint32_t))

  expr_generators['ehad'] = ehad
  expr_generators['arba'] = arba
  expr_generators['shesh'] = shesh
  expr_generators['smol'] = smol
  expr_generators['im'] = im
  expr_generators['bvnot'] = bvnot
  expr_generators['bvnot32'] = bvnot32
  expr_generators['bvneg'] = bvneg


def get_int_min(bitwidth):
  return { 8 : -128, 16: -32768, 32: -2147483648, 64: -9223372036854775808 }[bitwidth]

def get_binary_expr_generator(op_syntax, in_bitwidth, out_bitwidth, signed, is_division):
  def codegen(i, args, results, imm8=None, using_gpu=False):
    a, b = args
    [y] = results

    in_ty = get_type_name(in_bitwidth, signed=signed)
    out_ty = get_type_name(out_bitwidth, signed=signed)
    a = index_into(a, i, in_ty)
    b = index_into(b, i, in_ty)
    y = index_into(y, '0' if using_gpu else i, out_ty)

    if using_gpu:
      handle_div_by_zero = 'return'
    else: 
      handle_div_by_zero = 'return 1'

    guard_div_by_zero = ''
    if is_division and signed:
      guard_div_by_zero = 'if ({b} == 0 || ({a} == {int_min} && {b} == -1)) {handle_div_by_zero}; else'.format(
          a=a, b=b, int_min=get_int_min(in_bitwidth), handle_div_by_zero=handle_div_by_zero)
    elif is_division and not signed:
      guard_div_by_zero = 'if ({b} == 0) {handle_div_by_zero}; else'.format(b=b, handle_div_by_zero=handle_div_by_zero)
    return '{guard_div_by_zero} {y} = {a} {op} {b};'.format(
        guard_div_by_zero=guard_div_by_zero,
        op=op_syntax,
        a=a, b=b, y=y)
  return codegen

def get_select_generator(bitwidth):
  def codegen(i, args, results, imm8=None, using_gpu=False):
    c, a, b = args
    [y] = results

    c = index_into(c, i, get_type_name(1, signed=False))
    a = index_into(a, i, get_type_name(bitwidth, signed=False))
    b = index_into(b, i, get_type_name(bitwidth, signed=False))
    y = index_into(y, '0' if using_gpu else i, get_type_name(bitwidth, signed=False))

    return '{y} = ({c})?{a}:{b};'.format(y=y, c=c, a=a, b=b)
  return codegen

def get_sext_generator(in_bitwidth, out_bitwidth):
  def codegen(i, args, results, imm8=None, using_gpu=False):
    [x] = args
    [y] = results

    in_ty = get_type_name(in_bitwidth, signed=True)
    out_ty = get_type_name(out_bitwidth, signed=True)
    x = index_into(x, i, in_ty)
    y = index_into(y, '0' if using_gpu else i, out_ty)
    
    return '{y} = {x};'.format(x=x, y=y)
  return codegen

def get_zext_generator(in_bitwidth, out_bitwidth):
  def codegen(i, args, results, imm8=None, using_gpu=False):
    [x] = args
    [y] = results

    in_ty = get_type_name(in_bitwidth, signed=False)
    out_ty = get_type_name(out_bitwidth, signed=False)
    x = index_into(x, i, in_ty)
    y = index_into(y, '0' if using_gpu else i, out_ty)

    return '{y} = {x};'.format(x=x, y=y)
  return codegen

def get_trunc_generator(in_bitwidth, out_bitwidth):
  def codegen(i, args, results, imm8=None, using_gpu=False):
    [x] = args
    [y] = results

    in_ty = get_type_name(in_bitwidth, signed=False)
    out_ty = get_type_name(out_bitwidth, signed=False)
    x = index_into(x, i, in_ty)
    y = index_into(y, '0' if using_gpu else i, out_ty)

    return '{y} = {x};'.format(x=x, y=y)
  return codegen

bitwidths = [1, 8, 16, 32, 64]
# TODO include llvm floating point operations
# Binary operations
for op, impl in binary_ops.items():
  for bw in bitwidths:
    if bw == 1:
      continue
    op_name = get_llvm_op_name(op, bw)
    x = z3.BitVec(op_name + '_x', bw)
    y = z3.BitVec(op_name + '_y', bw)
    out_bw = bw
    if op in comparisons:
      out_bw = 1
    expr_generators[op_name] = get_binary_expr_generator(binary_syntaxes[op], bw, out_bw, op in signed_binary_ops, op in divisions)

## Select
for bw in bitwidths:
  if bw == 1:
    continue
  op_name = get_llvm_op_name('Select', bw)
  expr_generators[op_name] = get_select_generator(bw)

# get generators for Trunc/SExt/ZExt
for bw_in, bw_out in itertools.product(bitwidths, bitwidths):
  if bw_in == bw_out:
    continue 
  if bw_in < bw_out:
    sext_name = get_sext_name(bw_in, bw_out)
    zext_name = get_zext_name(bw_in, bw_out)
    if bw_in > 1:
      expr_generators[sext_name] = get_sext_generator(bw_in, bw_out)
    expr_generators[zext_name] = get_zext_generator(bw_in, bw_out)
  elif bw_out > 1:
    trunc_name = get_trunc_name(bw_in, bw_out)
    expr_generators[trunc_name] = get_trunc_generator(bw_in, bw_out)

ParamId = namedtuple('ParamId', ['is_input', 'idx', 'is_constant'])

def get_imm8_id():
  return ParamId(is_input=False, idx=None, is_constant=True)

def get_in_param_id(idx):
  return ParamId(is_input=True, idx=idx, is_constant=False)

def get_out_param_id(idx):
  return ParamId(is_input=False, idx=idx, is_constant=False)

def get_intrinsic_generator(spec):
  # mapping each input or output to its position in the intrinsic params
  param_ids = []

  input_types = []
  output_sizes = []
  inst_form = spec.inst_form.split(', ')
  no_imm8 = 'imm8' not in (param.name for param in spec.params)

  # C typenames for the parameters
  out_param_types = []
  in_param_types = []

  has_explicit_retval = spec.rettype != 'void'

  for i, param in enumerate(spec.params):
    if ((no_imm8 and i < len(inst_form) and inst_form[i] == 'imm') or
        param.name == 'imm8'):
      param_ids.append(get_imm8_id())
      continue

    if param.type.endswith('*'):
      idx = len(out_param_types) + (1 if has_explicit_retval else 0)
      param_id = get_out_param_id(idx)
      param_ids.append(param_id)
      out_param_types.append(param.type[:-1].strip())
    else:
      idx = len(in_param_types)
      param_id = get_in_param_id(idx)
      param_ids.append(param_id)
      in_param_types.append(param.type.strip())

  def codegen(i, args, results, imm8=None, using_gpu=False):
    params = []
    for param_id in param_ids:
      if param_id.is_constant:
        assert imm8 is not None
        params.append(imm8)
      elif param_id.is_input:
        param_type = in_param_types[param_id.idx]
        params.append(index_into(args[param_id.idx], i, param_type))
      else:
        param_type = out_param_types[param_id.idx - (1 if has_explicit_retval else 0)]
        params.append(address_of(index_into(args[param_id.idx], '0' if using_gpu else i, param_type)))

    call = '%s(%s);' % (spec.intrin, ', '.join(params))
    if has_explicit_retval:
      call = '%s = %s' % (index_into(results[0], '0' if using_gpu else i, spec.rettype), call)
    return call

  return codegen

for inst, spec in specs.items():
  expr_generators[inst] = get_intrinsic_generator(spec)
