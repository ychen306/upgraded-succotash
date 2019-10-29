'''
Code generators for each intrinsics/llvm scalar ops
'''

from collections import namedtuple
from llvm_sema import *
from specs import specs
import itertools

# mapping <instrinsic name> -> <inputs, outputs> -> <string representing computation>
expr_generators = {}

def get_type_name(bitwidth, signed):
  bitwidth = max(8, bitwidth)
  return '%sint%d_t' % ('' if signed else 'u', bitwidth)

def index_into(xs, i, ty):
  return '(({ty} *){xs})[{i}]'.format(ty=ty, xs=xs, i=i)

def address_of(x):
  return '&' + x;

def get_int_min(bitwidth):
  return { 8 : -128, 16: -32768, 32: -2147483648, 64: -9223372036854775807 }[bitwidth]

def get_binary_expr_generator(op_syntax, in_bitwidth, out_bitwidth, signed, is_division):
  def codegen(i, args, results, imm8=None):
    a, b = args
    [y] = results

    in_ty = get_type_name(in_bitwidth, signed=signed)
    out_ty = get_type_name(out_bitwidth, signed=signed)
    a = index_into(a, i, in_ty)
    b = index_into(b, i, in_ty)
    y = index_into(y, i, out_ty)

    guard_div_by_zero = ''
    if is_division and signed:
      guard_div_by_zero = 'if ({b} == 0 || ({a} == {int_min} && {b} == -1)) div_by_zero = 1; else'.format(
          a=a, b=b, int_min=get_int_min(in_bitwidth))
    elif is_division and not signed:
      guard_div_by_zero = 'if ({b} == 0) div_by_zero = 1; else'.format(b=b)
    return '{guard_div_by_zero} {y} = {a} {op} {b};'.format(
        guard_div_by_zero=guard_div_by_zero,
        op=op_syntax,
        a=a, b=b, y=y)
  return codegen

def get_select_generator(bitwidth):
  def codegen(i, args, results, imm8=None):
    c, a, b = args
    [y] = results

    c = index_into(c, i, get_type_name(1, signed=False))
    a = index_into(a, i, get_type_name(bitwidth, signed=False))
    b = index_into(b, i, get_type_name(bitwidth, signed=False))
    y = index_into(y, i, get_type_name(bitwidth, signed=False))

    return '{y} = ({c})?{a}:{b};'.format(y=y, c=c, a=a, b=b)
  return codegen

def get_sext_generator(in_bitwidth, out_bitwidth):
  def codegen(i, args, results, imm8=None):
    [x] = args
    [y] = results

    in_ty = get_type_name(in_bitwidth, signed=True)
    out_ty = get_type_name(out_bitwidth, signed=True)
    x = index_into(x, i, in_ty)
    y = index_into(y, i, out_ty)
    
    return '{y} = {x};'.format(x=x, y=y)
  return codegen

def get_zext_generator(in_bitwidth, out_bitwidth):
  def codegen(i, args, results, imm8=None):
    [x] = args
    [y] = results

    in_ty = get_type_name(in_bitwidth, signed=False)
    out_ty = get_type_name(out_bitwidth, signed=False)
    x = index_into(x, i, in_ty)
    y = index_into(y, i, out_ty)

    return '{y} = {x};'.format(x=x, y=y)
  return codegen

def get_trunc_generator(in_bitwidth, out_bitwidth):
  def codegen(i, args, results, imm8=None):
    [x] = args
    [y] = results

    in_ty = get_type_name(in_bitwidth, signed=False)
    out_ty = get_type_name(out_bitwidth, signed=False)
    x = index_into(x, i, in_ty)
    y = index_into(y, i, out_ty)

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
  return ParamId(is_input=False, idx=idx, is_constant=True)

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

  def codegen(i, args, results, imm8=None):
    params = []
    for param_id in param_ids:
      if param_id.is_constant:
        assert imm8 is not None
        params.append(imm8)
      elif param_id.is_input:
        param_type = in_param_types[param_id.idx]
        params.append(index_into(args[param_id.idx], i, param_type))
      else:
        param_type = out_param_types[param_idx.idx - (1 if has_explicit_retval else 0)]
        params.append(address_of(index_into(args[param_id.idx], i, param_type)))

    call = '%s(%s);' % (spec.intrin, ', '.join(params))
    if has_explicit_retval:
      call = '%s = %s' % (index_into(results[0], i, spec.rettype), call)
    return call

  return codegen

for inst, spec in specs.items():
  expr_generators[inst] = get_intrinsic_generator(spec)
