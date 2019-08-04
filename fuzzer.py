from intrinsic_types import intrinsic_types, is_float
from bitstring import Bits, BitArray
import random
from tempfile import NamedTemporaryFile
import os
import subprocess
from interp import interpret
from bit_util import *
import math

'''
TODO: handle parameters named 'imm8..' specially
'''

src_path = os.path.dirname(os.path.abspath(__file__))

load_intrinsics = {
    '_m512i': '_mm512_loadu_si32',
    '__m512i': '_mm512_loadu_si32',
    '__m256i': '_mm256_loadu_si256',
    '__m128i': '_mm_loadu_si128',

    # single precision floats
    '__m512': '_mm512_loadu_ps',
    '__m256': '_mm256_loadu_ps',
    '__m128': '_mm_loadu_ps',
    '_m512': '_mm512_loadu_ps',
    '_m256': '_mm256_loadu_ps',
    '_m128': '_mm_loadu_ps',

    # double precision floats
    '__m512d': '_mm512_loadu_pd',
    '__m256d': '_mm256_loadu_pd',
    '__m128d': '_mm_loadu_pd',
    '_m512d': '_mm512_loadu_pd',
    '_m256d': '_mm256_loadu_pd',
    '_m128d': '_mm_loadu_pd',
    }

# functions that prints these vector registers
printers = {
    '_m512i': 'print__m512i',
    '__m512i': 'print__m512i',
    '__m256i': 'print__m256i',
    '__m128i': 'print__m128i',

    # single precision floats
    '__m512': 'print__m512',
    '__m256': 'print__m256',
    '__m128': 'print__m128',
    '_m512': 'print__m512',
    '_m256': 'print__m256',
    '_m128': 'print__m128',

    # double precision floats
    '__m512d': 'print__m512d',
    '__m256d': 'print__m256d',
    '__m128d': 'print__m128d',
    '_m512d': 'print__m512d',
    '_m256d': 'print__m256d',
    '_m128d': 'print__m128d',
    }

def emit_load(outf, dst, src, typename):
  if typename in load_intrinsics:
    load_intrinsic = load_intrinsics[typename]
    outf.write('%s %s = %s((%s *)&%s);\n' % (
      typename, dst, load_intrinsic, typename, src
      ))
  else:
    outf.write('%s %s = *(%s *)(&%s);\n' % (
      typename, dst, typename, src
      ))

def gen_rand_data(outf, name, typename):
  '''
  declare a variable of `data_type` called `name`

  print result to `outf` and return the actual bytes in bits

  e.g. for ty=__m512i, name = x1, we declare the following

  unsigned char x1[64] = { ... };
  '''

  if typename.endswith('*'):
    is_pointer = True
    typename = typename[:-1].strip()
  else:
    is_pointer = False

  ty = intrinsic_types[typename]

  # generate floats separates for integer because we don't
  #  want to deal with underflow and other floating point issues
  if not is_float(ty):
    num_bytes = ty.bitwidth // 8
    bytes = [random.randint(0, 255) for _ in range(num_bytes)]
    outf.write('unsigned char %s_aux[%d] = { %s };\n' % (
      name, num_bytes, ','.join(map(str, bytes))
      ))
    bits = BitArray(length=ty.bitwidth)
    for i, byte in enumerate(bytes):
      update_bits(bits, i*8, i*8+8, byte)
  else:
    float_size = 32 if ty.is_float else 64
    c_type = 'float' if ty.is_float else 'double'
    num_floats = ty.bitwidth // float_size
    floats = [random.random() for _ in range(num_floats)]
    outf.write('%s %s_aux[%d] = { %s };\n' % (
      c_type, name, num_floats, ','.join(map(str, floats))
      ))
    bits = float_vec_to_bits(floats, float_size=float_size)

  if not is_pointer:
    # in this case we need to load the bytes
    emit_load(outf, src='%s_aux'%name, dst=name, typename=typename)
  else:
    # in this case we just take the address
    outf.write('%s *%s = (%s *)(&%s_aux);\n' % (
        typename, name, typename, name 
      ))

  return bits

def emit_print(outf, var, typename):
  if typename.endswith('*'):
    typename = typename[:-1].strip()
    ty = intrinsic_types[typename]
    is_pointer = True
  else:
    is_pointer = False
    ty = intrinsic_types[typename]

  if is_pointer:
    # need to load the value first first
    tmp = get_temp_name()
    emit_load(outf, dst=tmp, src=var, typename=typename)
    var = tmp

  if typename in printers:
    # use the predefined printers
    printer = printers[typename]
    if ty.is_float:
      param_ty = 'float'
    elif ty.is_double:
      param_ty = 'double'
    else:
      param_ty = 'unsigned long'

    outf.write('%s((%s *)&%s);\n' % (printer, param_ty, var))
  else:
    if ty.is_float:
      outf.write('printf("%%f\\n", %s);\n' % var)
    elif ty.is_double:
      outf.write('printf("%%lf\\n", %s);\n' % var)
    else:
      outf.write('printf("%%lu\\n", (unsigned long)%s);\n' % var)


counter = 0
def get_temp_name():
  global counter
  counter += 1
  return 'tmp%d' % counter

def fuzz_intrinsic_once(outf, spec):
  '''
  1) generate test (in C) that exercises the intrinsic
  2) run the interpreter per the spec and return the expected output
  '''
  # generate random arguments
  c_vars = []
  arg_vals = []
  out_params = []
  out_param_types = []
  for param in spec.params:
    c_var = get_temp_name()
    arg_val = gen_rand_data(outf, c_var, param.type)
    c_vars.append(c_var)
    arg_vals.append(arg_val)
    if param.type.endswith('*'):
      out_params.append(c_var)
      out_param_types.append(param.type)

  # call the intrinsic
  has_return_val = spec.rettype != 'void'
  if has_return_val:
    ret_var = get_temp_name()
    outf.write('%s %s = %s(%s);\n' % (
      spec.rettype, ret_var, spec.intrin, ','.join(c_vars)
      ))
    # print the result
    emit_print(outf, ret_var, spec.rettype)
  else:
    outf.write('%%s(%s);\n' % (
      spec.intrin, ','.join(c_vars)
      ))

  out_types = []

  for param, param_type in zip(out_params, out_param_types):
    emit_print(outf, param, param_type)

  out, out_params = interpret(spec, arg_vals)
  if has_return_val:
    return [out] + out_params, [spec.rettype] + out_param_types
  return out_params, out_param_types

def get_err(a, b, is_float):
  err = a - b
  if not is_float:
    return err
  if math.isnan(a) and math.isnan(b):
    return 0
  return err

def identical_vecs(a, b, is_float):
  errs = [get_err(aa, bb, is_float)
      for aa,bb in zip(a, b)]
  if is_float:
    return all(abs(err) <= 1e-6 for err in errs)
  return all(err == 0 for err in errs)

def bits_to_vec(bits, typename):
  if typename.endswith('*'):
    ty = intrinsic_types[typename[:-1]]
  else:
    ty = intrinsic_types[typename]
  if ty.is_float:
    return bits_to_float_vec(bits, float_size=32)
  elif ty.is_double:
    return bits_to_float_vec(bits, float_size=64)

  # integer type
  return bits_to_long_vec(bits)

# TODO: make this return True if correct
def fuzz_intrinsic(spec, num_tests=10):
  interpreted = []
  with NamedTemporaryFile(suffix='.c', mode='w') as outf, NamedTemporaryFile(delete=False) as exe:
    outf.write('''
#include <emmintrin.h> 
#include <immintrin.h>
#include <nmmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <wmmintrin.h>
#include <xmmintrin.h>

#include <stdio.h>
#include "printers.h"

#define __int64_t __int64;
#define __int64 long long

int main() {
        ''')
    
    for _ in range(num_tests):
      interpreted.append(fuzz_intrinsic_once(outf, spec))

    outf.write('''
  return 0;
}
      ''')
    outf.flush()

    # TODO: add CPUIDs 
    try:
      subprocess.check_output(
          'cc %s -o %s -I%s %s/printers.o >/dev/null 2>/dev/null -mavx -mavx2 -mfma' % (
            outf.name, exe.name, src_path, src_path),
          shell=True)
    except subprocess.CalledProcessError:
      return False

    num_outputs_per_intrinsic = len(interpreted[0][0])

    stdout = subprocess.check_output([exe.name])
    lines = stdout.decode('utf-8').strip().split('\n')
    assert(len(lines) == len(interpreted) * num_outputs_per_intrinsic)

  os.system('rm '+exe.name)

  for i in range(0, len(lines), num_outputs_per_intrinsic):
    outputs, output_types = interpreted[i // num_outputs_per_intrinsic]
    for output, output_typename, line in zip(outputs, output_types, lines[i:i+num_outputs_per_intrinsic]):
      fields = line.strip().split()
      output_is_float = is_float(intrinsic_types[output_typename])
      if output_is_float:
        ref_vec = [float(x) for x in fields]
      else:
        ref_vec = [int(x) for x in fields]
      vec = bits_to_vec(output, output_typename)
      if not identical_vecs(vec, ref_vec, is_float):
        return False
  return True

if __name__ == '__main__':
  import sys
  import xml.etree.ElementTree as ET
  from manual_parser import get_spec_from_xml

  sema = '''
<intrinsic tech='Other' rettype='__m512i' name='_mm512_mask_gf2p8affine_epi64_epi8'>
	<type>Integer</type>
	<CPUID>GFNI</CPUID>
	<CPUID>AVX512F</CPUID>
	<category>Arithmetic</category>
	<parameter varname='src' type='__m512i'/>
	<parameter varname='k' type='__mmask64'/>
	<parameter varname='x' type='__m512i'/>
	<parameter varname='A' type='__m512i'/>
	<parameter varname='b' type='int'/>
	<description>
		Compute an affine transformation in the Galois Field 2^8. An affine transformation is defined by "A" * "x" + "b", where "A" represents an 8 by 8 bit matrix, "x" represents an 8-bit vector, and "b" is a constant immediate byte. Store the packed 8-bit results in "dst" using writemask "k" (elements are copied from "src" when the corresponding mask bit is not set).
	</description>
	<operation>
DEFINE parity(x) {
	t := 0
	FOR i := 0 to 7
		t := t XOR x.bit[i]
	ENDFOR
	RETURN t
}
DEFINE affine_byte(tsrc2qw, src1byte, imm8) {
	FOR i := 0 to 7
		retbyte.bit[i] := parity(tsrc2qw.byte[7-i] AND src1byte) XOR imm8.bit[i]
	ENDFOR
	RETURN retbyte
}
FOR j := 0 TO 7
	FOR bb := 0 to 7
		IF k[j*8+bb]
			dst.qword[j].byte[bb] := affine_byte(A.qword[j], x.qword[j].byte[bb], b)
		ELSE
			dst.qword[j].byte[bb] := src.qword[j].byte[bb]
		FI
	ENDFOR
ENDFOR
dst[MAX:512] := 0
	</operation>
	
	<instruction name='VGF2P8AFFINEQB' form='zmm {k}, zmm, zmm, imm8' xed=''/>
	<header>immintrin.h</header>
</intrinsic>
  '''
  intrin_node = ET.fromstring(sema)
  spec = get_spec_from_xml(intrin_node)
  ok = fuzz_intrinsic(spec)
  print(ok)
