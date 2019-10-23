'''
Many operators are ambiguously defined in the manual.
For instance, '<' is used to compare both signed and unsigned bitvectors.
'''
from fuzzer import fuzz_intrinsic
import itertools

# these operators are ambiguously defined (so far as signedness is concerned)
configurable_op = {
    '<', '<=',
    '>', '>=',
    '>>',
    '*', '+', '-',
    }

def configure_spec(spec, num_tests=10, num_iters=32):
  '''
  given a spec, configure its binary operators so that
  the spec conforms with real-world behavior.

  return <success?>, <spec>
  '''
  ok, compiled = fuzz_intrinsic(spec, num_tests)
  if not compiled:
    # we don't have the groundtruth
    #  if we can't even compile code using this intrinsic
    return False, False, spec

  if ok: # already correct spec
    return True, True, spec

  configurable_exprs = [expr 
      for expr in spec.binary_exprs if expr.op in configurable_op]
  num_configs = len(configurable_exprs)
  config_space = itertools.product(*[(True, False) for _ in range(num_configs)])
  for i, encoded_config in enumerate(config_space):
    if i >= num_iters:
      break
    configs = {
        expr.expr_id: signedness 
        for expr, signedness in zip(configurable_exprs, encoded_config)
        }
    new_spec = spec._replace(configs=configs)
    ok, _ = fuzz_intrinsic(new_spec)
    if ok:
      return True, True, new_spec
  return False, True, spec

if __name__ == '__main__':
  import sys
  import xml.etree.ElementTree as ET
  from manual_parser import get_spec_from_xml

  sema = '''
  <intrinsic tech='SSSE3' vexEq='TRUE' rettype='__m128i' name='_mm_sign_epi8'>
        <type>Integer</type>
        <CPUID>SSSE3</CPUID>
        <category>Arithmetic</category>
        <parameter varname='a' type='__m128i'/>
        <parameter varname='b' type='__m128i'/>
        <description>Negate packed 8-bit integers in "a" when the corresponding signed 8-bit integer in "b" is negative, and store the results in "dst". Element in "dst" are zeroed out when the corresponding element in "b" is zero.</description>
        <operation>
FOR j := 0 to 15
        i := j*8
        IF b[i+7:i] &lt; 0
                dst[i+7:i] := -(a[i+7:i])
        ELSE IF b[i+7:i] == 0
                dst[i+7:i] := 0
        ELSE
                dst[i+7:i] := a[i+7:i]
        FI
ENDFOR
        </operation>
        <instruction name='psignb' form='xmm, xmm'/>
        <header>tmmintrin.h</header>
</intrinsic>
  '''

  intrin_node = ET.fromstring(sema)
  spec = get_spec_from_xml(intrin_node)
  ok, compiled, new_spec = configure_spec(spec, num_tests=100)
  print(ok, compiled)
