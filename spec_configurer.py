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
    '*', '+', '-',
    }

def configure_spec(spec):
  '''
  given a spec, configure its binary operators so that
  the spec conforms with real-world behavior.

  return <success?>, <spec>
  '''
  ok, compiled = fuzz_intrinsic(spec)
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
  for encoded_config in config_space:
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
<intrinsic tech='MMX' rettype='__m64' name='_m_psubusb'>
	<type>Floating Point</type>
	<type>Integer</type>
	<CPUID>MMX</CPUID>
	<category>Arithmetic</category>
	<parameter varname='a' type='__m64'/>
	<parameter varname='b' type='__m64'/>
	<description>Subtract packed unsigned 8-bit integers in "b" from packed unsigned 8-bit integers in "a" using saturation, and store the results in "dst".</description>
	<operation>
FOR j := 0 to 7
	i := j*8
	dst[i+7:i] := Saturate_To_UnsignedInt8(a[i+7:i] - b[i+7:i])	
ENDFOR
	</operation>
	<instruction name='psubusb' form='mm, mm'/>
	<header>mmintrin.h</header>
</intrinsic>
  '''

  intrin_node = ET.fromstring(sema)
  spec = get_spec_from_xml(intrin_node)
  ok, compiled, new_spec = configure_spec(spec)
  print(ok, compiled)
