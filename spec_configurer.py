'''
Many operators are ambiguously defined in the manual.
For instance, '<' is used to compare both signed and unsigned bitvectors.
'''
from fuzzer import fuzz_intrinsic
import itertools

# these operators are ambiguously defined (so far as signedness is concerned)
configurable_op = {
    '<', '<=',
    '>', '>='
    }

def configure_spec(spec):
  '''
  given a spec, configure its binary operators so that
  the spec conforms with real-world behavior.

  return <success?>, <spec>
  '''
  ok, compiled = fuzz_intrinsic(spec)
  if not compiled:
    # if we can't even compile code using this intrinsic
    # we don't have groundtruth
    return False, spec

  if ok: # already correct spec
    return True, spec

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
      return True, new_spec
  return False, spec

if __name__ == '__main__':
  import sys
  import xml.etree.ElementTree as ET
  from manual_parser import get_spec_from_xml

  sema = '''
<intrinsic tech='MMX' rettype='__m64' name='_m_pcmpgtb'>
	<type>Integer</type>
	<CPUID>MMX</CPUID>
	<category>Compare</category>
	<parameter varname='a' type='__m64'/>
	<parameter varname='b' type='__m64'/>
	<description>Compare packed 8-bit integers in "a" and "b" for greater-than, and store the results in "dst".</description>
	<operation>
FOR j := 0 to 7
	i := j*8
	dst[i+7:i] := ( a[i+7:i] &gt; b[i+7:i] ) ? 0xFF : 0
ENDFOR
	</operation>
	<instruction name='pcmpgtb' form='mm, mm'/>
	<header>mmintrin.h</header>
</intrinsic>
  '''

  intrin_node = ET.fromstring(sema)
  spec = get_spec_from_xml(intrin_node)
  ok, new_spec = configure_spec(spec)
  print(ok)
