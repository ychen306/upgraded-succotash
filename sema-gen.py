import xml.etree.ElementTree as ET
from manual_parser import get_spec_from_xml
import sys
from interp import interpret

data_f = sys.argv[1]
data_root = ET.parse(data_f)

num_parsed = 0
num_skipped = 0
supported_insts = set()
skipped_insts = set()

for intrin in data_root.iter('intrinsic'):
  cpuid = intrin.find('CPUID')
  sema = intrin.find('operation') 
  inst = intrin.find('instruction')
  inst_form = None
  if inst is None:
    continue
  inst_form = inst.attrib['name'], inst.attrib.get('form')
  if not (intrin.attrib['name'].startswith('_mm') or
      intrin.attrib['name'].startswith('_mm')):
    continue
  if (intrin.attrib['name'].endswith('getcsr') or
      intrin.attrib['name'].endswith('setcsr')):
    continue
  if sema is not None and 'MEM' in sema.text:
    continue
  if 'str' in intrin.attrib['name']:
    if inst is not None:
      skipped_insts.add(inst_form)
    num_skipped += 1
    continue
  #if cpuid is not None and cpuid.text in ('MPX', 'KNCNI'):
  #  continue

  if 'fixup' in intrin.attrib['name']:
    if inst is not None:
      skipped_insts.add(inst_form)
    num_skipped += 1
    continue
  if 'round' in intrin.attrib['name']:
    if inst is not None:
      skipped_insts.add(inst_form)
    num_skipped += 1
    continue
  if 'prefetch' in intrin.attrib['name']:
    if inst is not None:
      skipped_insts.add(inst_form)
    num_skipped += 1
    continue

  print(intrin.attrib['name'])
  if inst is not None and sema is not None:
    try:
      #if 'ELSE IF' in sema.text:
      #  continue
      spec = get_spec_from_xml(intrin)
      interpret(spec)
      supported_insts.add(inst_form)
      num_parsed += 1
    except SyntaxError:
      print('Parsed', num_parsed, ' semantics, failling:')
      print(sema.text)
      print(intrin.attrib['name'])
      break
print('Parsed:', num_parsed,
    'Skipped:', num_skipped,
    'Num unique inst forms parsed:', len(supported_insts),
    'Num inst forms skipped:', len(skipped_insts)
    )
