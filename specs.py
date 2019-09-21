import xml.etree.ElementTree as ET
from manual_parser import get_spec_from_xml

specs = {}

for intrin in ET.parse('data-latest.xml').iter('intrinsic'):
  try:
    spec = get_spec_from_xml(intrin)
    specs[spec.intrin] = spec
  except:
    continue
