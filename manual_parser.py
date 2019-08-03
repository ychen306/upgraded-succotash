from ast import Parameter, Spec
from parser import parser

def get_spec_from_xml(node):
  params = []
  for param_node in node.findall('parameter'):
    name = param_node.attrib['varname']
    type = param_node.attrib['type']
    if name == '':
      continue
    params.append(Parameter(name, type))
  intrin = node.attrib['name']
  inst = node.find('instruction')
  assert(inst is not None)
  operation = node.find('operation')
  assert(operation is not None)
  spec = parser.parse(operation.text)
  rettype = node.attrib['rettype']
  return Spec(
      intrin=intrin,
      inst=inst,
      spec=spec,
      params=params,
      rettype=rettype)
