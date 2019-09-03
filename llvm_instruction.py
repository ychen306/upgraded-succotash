from collections import namedtuple

Instruction = namedtuple('Instruction', ['op', 'bitwidth', 'vectorlen', 'op_a', 'op_b', 'op_c', 'init_list'])
