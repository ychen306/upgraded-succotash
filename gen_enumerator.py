from codegen import expr_generators
from collections import namedtuple, defaultdict
import itertools
import io
from expr_sampler import sigs
import sys
import pynauty
from copy import copy

import z3
import json

ConcreteInst = namedtuple('ConcreteInst', ['name', 'imm8'])
Variable = namedtuple('Variable', ['name', 'bitwidth'])
Constant = namedtuple('Constant', ['value', 'bitwidth'])
InstNode = namedtuple('InstNode', ['level', 'sig', 'args'])
LiveInNode = namedtuple('LiveInNode', ['x'])
InstSignature = namedtuple(
    'InstSignature',
    ['inputs', 'outputs', 'commutative_pair'])

with open('commutative-params.json') as f:
  commutative_params = json.load(f)

def bits2bytes(bits):
  if bits < 8:
    assert bits == 1
    return 4
  return bits // 8

def emit_includes(out):
  out.write('#include <string.h>\n') # memcmp
  out.write('#include <immintrin.h>\n') # duh
  out.write('#include <stdio.h>\n') # debug
  out.write('#include <stdint.h>\n')
  out.write('#define __int64_t __int64\n')
  out.write('#define __int64 long long\n')

def get_inst_runner_name(inst):
  return 'run_{inst}_{imm8}'.format(
      inst=inst.name,
      imm8=str(inst.imm8) if inst.imm8 else '0')

def emit_func_table(table_name, sig, insts):
  tmpl = 'static int (*{name}[{num_insts}])({param_list}) = {{ {funcs} }};\n'
  params = ['int'] # num tests
  for _ in range(len(sig.inputs) + len(sig.outputs)):
    params.append('char *__restrict__')
  return tmpl.format(
      name=table_name,
      num_insts=len(insts),
      param_list=', '.join(params),
      funcs=', '.join(get_inst_runner_name(inst) for inst in insts))

class FuncTableSet:
  def __init__(self, sig2insts):
    # mapping <sig> -> <table name>, <table defn>
    self.tables = {}

    for i, (sig, insts) in enumerate(sig2insts.items()):
      table_name = 'table_%d' % i
      self.tables[sig] = table_name, emit_func_table(table_name, sig, insts)

  def get_table_def(self, sig):
    _, defn = self.tables[sig]
    return defn

  def get_table(self, sig):
    table_name, _ = self.tables[sig]
    return table_name

def declare_buffer(name, bitwidth, size):
  return 'static char %s[%d] __attribute__ ((aligned (64)));\n' % (
      name, bits2bytes(bitwidth)*size)

def get_livein_buf_name(x):
  return 'buf_input_%s' % x.name

def get_const_buf_name(c):
  return 'buf_const_%d_%d' % (c.value, c.bitwidth)

class BufferAllocator:
  def __init__(self, graphs, liveins, constants, max_tests=128):
    self.max_tests = max_tests
    # mapping <bitwidth> -> <maximum number of buffer required for the bitwidth>
    bitwidths = defaultdict(int)
    bitwidths_max = defaultdict(int)
    for g in graphs.values():
      bitwidths = defaultdict(int)
      for node in g:
        bw = node.sig.outputs[0]
        bitwidths[bw] += 1
      for bw, count in bitwidths.items():
        bitwidths_max[bw] = max(bitwidths_max[bw], count)

    # mapping <bitwidth> -> [<buf name>, <buf defn>]
    self.buffers = defaultdict(list)
    for bw, count in bitwidths_max.items():
      for i in range(count):
        buf_name = 'buf_%d_%d' % (bw, i)
        buf_def = declare_buffer(buf_name, bw, max_tests)
        self.buffers[bw].append((buf_name, buf_def))

    self.livein_bufs = {}
    for x in liveins:
      buf_name = get_livein_buf_name(x)
      buf_def = declare_buffer(buf_name, x.bitwidth, max_tests)
      self.livein_bufs[x] = buf_name, buf_def

    self.const_bufs = {}
    for c in constants:
      buf_name = get_const_buf_name(c)
      buf_def = declare_buffer(buf_name, c.bitwidth, max_tests)
      self.const_bufs[c] = buf_name, buf_def

  def reset(self):
    self.counters = defaultdict(int)

  def allocate(self, bitwidth):
    i = self.counters[bitwidth]
    self.counters[bitwidth] += 1
    buf, _ = self.buffers[bitwidth][i]
    return buf

  def get_livein_buf(self, x):
    buf_name, _ = self.livein_bufs[x]
    return buf_name

  def get_constant_buf(self, c):
    buf_name, _ = self.const_bufs[c]
    return buf_name

  def declare_buffers(self, out):
    for buffers in self.buffers.values():
      for _, buf_def in buffers:
        out.write(buf_def)
    for _, buf_def in self.livein_bufs.values():
      out.write(buf_def)
    for _, buf_def in self.const_bufs.values():
      out.write(buf_def)

class SolutionHandler:
  def __init__(self, name, g, sig2insts):
    self.g = g
    self.name = name
    iterator_name = lambda i : 'op_%d' % i
    with io.StringIO() as buf:
      decl = 'static void %s(%s)' % (
          name, 
          ', '.join('int %s' % iterator_name(i) for i in range(len(g))))
      # first tell the compiler not to inline the function
      buf.write('%s __attribute__((noinline));\n' % decl)
      # now define the handler
      buf.write('%s {\n' % decl)
      buf.write('printf("\\n");\n')

      for i, node in enumerate(g):
        op_it = iterator_name(i)

        # print the op
        buf.write('switch (%s) {\n' % op_it)
        for j, inst in enumerate(sig2insts[node.sig]):
          buf.write(
              'case {id}: printf("y{level} = {name} (imm8={imm8}) "); break;\n' 
              .format(
                id=j, 
                level=node.level,
                name=inst.name,
                imm8=inst.imm8))
        buf.write('}\n') # close the switch

        # print the args
        args = []
        for arg in node.args:
          if type(arg) == LiveInNode:
            v, _ = arg.x
            arg_name = str(v)
          else:
            arg_name = 'y%d' % arg.level
          args.append(arg_name)
        buf.write('printf("%s\\n");\n' % ', '.join(args))

      buf.write('}\n') # close the function

      self.defn = buf.getvalue()

  def declare(self, out):
    out.write(self.defn)

  def handle(self, iterators):
    return '%s(%s);\n' % (self.name, ', '.join(iterators))

def prune_enum_history(enum_history, cert2graph, leaf_graphs): 
  '''
  remove nodes in `enum_history` that do not lead to graphs
  '''
  inverted_history = defaultdict(list)
  for v, children in enum_history.items():
    for w in children:
      inverted_history[w].append(v)

  alive_certs = set()
  def visit(cert):
    if cert in alive_certs:
      return
    alive_certs.add(cert)
    for parent_cert in inverted_history.get(cert, []):
      visit(parent_cert)

  for cert in leaf_graphs.keys():
    visit(cert)

  dead_certs = set(cert2graph.keys()) - alive_certs
  for cert in dead_certs:
    if cert in enum_history:
      del enum_history[cert]
    del cert2graph[cert]

  for cert in leaf_graphs:
    assert cert in cert2graph

def emit_enumerator(target_size, graphs, 
    sig2insts, liveins, constants, 
    enum_history, cert2graph,
    out, max_tests=128):
  func_tables = FuncTableSet(sig2insts)
  allocator = BufferAllocator(graphs, liveins, constants)
  soln_handlers = {
      cert : SolutionHandler('handler_%d' % i, g, sig2insts)
      for i, (cert, g) in enumerate(graphs.items())
      }

  # declare the handler functions
  for handler in soln_handlers.values():
    handler.declare(out)

  # declare the tables
  for sig in sig2insts.keys():
    out.write(func_tables.get_table_def(sig))

  # declare the buffers
  allocator.declare_buffers(out)
  out.write(declare_buffer('target', target_size, max_tests))

  # mapping node -> buffer
  base_buffers = {}
  for x in liveins:
    base_buffers[LiveInNode(x)] = allocator.get_livein_buf(x)
  for c in constants:
    base_buffers[LiveInNode(c)] = allocator.get_constant_buf(c)

  # mapping cert -> enumerator
  enumerators = {
      cert : ('enumerator_%d' % i)
      for i, cert in enumerate(cert2graph.keys()) 
      }
  max_level = max(len(g) for g in graphs.values())
  # declare the enumerators
  for cert in cert2graph.keys():
    out.write('static void %s(%s);\n' % (
      enumerators[cert], ', '.join(itertools.repeat('int', max_level+1))))

  # params of the enumerators
  params = ['op_%d' % i for i in range(max_level)]

  out.write('static unsigned long long num_enumerated = 0;\n')

  # set of certificates that has a callee
  called = set()

  root_cert = None

  # generate the functions that mirror the search tree
  for cert, g in cert2graph.items():
    out.write('static void {enumerator}(int num_tests, {params}) {{\n'
        .format(
          enumerator=enumerators[cert],
          params=', '.join('int '+p for p in params)))
    buffers = dict(base_buffers)

    # figure out which buffer we should use
    allocator.reset()
    for node in g[:-1]:
      # select a buffer to store the result of running the selected instruction
      out_bitwidth = node.sig.outputs[0]
      out_buf = allocator.allocate(out_bitwidth)
      buffers[node] = out_buf

    num_nodes = len(g)
    if num_nodes > 0:
      node = g[-1]
      out.write('for (int op_mine = 0; op_mine < {num_insts}; op_mine++) {{\n'
        .format(num_insts=len(sig2insts[node.sig])))

      # select arguments for the selected instruction
      args = [buffers[arg] for arg in node.args]

      # select a buffer to store the result of running the selected instruction
      out_bitwidth = node.sig.outputs[0]
      out_buf = allocator.allocate(out_bitwidth)

      # run it
      arg_list = ['num_tests'] + args + [out_buf]
      func_table = func_tables.get_table(node.sig)
      out.write('int div_by_zero = {funcs}[op_mine]({args});\n'.format(
        funcs=func_table,
        args=', '.join(arg_list)))

      # skip if divide by zero
      out.write('if (__builtin_expect(div_by_zero, 0)) { num_enumerated++; continue; }\n')
    else:
      if root_cert is not None:
        print(g)
      assert root_cert is None
      root_cert = cert

    iterators = list(params)
    if num_nodes > 0:
      iterators[num_nodes-1] = 'op_mine'
    # enumerate the children nodes
    for next_cert in enum_history.get(cert, []):
      # if next_cert is not in cert2graph then it's dead
      if next_cert not in cert2graph or next_cert in called:
        continue
      called.add(next_cert)
      out.write('{enumerator}(num_tests, {args});\n'
          .format(
            enumerator=enumerators[next_cert],
            args=', '.join(iterators)))

    if cert in soln_handlers:
      out.write('if (bcmp(target, {out_buf}, {size}*num_tests) == 0)\n'
          .format(
            out_buf=out_buf,
            size=bits2bytes(target_size)))
      # check solution
      out.write(soln_handlers[cert].handle(iterators))
      out.write('num_enumerated ++;\n')

    if num_nodes > 0:
      out.write('}\n') # close the loop
    out.write('}\n') # close the function

  out.write('static void enumerate(int num_tests) {\n')
  out.write('%s(num_tests, %s);\n' % (
    enumerators[root_cert], ', '.join(itertools.repeat('0', max_level))))
  out.write('}\n') # close enumerate

class NodeIdManager:
  def __init__(self, liveins, bitwidths, params):
    self.liveins = list(liveins)
    self.cumsum_bw = {}
    self.cumsum_param = {}
    self.partition = [{self.live_in_id(x) for x in liveins}]

    s = len(self.liveins)
    for bw, count in bitwidths.items():
      self.cumsum_bw[bw] = s
      self.partition.append(set(range(s, s+count)))
      s += count

    for param_id, count in params.items():
      self.cumsum_param[param_id] = s
      self.partition.append(set(range(s, s+count)))
      s += count

    self.num_nodes = s

  def live_in_id(self, livein):
    return self.liveins.index(livein)

  def new_node_id(self, bitwidth):
    id = self.cumsum_bw[bitwidth]
    self.cumsum_bw[bitwidth] += 1
    return id

  def new_param_id(self, param_id):
    id = self.cumsum_param[param_id]
    self.cumsum_param[param_id] += 1
    return id

def classify_insts(insts):
  sig2insts = defaultdict(list)

  for inst in insts:
    input_types, out_sig = sigs[inst.name]
    comm_pairs = commutative_params.get(inst.name, [])
    comm_pair = None
    if len(comm_pairs) >= 1:
      comm_pair = tuple(comm_pairs[0])
    # don't consider imm8
    input_sig = tuple(
        ty.bitwidth for ty in input_types if not ty.is_constant)
    sig = InstSignature(input_sig, out_sig, comm_pair)
    sig2insts[sig].append(inst)

  return sig2insts

def enumerate_graphs(
  target_size, liveins, sig2insts, num_levels, constants):
  '''
  Given non-isomorphic insturctions
    * modulo distinct instructions
    * BUT with consideration to instruction signatures
  '''
  def get_node_id(node):
    if type(node) == LiveInNode:
      return id_manager.live_in_id(node.x)
    node_bw = node.sig.outputs[0]
    node_id = id_manager.inst_id(node.level, node_bw)
    return node_id

  def populate_adj(node, adj):
    if type(node) == LiveInNode:
      return

    node_id = get_node_id(node)
    assert node_id not in adj
    adj[node_id] = []
    comm_pair = node.sig.commutative_pair
    for i, used in enumerate(node.args):
      j = i
      if comm_pair is not None and comm_pair[1] == i:
        i == comm_pair[0]
      used_id = get_node_id(used)
      use_id = id_manager.param_id(node.level, i, j)
      adj[node_id].append(use_id)
      adj[use_id] = [used_id]

  def compute_certificate(g):
    # mapping <bitwidth> -> <number of instruction with the bw>
    bitwidths = defaultdict(int)
    # mapping <param ids> -> <number of occurence>
    params = defaultdict(int)

    for node in g:
      bw = node.sig.outputs[0]
      bitwidths[bw] += 1

      comm_pair = node.sig.commutative_pair
      for param_id, used in enumerate(node.args):
        if comm_pair is not None and comm_pair[1] == param_id:
          param_id == comm_pair[0]
        params[param_id] += 1

    id_manager = NodeIdManager(liveins+constants, bitwidths, params)
    # level -> <node id>
    node_ids = {}
    def get_node_id(node):
      if type(node) == LiveInNode:
        return id_manager.live_in_id(node.x)
      return node_ids[node.level]

    adj = {}
    for node in g:
      assert node.level not in node_ids
      bw = node.sig.outputs[0]
      node_id = id_manager.new_node_id(bw)
      node_ids[node.level] = node_id
      uses = adj[node_id] = []
      for param_id, used in enumerate(node.args):
        if comm_pair is not None and comm_pair[1] == param_id:
          param_id == comm_pair[0]
        use_id = id_manager.new_param_id(param_id)
        uses.append(use_id)
        adj[use_id] = [get_node_id(used)]

    nauty_graph = pynauty.Graph(
        id_manager.num_nodes, 
        directed=True,
        adjacency_dict=adj,
        vertex_coloring=id_manager.partition)

    bw_cert = tuple(sorted(bitwidths.items()))
    param_cert = tuple(sorted(params.items()))
    graph_cert = pynauty.certificate(nauty_graph)
    cert = bw_cert, param_cert, graph_cert
    return cert

  # mapping <cert of parent> -> <certs of children>
  enum_history = defaultdict(list)
  cert2graph = {}
  empty_g = []
  empty_cert = compute_certificate(empty_g)
  cert2graph[empty_cert] = empty_g
  def enum(level):
    enumerated = set()

    if level == 0:
      base = [(empty_g, empty_cert)]
    else:
      base = enum(level-1)

    # extend from the base set of non-isomorphic graphs
    for g, base_cert in base:
      # mapping bitwidth -> list of values of the bitwidth
      available_bitwidths = defaultdict(list)
      for x in (liveins+constants):
        available_bitwidths[x.bitwidth].append(LiveInNode(x))
      for node in g:
        bw = node.sig.outputs[0]
        available_bitwidths[bw].append(node)

      for sig in sig2insts.keys():
        # check if we can use instructions of this signature
        usable = all(bw in available_bitwidths for bw in sig.inputs)
        if not usable:
          continue

        arg_configs = [available_bitwidths[bw] for bw in sig.inputs]
        for args in itertools.product(*arg_configs):
          assert len(args) == len(sig.inputs)
          new_node = InstNode(level, sig, args)
          g_extended = g + [new_node]
          cert = compute_certificate(g_extended)
          if cert not in enumerated:
            enumerated.add(cert)
            enum_history[base_cert].append(cert)
            cert2graph[cert] = g_extended
            yield g_extended, cert

  def is_valid_graph(g):
    # the graph is valid only if
    # 1) there is a single sink
    # 2) sink has the target size
    if g[-1].sig.outputs[0] != target_size:
      return False

    found_component = False
    visited = set()
    def visit(node):
      if node.level in visited:
        return
      visited.add(node.level)
      for used in node.args:
        if type(used) == InstNode:
          visit(used)

    for node in reversed(g):
      if not found_component:
        visit(node)
        found_component = True
      elif node.level not in visited:
        # found second component
        return False
    return True

  enumerated = set()
  def enumerator():
    for g, cert in enum(num_levels-1):
      if cert not in enumerated and is_valid_graph(g):
        enumerated.add(cert)
        yield g, cert
  return enumerator(), enum_history, cert2graph

def emit_assignment(var, bitwidth, val, i, out):
  mask = 255
  num_bytes = bits2bytes(bitwidth)
  for j in range(num_bytes):
    byte = val & mask
    out.write('((uint8_t *){var})[{i} * {num_bytes} + {j}] = {byte};\n'.format(
      var=var, i=i, j=j, num_bytes=num_bytes, byte=byte
      ))
    val >>= 8

def emit_init(target, liveins, constants, test_inputs, out):
  import random
  vars = {x : z3.BitVec(x.name, x.bitwidth) for x in liveins}

  out.write('void init() {\n')
  for i in range(32):
    values = {}
    for x in liveins:
      counter_examples = test_inputs.get(x, [])
      if i < len(counter_examples):
        val = counter_examples[i]
      else:
        val = random.randint(0, (1<<x.bitwidth)-1)
      values[x] = val

    if i < len(test_inputs.get('target', [])):
      soln = test_inputs['target'][i]
    else:
      substitutions = [
          (vars[x], z3.BitVecVal(values[x], x.bitwidth))
          for x in liveins]
      z3_soln = z3.simplify(z3.substitute(target, *substitutions))
      assert z3.is_const(z3_soln)
      soln = z3_soln.as_long()

    emit_assignment('target', target.size(), soln, i, out)
    for x, val in values.items():
      emit_assignment(get_livein_buf_name(x), x.bitwidth, val, i, out)
    for c in constants:
      emit_assignment(get_const_buf_name(c), c.bitwidth, c.value, i, out)
    
  out.write('}\n')

def emit_inst_runners(sketch_nodes, out, h_out):
  emitted = set()
  for n in sketch_nodes.values():
    if n.inst_groups is None:
      continue

    for inst_group in n.inst_groups:
      num_inputs = len(inst_group.input_sizes)
      num_outputs = len(inst_group.output_sizes)
      inputs = ['x%d' % i for i in range(num_inputs)]
      outputs = ['y%d' % i for i in range(num_outputs)]

      for inst in inst_group.insts:
        if inst in emitted:
          continue
        emitted.add(inst)
        decl = 'int run_{inst}_{imm8}(int num_tests, {params})\n'.format(
          inst=inst.name, imm8=str(inst.imm8) if inst.imm8 else '0', 
          params=', '.join('char *__restrict__ '+x for x in (inputs+outputs)),
          )
        h_out.write(decl + ';\n')
        out.write(decl + '{\n')

        # tell the compiler that the arguments are aligned
        for x in (inputs + outputs):
          out.write('{x} = __builtin_assume_aligned({x}, 64);\n'.format(x=x))

        out.write('for (int i = 0; i < num_tests; i++) {')
        out.write(expr_generators[inst.name]('i', inputs, outputs, inst.imm8))
        out.write('}\n') # end for

        out.write('return 0;\n') # report we didn't encounter div-by-zero

        out.write('}\n') # end function

if __name__ == '__main__':
  import sys
  insts = []

  bw = 256

  for inst, (input_types, _) in sigs.items():
    if sigs[inst][1][0] != 256:
      continue

    #if str(bw) not in inst or 'llvm' not in inst:
    #  continue

    #if 'llvm' not in inst:
    #  continue

    #if 'Div' in inst or 'Rem' in inst:
    #  continue

    #if sigs[inst][1][0] not in (256, ):
    #  continue
    #if ((sigs[inst][1][0] not in (256,128) or ('epi64' not in inst)) and
    #    ('llvm' not in inst or '64' not in inst)):
    #  if 'broadcast' not in inst:
    #    continue

    #if 'llvm' not in inst or '64' not in inst:
    #  continue

    #if not sigs[inst][1][0] == 64:
    #  continue

    has_imm8 = any(ty.is_constant for ty in input_types)
    if not has_imm8:
      insts.append(ConcreteInst(inst, imm8=None))
    else:
      insts.append(ConcreteInst(inst, imm8=str(0)))
      continue
      for imm8 in range(256):
        insts.append(ConcreteInst(inst, imm8=str(imm8)))

  import random
  random.seed(42)
  random.shuffle(insts)
  insts = insts[:30]

  liveins = [Variable('x', bw), Variable('y', bw)]#, ('z', bw)]
  constants = [Constant(1,256), Constant(0,256)]
  x, y, z = z3.BitVecs('x y z', bw)
  target = z3.If(x >= y, x, y)

  sig2insts = classify_insts(insts)
  graphs, enum_history, cert2graph = enumerate_graphs(
      target.size(), liveins, sig2insts, 4, constants)
  from tqdm import tqdm
  unique_graphs = {}
  for g, cert in tqdm(iter(graphs), total=1e9):
    unique_graphs[cert] = g

  prune_enum_history(enum_history, cert2graph, unique_graphs)

  with open('t.c', 'w') as out:
    out.write('#include "insts.h"\n')
    emit_includes(out)
    emit_enumerator(
        target.size(),
        unique_graphs, sig2insts, liveins, constants,
        enum_history, cert2graph, out)
    emit_init(target, liveins, constants, {}, out)
    out.write('''
int main() { 
  init();
  enumerate(32);
  printf("num enumerated: %llu\\n", num_enumerated); 
}
  ''')
