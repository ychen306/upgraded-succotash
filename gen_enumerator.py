from codegen import expr_generators
from collections import namedtuple, defaultdict
import io
from expr_sampler import sigs

SketchNode = namedtuple('SketchNode', ['insts', 'input_sizes', 'output_sizes', 'var'])
ConcreteInst = namedtuple('ConcreteInst', ['name', 'imm8'])
ArgConfig = namedtuple('ArgConfig', ['name', 'options'])
InstConfig = namedtuple('InstConfig', ['name', 'node_id', 'options', 'args'])

def create_inst_node(insts, input_sizes, output_sizes):
  return SketchNode(insts, input_sizes, output_sizes, None)

def create_var_node(var, size):
  return SketchNode(None, None, [size], var)

def bits2bytes(bits):
  return max(bits, 8) // 8

def get_usable_inputs(input_size, sketch_graph, sketch_nodes, outputs, v):
  input_idxs = []
  for w in sketch_graph[v]:
    # dead node, bail!
    if len(outputs[w]) == 0:
      continue
    for i, size in enumerate(sketch_nodes[w].output_sizes):
      if size == input_size:
        # we can use the i'th output of w
        input_idxs.append((w, i))
  return input_idxs

def emit_inst_evaluations(target_size, sketch_graph, sketch_nodes, out, max_tests=128):
  '''
  a sketch is a directed graph, where
    * each node is a set of instructions of the same signatures
    * each edges specifies nodes from which the set of instructions can use as instruction arguments

  sketch_graph : a map from nodes to dependencies
  sketch_nodes : node id -> [ <sketch node> ]
  '''
  visited = set()
  # mapping node -> variable representing the output
  outputs = {}
  inst_evaluations = []
  configs = []
  liveins = []

  def visit(v):
    if v in visited:
      return
    visited.add(v)

    is_leaf = v not in sketch_graph
    if is_leaf:
      var = sketch_nodes[v].var
      [size] = sketch_nodes[v].output_sizes
      assert var is not None
      outputs[v] = [(var, size)]
      liveins.append((var, size))
      return

    for w in sketch_graph[v]:
      visit(w)
    evaluator_name = 'inst_%d' % v
    with io.StringIO() as out:
      num_inputs = len(sketch_nodes[v].input_sizes)
      num_outputs = len(sketch_nodes[v].output_sizes)
    
      out.write('{\n') # new scope

      out.write('int div_by_zero = 0;\n')

      # generate code to select arguments
      arg_configs = []
      for i, x_size in enumerate(sketch_nodes[v].input_sizes):

        usable_outputs = get_usable_inputs(x_size, sketch_graph, sketch_nodes, outputs, v)
        if len(usable_outputs) == 0:
          # cant' use this node, bail!
          outputs[v] = []
          return

        x = 'x%i' % i
        out.write('char *%s;\n' % x)
        arg_config = 'arg_%d_%d' % (v, i)
        # FIXME: this is broken (CHANGWAN!) because *statically* we will have redundant computation here
        out.write('switch (%s) {\n' % arg_config)
        for j, (w, out_idx) in enumerate(usable_outputs):
          var_to_use, _ = outputs[w][out_idx]
          out.write('case %d: x%d = %s; break;\n' % (j, i, var_to_use))
        out.write('}\n') # end switch

        arg_configs.append(ArgConfig(arg_config, usable_outputs))

      v_outputs = ['y_%d_%d'%(v, i) for i in range(num_outputs)]
      # call the evaluator
      arg_list = ['x%d'%i for i in range(num_inputs)] + [y for y in v_outputs]
        
      num_insts = len(sketch_nodes[v].insts)
      configs.append(InstConfig(name='op_%d'%v, node_id=v, args=arg_configs, options=sketch_nodes[v].insts))

      # now run the instruction
      out.write('switch(op_%d) {\n' % v)
      inputs = ['x%d'%i for i in range(num_inputs)]
      for i, inst in enumerate(sketch_nodes[v].insts):
        out.write('case %d: {\n' % i)
        out.write('div_by_zero = run_{inst}_{imm8}(num_tests, {args});\n'.format(
          inst=inst.name, args=', '.join(inputs + v_outputs), imm8=str(inst.imm8) if inst.imm8 else '0'))
        out.write('} break;\n') # end case
      out.write('}\n') # end switch

      # skip this instruction if it divs by zero
      out.write('if (div_by_zero) continue;\n')

      out.write('}\n') # end scope

      inst_evaluations.append(out.getvalue())

    outputs[v] = list(zip(v_outputs, sketch_nodes[v].output_sizes))

  with io.StringIO() as buf:
    for v in sketch_graph:
      visit(v)

  # allocate the buffers storing the variables/temporaries
  for bufs in outputs.values():
    for var, size in bufs:
      bytes = max(size, 8) // 8
      out.write('static char %s[%d];\n' % (var, bytes * max_tests))
  # also allocate the variable storing the targets
  target_bytes = max(target_size, 8) // 8
  out.write('static char target[%d];\n' % (target_bytes * max_tests))

  # declare the configs as global variable
  for inst_config in configs:
    out.write('static int %s = -1;\n' % inst_config.name)
    for arg in inst_config.args:
      out.write('static int %s = -1;\n' % arg.name)

  return inst_evaluations, liveins, configs

def emit_includes(out):
  out.write('#include <string.h>\n') # memcmp
  out.write('#include <immintrin.h>\n') # duh
  out.write('#include <stdio.h>\n') # debug
  out.write('#include <stdint.h>\n')
  out.write('#define __int64_t __int64\n')
  out.write('#define __int64 long long\n')

# FIXME: this is broken (asymptotically)!!!! We shouldn't nest two `parallel' nodes together
def emit_enumerator(target_size, sketch_nodes, inst_evaluations, configs, out):
  out.write('void enumerate(int num_tests) {\n')
  out.write('unsigned long long num_evaluated = 0;\n')

  num_right_braces = 0
  # do a DFS over the configs
  for inst_config, inst_eval in zip(configs, inst_evaluations):
    out.write('for ({op} = 0; {op} < {options}; {op}++) {{\n'.format(
      op=inst_config.name, options=len(inst_config.options)))
    # remember to close the brace for this loop
    num_right_braces += 1

    for arg in inst_config.args:
      out.write('for ({arg} = 0; {arg} < {options}; {arg}++) {{\n'.format(
        arg=arg.name, options=len(arg.options)))
      num_right_braces += 1

    # evaluate the inst once we've fixed its configs
    out.write(inst_eval)

    # now check the result
    for i, y_size in enumerate(sketch_nodes[inst_config.node_id].output_sizes):
      if y_size == target_size:
        output_name = 'y_%d_%d' % (inst_config.node_id, i)
        out.write('if (memcmp(target, {y}, {y_size}*num_tests) == 0) handle_solution(num_evaluated, {v});\n'.format(
          y=output_name, y_size=bits2bytes(y_size), v=inst_config.node_id
          ))

  out.write('num_evaluated += 1;\n')

  # close the braces
  out.write('}\n' * num_right_braces)

  out.write('}\n') # end func

  # FIXME: make this real...
  out.write('int main() { init(); enumerate(32); } \n')

# FIXME: also make it real
def emit_solution_handler(configs, out):
  out.write('void handle_solution(int num_evaluated, int _) {\n')
  out.write('printf("found a solution at iter %lu!\\n", num_evaluated);\n')
  for inst_config in configs:

    out.write('printf("%d = ");\n' % inst_config.node_id)
    out.write('switch (%s) {\n' % inst_config.name)
    for i, inst in enumerate(inst_config.options):
      # FIXME: this ignores imm8
      out.write('case %d: printf("%s "); break;\n' % (i, inst.name))
    out.write('}\n') # end switch

    for arg_config in inst_config.args:
        # FIXME: this ignores the case when an instruction can output multiple values
        out.write('switch (%s) {\n' % arg_config.name)
        for i, arg in enumerate(arg_config.options):
          out.write('case %d: printf("%s "); break;\n' % (i, arg[0]))
        out.write('}\n') # end switch
    out.write('printf("\\n");\n')

  #out.write('exit(1);\n')

  out.write('}') # end func

def make_fully_connected_graph(liveins, insts, num_levels):
  # categorize the instructions by their signature first
  sig2insts = defaultdict(list)

  # FIXME: we need to instantiate an instruction for every possible imm8 if it uses imm8
  for inst in insts:
    input_types, out_sigs = sigs[inst.name]
    sig_without_imm8 = (tuple(ty.bitwidth for ty in input_types if not ty.is_constant), out_sigs)
    sig2insts[sig_without_imm8].append(inst)

  graph = {}
  nodes = {}

  node_counter = 0

  available_bitwidths = set()
  # create nodes for the liveins
  for x, size in liveins:
    available_bitwidths.add(size)
    node_id = node_counter
    node_counter += 1
    nodes[node_id] = create_var_node(x, size)

  for _ in range(num_levels):
    # nodes for current level
    cur_nodes = []
    for (input_sizes, output_sizes), insts in sig2insts.items():
      usable = all(bitwidth in available_bitwidths for bitwidth in input_sizes)
      if not usable:
        continue

      # we can use this category of instructions. make a node for it
      node = create_inst_node(insts, input_sizes, output_sizes)
      node_id = node_counter
      cur_nodes.append((node_id, node))
      node_counter += 1

      # this is fully connected...
      graph[node_id] = list(nodes.keys())

      for size in output_sizes:
        available_bitwidths.add(size)

    # now add the nodes of this level
    for node_id, node in cur_nodes:
      nodes[node_id] = node

  return graph, nodes

# FIXME: don't hardcode this
# FIXME: THIS IS BROKEN---we should have found a solution!
def emit_init(liveins, out):
  import random
  (x, _), (y, _) = liveins
  out.write('void init() {\n')
  for i in range(32):
    a = random.randint(-(1<<16), 1<<16)
    b = random.randint(-(1<<16), 1<<16)
    soln = max(a,b)
    out.write('((int64_t *)x)[%d] = %d;\n' % (i, a))
    out.write('((int64_t *)y)[%d] = %d;\n' % (i, b))
    out.write('((int64_t *)target)[%d] = %d;\n' % (i, soln))
  out.write('}\n')


def emit_inst_runners(sketch_nodes, out, h_out):
  emitted = set()
  for n in sketch_nodes.values():
    if n.insts is None:
      continue

    num_inputs = len(n.input_sizes)
    num_outputs = len(n.output_sizes)
    inputs = ['x%d' % i for i in range(num_inputs)]
    outputs = ['y%d' % i for i in range(num_outputs)]

    for inst in n.insts:
      if inst in emitted:
        continue
      emitted.add(inst)
      decl = 'int run_{inst}_{imm8}(int num_tests, {params})\n'.format(
        inst=inst.name, imm8=str(inst.imm8) if inst.imm8 else '0', 
        params=', '.join('char *__restrict__ '+x for x in (inputs+outputs)),
        )
      h_out.write(decl + ';\n')
      out.write(decl + '{\n')
      out.write('for (int i = 0; i < num_tests; i++) {')
      out.write(expr_generators[inst.name]('i', inputs, outputs, inst.imm8))
      out.write('}\n') # end for

      out.write('return 0;\n') # report we didn't encounter div-by-zero

      out.write('}\n') # end function

def emit_everything(target_size, sketch_graph, sketch_nodes, out):
  emit_includes(out)
  out.write('#include "insts.h"\n')
  insts = set()
  inst_evaluations, liveins, configs = emit_inst_evaluations(target_size, g, nodes, out)
  emit_init(liveins, out)
  emit_solution_handler(configs, out)
  emit_enumerator(target_size, nodes, inst_evaluations, configs, out)

def emit_insts_lib(out, h_out):
  for inst, (input_types, _) in sigs.items():
    has_imm8 = any(ty.is_constant for ty in input_types)
    if not has_imm8:
      insts.append(ConcreteInst(inst, imm8=None))
    else:
      insts.append(ConcreteInst(inst, imm8='12'))
  _, nodes = make_fully_connected_graph(
      liveins=[('x', 64), ('y', 64)],
      insts=insts,
      num_levels=4)

  emit_includes(out)
  emit_inst_runners(nodes, out, h_out)

if __name__ == '__main__':
  import sys
  insts = []

  for inst, (input_types, _) in sigs.items():
    if 'llvm' not in inst or '64' not in inst:
      continue
    if not sigs[inst][1][0] == 64:
      continue
    has_imm8 = any(ty.is_constant for ty in input_types)
    if not has_imm8:
      insts.append(ConcreteInst(inst, imm8=None))
    else:
      insts.append(ConcreteInst(inst, imm8='12'))
  g, nodes = make_fully_connected_graph(
      liveins=[('x', 64), ('y', 64)],
      insts=insts,
      num_levels=4)
  emit_everything(64, g, nodes, sys.stdout)
  exit(1)

  with open('insts.c', 'w') as out, open('insts.h', 'w') as h_out:
    emit_insts_lib(out, h_out)
