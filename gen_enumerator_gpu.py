from codegen import expr_generators
from collections import namedtuple, defaultdict
import itertools
import io
from expr_sampler import sigs
import sys

import z3

def debug(*args):
  print(*args, file=sys.stderr)

constants = [0,1,2,4,8,16,32,64,128]
constants = []
constant_pool = list(zip(constants, itertools.repeat(8)))

InstGroup = namedtuple('InstGroup', ['insts', 'input_sizes', 'output_sizes'])
SketchNode = namedtuple('SketchNode', ['inst_groups', 'var', 'var_size', 'const_val'])
ConcreteInst = namedtuple('ConcreteInst', ['name', 'imm8'])
ArgConfig = namedtuple('ArgConfig', ['name', 'options'])
InstConfig = namedtuple('InstConfig', ['name', 'node_id', 'group_id', 'options', 'args'])

def create_inst_node(inst_groups):
  return SketchNode(inst_groups, None, None, None)

def create_var_node(var, size, const_val=None):
  return SketchNode(None, var, size, const_val)

def bits2bytes(bits):
  return max(bits, 8) // 8

def get_usable_inputs(input_size, sketch_graph, sketch_nodes, outputs, v):
  usable_inputs = []
  for w in sketch_graph[v]:
    ## dead node, bail!
    #if len(outputs[w]) == 0:
    #  continue
    #for i, size in enumerate(sketch_nodes[w].output_sizes):
    #  if size == input_size:
    #    # we can use the i'th output of w
    #    input_idxs.append((w, i))

    for group_id, output, size in outputs[w]:
      if size == input_size:
        # FIXME: use a namedtuple
        # also include the group id, so that we only select the output if the group is active
        usable_inputs.append((w, group_id, output, size))

  return usable_inputs

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
      size = sketch_nodes[v].var_size
      const_val = sketch_nodes[v].const_val
      assert var is not None
      outputs[v] = [(0, var, size)]
      liveins.append((var, size, const_val))
      return

    for w in sketch_graph[v]:
      visit(w)

    node_evals = []
    node_configs = []
    for group_id, inst_group in enumerate(sketch_nodes[v].inst_groups):
      with io.StringIO() as out:
        num_inputs = len(inst_group.input_sizes)
        num_outputs = len(inst_group.output_sizes)

        out.write('{\n') # new scope

        # generate code to select arguments
        arg_configs = []
        inst_group_inactive = False
        for i, x_size in enumerate(inst_group.input_sizes):
          x_size_bytes = max(x_size, 8) // 8

          usable_outputs = get_usable_inputs(x_size, sketch_graph, sketch_nodes, outputs, v)
          if len(usable_outputs) == 0:
            # cant' use this node, bail!
            outputs.setdefault(v, [])
            inst_group_inactive = True
            break

          x = 'x%d' % i
          out.write('char *%s;\n' % x)
      
          arg_config = 'arg_%d_%d_%d' % (v, group_id, i)
          out.write('switch (%s) {\n' % arg_config)
          for j, (w, group_id2, var_to_use, _) in enumerate(usable_outputs):
            var_is_livein = any(var_to_use == var for var, _ , _ in liveins) 
            if var_is_livein:
              var_to_use = '(%s+test_id * %d)' % (var_to_use, x_size_bytes)
            # bail if the group whose output we are using is not active
            guard_inactive_input = 'if (!active_{w}_{group_id}) return;'.format(w=w, group_id=group_id2)
            if sketch_nodes[w].inst_groups is None:
              # w is one of the livens so always active:
              guard_inactive_input = ''
            out.write('case {j}: {guard_inactive_input} x{i} = {var_to_use}; break;\n'.format(
              j=j, i=i, var_to_use=var_to_use, guard_inactive_input=guard_inactive_input))
          out.write('}\n') # end switch

          arg_configs.append(ArgConfig(arg_config, usable_outputs))

        # move on if we can statically show that this group never has usable values
        if inst_group_inactive:
          continue

        v_outputs = ['y_%d_%d_%d'%(v, group_id, i) for i in range(num_outputs)]
          
        num_insts = len(inst_group.insts)
        node_configs.append(
          InstConfig(name='op_%d_%d' % (v, group_id), node_id=v, group_id=group_id, args=arg_configs, options=inst_group.insts))

        # now run the instruction
        out.write('switch(op_%d_%d) {\n' % (v, group_id))
        inputs = ['x%d'%i for i in range(num_inputs)]
        for i, inst in enumerate(inst_group.insts):
          out.write('case %d: {\n' % i)
          computation = expr_generators[inst.name]('0', 
            inputs, ['&'+y for y in v_outputs], str(inst.imm8), using_gpu=True)
          out.write(computation)
          out.write('} break;\n') # end case
        out.write('}\n') # end switch

        out.write('}\n') # end scope

        node_evals.append(out.getvalue())
        outputs.setdefault(v, []).extend(zip(itertools.repeat(group_id), v_outputs, inst_group.output_sizes))

        ###### end buf scope...
      # end loop...
    inst_evaluations.append(node_evals)
    configs.append(node_configs)

  with io.StringIO() as buf:
    for v in sketch_graph:
      visit(v)

  livein_names = [x for x, _, _ in liveins]

  # allocate global buffers for livens and the target
  for bufs in outputs.values():
    for _, var, size in bufs:
      if var not in livein_names:
        continue
      bytes = max(size, 8) // 8
      out.write('char %s_host[%d];\n' % (var, bytes * max_tests))

  # also allocate buffer to store the target
  target_bytes = max(target_size, 8) // 8
  out.write('char target_host[%d];\n' % (target_bytes * max_tests))

  return outputs, inst_evaluations, liveins, configs

def emit_includes(out):
  out.write('#include <string.h>\n') # memcmp
  out.write('#include <immintrin.h>\n') # duh
  out.write('#include <stdio.h>\n') # debug
  out.write('#include <stdint.h>\n')
  out.write('#define __int64_t __int64\n')
  out.write('#define __int64 long long\n')

def emit_enumerator(outputs, liveins, target_size, sketch_nodes, inst_evaluations, configs, out):
  livein_names = [x for x, _, _ in liveins]
  params = livein_names + ['target']
  param_list = ', '.join('char *'+p for p in params)
  out.write('__global__ void run_inst(%s) {\n' % param_list)
  out.write('int idx = threadIdx.x + blockDim.x * blockIdx.x;\n')
  # FIXME : dont hard-code 32
  out.write('int config_id = idx / 32;\n')
  out.write('int test_id = idx % 32;\n')

  # allocate the buffers storing the variables/temporaries
  for bufs in outputs.values():
    for _, var, size in bufs:
      if var in livein_names:
        continue
      bytes = max(size, 8) // 8
      out.write('char __align__(64) %s[%d];\n' % (var, bytes))

  # declare the ``active'' flags
  for node_configs in configs:
    for inst_config in node_configs:
      out.write('int active_%d_%d = 0;\n' % (inst_config.node_id, inst_config.group_id))

  # reverse the top-sorted inst/configs and emit code for the last instruction group
  for node_configs, node_evals in list(zip(configs, inst_evaluations)):
    node_id = node_configs[0].node_id

    # go through each inst group
    num_configs_in_node = 1
    num_group_configs = []
    for inst_eval, inst_config in zip(node_evals, node_configs):
      num_configs = len(inst_config.options)
      for arg in inst_config.args:
          num_configs *= len(arg.options)
      num_configs_in_node += num_configs
      num_group_configs.append(num_configs)

    node_config_id = 'node_config_id_%d' % inst_config.node_id
    out.write('int {node_config_id} = config_id % {num_configs_in_node};\n'.format(
      node_config_id=node_config_id, num_configs_in_node=num_configs_in_node
      ))
    out.write('config_id /= {num_configs_in_node};\n'.format(num_configs_in_node=num_configs_in_node))

    configs_processed = 0
    for inst_eval, inst_config, num_configs in zip(node_evals, node_configs, num_group_configs):
      out.write('if ({node_config_id} >= 0 && {node_config_id} < {hi}) {{\n'.format(
        node_config_id=node_config_id, hi=num_configs))

      # activate this group
      out.write('active_%d_%d = 1;\n' % (inst_config.node_id, inst_config.group_id))

      # decode the inst opcode
      out.write('int {op} = {node_config_id} % {num_insts};\n'.format(
        op=inst_config.name, node_config_id=node_config_id, num_insts=len(inst_config.options)))
      out.write('{node_config_id} /= {num_insts};\n'.format(
        node_config_id=node_config_id, num_insts=len(inst_config.options)))

      # decode the operands
      for arg in inst_config.args:
        out.write('int {arg} = {node_config_id} % {num_args};\n'.format(
          arg=arg.name, node_config_id=node_config_id, num_args=len(arg.options)))
        out.write('{node_config_id} /= {num_args};\n'.format(
          node_config_id=node_config_id, num_args=len(arg.options)))

      # evaluate the inst once we've fixed its configs
      out.write(inst_eval)

      out.write('}\n') # close the if

      out.write('{node_config_id} -= {num_configs};\n'.format(
        node_config_id=node_config_id, num_configs=num_configs))

  out.write('}\n') # close run_inst


  # FIXME: make this real...
  out.write('int main() {\n')
  out.write('init();\n')

  # allocate device buffers for liveins and target
  for bufs in outputs.values():
    for _, var, size in bufs:
      if var not in livein_names:
        continue
      bytes = max(size, 8) // 8
      out.write('char *%s;\n' % var)
      out.write('cudaMalloc((void **)&%s, %d);\n' % (var, bytes * 32))
      out.write('cudaMemcpy({var}, {var}_host, {size}, cudaMemcpyHostToDevice);\n'.format(
        var=var, size=bytes*32 # FIXME: don't hardcode 32
        ))

  target_bytes = max(target_size, 8) // 8
  out.write('char *target;\n')
  out.write('cudaMalloc((void **)&target, %d);\n' % (target_bytes * 32))
  out.write('cudaMemcpy(target, target_host, {size}, cudaMemcpyHostToDevice);\n'.format(
    size=target_bytes * 32 # FIXME: don't hardcode 32
    ))

  out.write('}\n') # end main

# FIXME: also make it real
def emit_solution_handler(configs, out):
  out.write('void handle_solution(int num_evaluated, int _) {\n')
  out.write('printf("found a solution at iter %lu!\\n", num_evaluated);\n')
  for node_configs in configs:
    if len(node_configs) == 0:
      continue
    node_id = node_configs[0].node_id
    out.write('printf("%d = ");\n' % node_id)
    for inst_config in node_configs:

      out.write('switch (%s) {\n' % inst_config.name)
      for i, inst in enumerate(inst_config.options):
        # FIXME: this ignores imm8
        out.write('case %d: printf("%s "); break;\n' % (i, inst.name))
      out.write('}\n') # end switch

      for arg_config in inst_config.args:
          out.write('switch (%s) {\n' % arg_config.name)
          for i, arg in enumerate(arg_config.options):
            out.write('case %d: printf("%s "); break;\n' % (i, arg[2]))
          out.write('}\n') # end switch
    out.write('printf("\\n");\n')

  #out.write('exit(1);\n')

  out.write('}\n') # end func

def make_fully_connected_graph(liveins, insts, num_levels, constants=constant_pool):
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

  # create nodes for the constant pool
  for val, size in constants:
    available_bitwidths.add(size)
    node_id = node_counter
    node_counter += 1
    const_name = 'const_%d_%d' % (val, size)
    nodes[node_id] = create_var_node(const_name, size, val)

  for _ in range(num_levels):
    # nodes for current level
    inst_groups = []
    for (input_sizes, output_sizes), insts in sig2insts.items():
      usable = all(bitwidth in available_bitwidths for bitwidth in input_sizes)
      if not usable:
        continue

      # we can use this category of instructions. make a group for it
      inst_groups.append(InstGroup(insts, input_sizes, output_sizes))

      for size in output_sizes:
        available_bitwidths.add(size)

    node_id = node_counter
    # this is fully connected...
    graph[node_id] = list(nodes.keys())
    node_counter += 1
    nodes[node_id] = create_inst_node(inst_groups)

  return graph, nodes

def emit_assignment(var, bitwidth, val, i, out):
  mask = 255
  num_bytes = max(bitwidth, 8) // 8
  for j in range(num_bytes):
    byte = val & mask
    out.write('((uint8_t *){var})[{i} * {num_bytes} + {j}] = {byte};\n'.format(
      var=var, i=i, j=j, num_bytes=num_bytes, byte=byte
      ))
    val >>= 8

def emit_init(target, liveins, out):
  import random
  vars = [z3.BitVec(var, size) for var, size, _ in liveins]

  out.write('void init() {\n')
  for i in range(32):
    # generate random input
    inputs = [
      # but use the fixed input val if it's a constant
      const_val if const_val is not None else random.randint(0, (1<<size)-1)
      for _, size, const_val in liveins]

    z3_inputs = [z3.BitVecVal(val, size) for val, (_, size, _) in zip(inputs, liveins)]
    z3_soln = z3.simplify(z3.substitute(target, *zip(vars, z3_inputs)))
    assert z3.is_const(z3_soln)

    soln = z3_soln.as_long()
    emit_assignment('target_host', target.size(), soln, i, out)
    for input, (var, size, _) in zip(inputs, liveins):
      emit_assignment(var+'_host', size, input, i, out)
    
  out.write('}\n')


def p24(x, *_):
  o1 = x-1 
  o2 = o1 >> 1
  o3 = o1 | o2
  o4 = o3 >> 2
  o5 = o3 | o4
  o6 = o5 >> 4
  o7 = o5 | o6
  o8 = o7 >> 8
  o9 = o7 | o8
  o10 = o9 >> 16
  return o10 + 1 

def emit_everything(target, sketch_graph, sketch_nodes, out):
  '''
  target is an smt formula
  '''
  target_size = target.size()

  emit_includes(out)
  out.write('#include "insts.h"\n')
  insts = set()
  outputs, inst_evaluations, liveins, configs = emit_inst_evaluations(target_size, sketch_graph, sketch_nodes, out)
  emit_init(target, liveins, out)
  emit_enumerator(outputs, liveins, target_size, sketch_nodes, inst_evaluations, configs, out)

if __name__ == '__main__':
  import sys
  insts = []

  for inst, (input_types, _) in sigs.items():
    if 'llvm' not in inst:
      continue
    if sigs[inst][1][0] not in (1, 64, 32):
      continue

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

  liveins = [('x', 64), ('y', 64)]
  x, y = z3.BitVecs('x y', 64)
  target = z3.If(x >= y, x, y) * y

  g, nodes = make_fully_connected_graph(
      liveins=liveins,
      constants=[],
      insts=insts,
      num_levels=4)
  emit_everything(target, g, nodes, sys.stdout)
