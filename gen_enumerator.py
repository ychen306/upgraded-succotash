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

        out.write('int div_by_zero = 0;\n')

        # generate code to select arguments
        arg_configs = []
        inst_group_inactive = False
        for i, x_size in enumerate(inst_group.input_sizes):

          usable_outputs = get_usable_inputs(x_size, sketch_graph, sketch_nodes, outputs, v)
          if len(usable_outputs) == 0:
            # cant' use this node, bail!
            outputs.setdefault(v, [])
            inst_group_inactive = True
            break

          x = 'x%i' % i
          out.write('char *%s;\n' % x)
          arg_config = 'arg_%d_%d_%d' % (v, group_id, i)
          out.write('switch (%s) {\n' % arg_config)
          for j, (w, group_id2, var_to_use, _) in enumerate(usable_outputs):
            # bail if the group whose output we are using is not active
            guard_inactive_input = 'if (!active_{w}_{group_id}) continue;'.format(w=w, group_id=group_id2)
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
        arg_list = ['x%d'%i for i in range(num_inputs)] + [y for y in v_outputs]
          
        num_insts = len(inst_group.insts)
        node_configs.append(
          InstConfig(name='op_%d_%d' % (v, group_id), node_id=v, group_id=group_id, args=arg_configs, options=inst_group.insts))

        # now run the instruction
        out.write('switch(op_%d_%d) {\n' % (v, group_id))
        inputs = ['x%d'%i for i in range(num_inputs)]
        for i, inst in enumerate(inst_group.insts):
          out.write('case %d: {\n' % i)
          out.write('div_by_zero = run_{inst}_{imm8}(num_tests, {args});\n'.format(
            inst=inst.name, args=', '.join(inputs + v_outputs), imm8=str(inst.imm8) if inst.imm8 else '0'))
          out.write('} break;\n') # end case
        out.write('}\n') # end switch

        # skip this instruction if it divs by zero
        out.write('if (div_by_zero) continue;\n')

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

  # allocate the buffers storing the variables/temporaries
  for bufs in outputs.values():
    for _, var, size in bufs:
      bytes = max(size, 8) // 8
      out.write('static char %s[%d] __attribute__ ((aligned (64)));\n' % (var, bytes * max_tests))
  # also allocate the variable storing the targets
  target_bytes = max(target_size, 8) // 8
  out.write('static char target[%d];\n' % (target_bytes * max_tests))

  # declare the configs as global variable
  for node_configs in configs:
    for inst_config in node_configs:
      out.write('static int active_%d_%d = 0;\n' % (inst_config.node_id, inst_config.group_id))
      out.write('static int %s = -1;\n' % inst_config.name)
      for arg in inst_config.args:
        out.write('static int %s = -1;\n' % arg.name)
  out.write('static unsigned long long num_evaluated = 0;\n')

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
  next_node_id = None

  # reverse the top-sorted inst/configs and emit code for the last instruction group
  for node_configs, node_evals in reversed(list(zip(configs, inst_evaluations))):
    node_id = node_configs[0].node_id
    out.write('static void run_node_%d(int num_tests) {\n' % node_id)

    # go through each inst group
    for inst_eval, inst_config in zip(node_evals, node_configs):

      # activate this group
      out.write('active_%d_%d = 1;\n' % (inst_config.node_id, inst_config.group_id))

      # remember the number of right braces we need to close
      num_right_braces = 1
      out.write('for ({op} = 0; {op} < {options}; {op}++) {{\n'.format(
        op=inst_config.name, options=len(inst_config.options)))


      for arg in inst_config.args:
        out.write('for ({arg} = 0; {arg} < {options}; {arg}++) {{\n'.format(
          arg=arg.name, options=len(arg.options)))
        num_right_braces += 1

      # evaluate the inst once we've fixed its configs
      out.write(inst_eval)

      # now check the result
      out_sizes = sketch_nodes[inst_config.node_id].inst_groups[inst_config.group_id].output_sizes
      for i, y_size in enumerate(out_sizes):
        if y_size == target_size:
          output_name = 'y_%d_%d_%d' % (inst_config.node_id, inst_config.group_id, i)
          out.write('if (memcmp(target, {y}, {y_size}*num_tests) == 0) handle_solution(num_evaluated, {v});\n'.format(
            y=output_name, y_size=bits2bytes(y_size), v=inst_config.node_id
            ))

      # after we've selected the configs for this node, run the next node
      if next_node_id is not None:
        out.write('run_node_%d(num_tests);\n' % next_node_id)
      else:
        out.write('num_evaluated += 1;\n')

      out.write('}\n' * num_right_braces)

      # deactivate this group
      out.write('active_%d_%d = 0;\n' % (inst_config.node_id, inst_config.group_id))

    next_node_id = inst_config.node_id

    out.write('}\n') # close the function for this node


  out.write('void enumerate(int num_tests) { run_node_%d(num_tests); }\n' % next_node_id)
  # FIXME: make this real...
  out.write('int main() { init(); enumerate(32); } \n')

# FIXME: also make it real
def emit_solution_handler(configs, out):
  out.write('void handle_solution(int num_evaluated, int _) {\n')
  out.write('printf("found a solution at iter %lu!\\n", num_evaluated);\n')
  for node_configs in configs:
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
    #a = random.randint(-(1<<16), 1<<16)
    #b = random.randint(-(1<<16), 1<<16)
    #soln = max(a,b)
    #out.write('((int64_t *)x)[%d] = %d;\n' % (i, a))
    #out.write('((int64_t *)y)[%d] = %d;\n' % (i, b))
    #out.write('((int64_t *)target)[%d] = %d;\n' % (i, soln))

    # generate random input
    inputs = [
      # but use the fixed input val if it's a constant
      const_val if const_val is not None else random.randint(0, (1<<size)-1)
      for _, size, const_val in liveins]

    z3_inputs = [z3.BitVecVal(val, size) for val, (_, size, _) in zip(inputs, liveins)]
    z3_soln = z3.simplify(z3.substitute(target, *zip(vars, z3_inputs)))
    assert z3.is_const(z3_soln)

    soln = z3_soln.as_long()
    emit_assignment('target', target.size(), soln, i, out)
    for input, (var, size, _) in zip(inputs, liveins):
      emit_assignment(var, size, input, i, out)
    
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

def emit_everything(target, sketch_graph, sketch_nodes, out):
  '''
  target is an smt formula
  '''
  target_size = target.size()

  emit_includes(out)
  out.write('#include "insts.h"\n')
  insts = set()
  inst_evaluations, liveins, configs = emit_inst_evaluations(target_size, sketch_graph, sketch_nodes, out)
  emit_init(target, liveins, out)
  emit_solution_handler(configs, out)
  emit_enumerator(target_size, sketch_nodes, inst_evaluations, configs, out)

def emit_insts_lib(out, h_out):
  for inst, (input_types, _) in sigs.items():
    has_imm8 = any(ty.is_constant for ty in input_types)
    if not has_imm8:
      insts.append(ConcreteInst(inst, imm8=None))
    else:
      for imm8 in range(255):
        insts.append(ConcreteInst(inst, imm8=str(imm8)))
  _, nodes = make_fully_connected_graph(
      liveins=[('x', 64), ('y', 64)],
      constants=[],
      insts=insts,
      num_levels=4)

  emit_includes(out)
  emit_inst_runners(nodes, out, h_out)

if __name__ == '__main__':
  import sys
  insts = []

  #with open('insts.c', 'w') as out, open('insts.h', 'w') as h_out:
  #  emit_insts_lib(out, h_out)

  for inst, (input_types, _) in sigs.items():
    #if ((sigs[inst][1][0] not in (256,128) or ('epi64' not in inst)) and
    #    ('llvm' not in inst or '64' not in inst)):
    #  if 'broadcast' not in inst:
    #    continue

    if 'llvm' not in inst or '64' not in inst:
      continue

    #if not sigs[inst][1][0] == 64:
    #  continue

    has_imm8 = any(ty.is_constant for ty in input_types)
    if not has_imm8:
      insts.append(ConcreteInst(inst, imm8=None))
    else:
      insts.append(ConcreteInst(inst, imm8=str(0)))
      continue
      for imm8 in range(255):
        insts.append(ConcreteInst(inst, imm8=str(imm8)))

  liveins = [('x', 64), ('y', 64)]
  x, y = z3.BitVecs('x y', 64)
  target = x * 8 

  g, nodes = make_fully_connected_graph(
      liveins=liveins,
      constants=[(8,8), (12, 8)],
      insts=insts,
      num_levels=4)
  emit_everything(target, g, nodes, sys.stdout)