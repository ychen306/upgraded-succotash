from codegen import expr_generators
from collections import namedtuple, defaultdict
import itertools
import io
from expr_sampler import sigs
import sys

import z3
import json

def debug(*args):
  print(*args, file=sys.stderr)

constants = [0,1,2,4,8,16,32,64,128]
constants = []
constant_pool = list(zip(constants, itertools.repeat(8)))

InstGroup = namedtuple('InstGroup', ['insts', 'input_sizes', 'output_sizes', 'commutative_pair'])
SketchNode = namedtuple('SketchNode', ['inst_groups', 'var', 'var_size', 'const_val'])
ConcreteInst = namedtuple('ConcreteInst', ['name', 'imm8'])
ArgConfig = namedtuple('ArgConfig', ['name', 'options', 'switch'])
InstConfig = namedtuple('InstConfig', ['name', 'node_id', 'group_id', 'options', 'args'])

with open('commutative-params.json') as f:
  commutative_params = json.load(f)

def get_selected_arg(arg):
  return 'selected_%s' % arg

def create_inst_node(inst_groups):
  return SketchNode(inst_groups, None, None, None)

def create_var_node(var, size, const_val=None):
  return SketchNode(None, var, size, const_val)

def bits2bytes(bits):
  if bits < 8:
    assert bits == 1
    return 4
  return bits // 8

def get_arg_counter(arg_name):
  return 'counter_' + arg_name

def get_usable_inputs(input_size, sketch_graph, sketch_nodes, outputs, v):
  usable_inputs = []
  for w in sketch_graph[v]:
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
        inputs = []
        inst_group_inactive = False
        for i, x_size in enumerate(inst_group.input_sizes):
          usable_outputs = get_usable_inputs(
            x_size, sketch_graph, sketch_nodes, outputs, v)
          usable_outputs.reverse()
          if len(usable_outputs) == 0:
            # cant' use this node, bail!
            outputs.setdefault(v, [])
            inst_group_inactive = True
            break

          usable_nodes = set(w for w, _, _, _ in usable_outputs)

          arg_config = 'arg_%d_%d_%d' % (v, group_id, i)
          selected = get_selected_arg(arg_config)
          inputs.append(selected)

          with io.StringIO() as switch_buf:
            ref_counter = get_arg_counter(arg_config)
            switch_buf.write('switch (%s) {\n' % arg_config)
            for j, (w, group_id2, var_to_use, _) in enumerate(usable_outputs):
              if sketch_nodes[w].inst_groups is None:
                # live-ins is always alive
                liveness_tracking = '%s = NULL;' % ref_counter
              else:
                liveness_tracking = '''
                alive_{w}_{group_id2} ++;
                {ref_counter} = &alive_{w}_{group_id2};
                '''.format(w=w, group_id2=group_id2, ref_counter=ref_counter)

              switch_buf.write(
                '''case {j}:
                  {selected} = {var_to_use};
                  {liveness_tracking}
                  break;\n'''
                .format(
                  j=j, i=i,
                  w=w,
                  var_to_use=var_to_use,
                  liveness_tracking=liveness_tracking,
                  selected=selected,
                  group_id=group_id2
                  ))
            switch_buf.write('}\n') # end switch
            switch = switch_buf.getvalue()

          arg_configs.append(ArgConfig(arg_config, usable_outputs, switch))

        # move on if we can statically show that this group never has usable values
        if inst_group_inactive:
          continue

        v_outputs = ['y_%d_%d_%d'%(v, group_id, i) for i in range(num_outputs)]

        num_insts = len(inst_group.insts)
        node_configs.append(
          InstConfig(name='op_%d_%d' % (v, group_id), 
            node_id=v,
            group_id=group_id,
            args=arg_configs,
            options=inst_group.insts))

        # now run the instruction
        # first put all the functions that we want to call into an indirect call table
        params = ['int'] # num tests
        for _ in range(len(inputs) + len(v_outputs)):
          params.append('char *__restrict__')
        funcs = ['run_{inst}_{imm8}'.format(
          inst=inst.name,
          imm8=str(inst.imm8) if inst.imm8 else '0')
          for inst in inst_group.insts]
        out.write('static int (*funcs[{num_funcs}]) ({param_list}) = {{ {funcs} }};\n'.format(
          num_funcs = str(len(inst_group.insts)),
          param_list = ', '.join(params),
          funcs = ', '.join(funcs)
          ))
        out.write('div_by_zero = funcs[op_{node_id}_{group_id}](num_tests, {args});\n'.format(
            args=', '.join(inputs + v_outputs),
            node_id=v,
            group_id=group_id,
            ))

        # skip this instruction if it divs by zero
        out.write('if (div_by_zero) continue;\n')

        out.write('}\n') # end scope

        node_evals.append(out.getvalue())
        outputs.setdefault(v, []).extend(
            zip(itertools.repeat(group_id), v_outputs, inst_group.output_sizes))

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
      bytes = bits2bytes(size)
      out.write('static char %s[%d] __attribute__ ((aligned (64)));\n' % (var, bytes * max_tests))
  # also allocate the variable storing the targets
  target_bytes = bits2bytes(target_size)
  out.write('static char target[%d];\n' % (target_bytes * max_tests))

  # declare the configs as global variable
  for i, node_configs in enumerate(configs):
    at_last_node = i == len(configs)-1
    alive = 1 if at_last_node else 0
    for inst_config in node_configs:
      out.write('static int alive_%s_%s = %d;\n' % (
        inst_config.node_id, 
        inst_config.group_id,
        alive
        ))
      out.write('static int active_%s_%s = 0;\n' % (
        inst_config.node_id, inst_config.group_id))
      out.write('static int %s = -1;\n' % inst_config.name)
      for arg in inst_config.args:
        out.write('static int %s = -1;\n' % arg.name)
        out.write('static char *%s;\n' % get_selected_arg(arg.name))
  out.write('static unsigned long long num_evaluated = 0;\n')

  return inst_evaluations, liveins, configs

def emit_includes(out):
  out.write('#include <string.h>\n') # memcmp
  out.write('#include <immintrin.h>\n') # duh
  out.write('#include <stdio.h>\n') # debug
  out.write('#include <stdint.h>\n')
  out.write('#define __int64_t __int64\n')
  out.write('#define __int64 long long\n')

def emit_enumerator(target_size, sketch_nodes, inst_evaluations, configs, out):
  node_ids = []
  # forward declare the funcs
  for node_configs in configs:
    node_id = node_configs[0].node_id
    out.write('static void run_node_%d(int);\n' % node_id)
    out.write('static void run_node_%d_aux(int);\n' % node_id)
    node_ids.append(node_id)

  # go through the levels bottom-up
  for level_id, (node_configs, node_evals) in enumerate(
      reversed(list(zip(configs, inst_evaluations)))):
    node_id = node_configs[0].node_id
    out.write('static void run_node_%d(int num_tests) {\n' % node_id)

    liveness_flags = []
    for inst_eval, inst_config in zip(node_evals, node_configs):
      comm_pair = sketch_nodes[inst_config.node_id].inst_groups[inst_config.group_id].commutative_pair
      p1, p2 = None, None
      if comm_pair is not None:
        p1, p2 = comm_pair

      # the liveness flag
      lf = 'alive_%d_%d' % (
          inst_config.node_id, inst_config.group_id)
      # indicate that we are enumerating this group
      out.write('active_%d_%d = 1;\n' % (
        inst_config.node_id, inst_config.group_id))
      # don't bother enumerate this config if it doesn't have any user
      out.write('if (%s)\n' % lf)
      liveness_flags.append(lf)

      if level_id < len(configs)-1:
        next_configs = configs[len(configs)-(level_id+1)-1]
        next_node_id = next_configs[0].node_id
        iter_next_level = 'run_node_%d(num_tests);\n' % next_node_id
      else:
        iter_next_level = 'run_node_%d_aux(num_tests);\n' % node_id

      # remember the number of right braces we need to close
      scope_stack = []

      for arg_id, arg in enumerate(inst_config.args):
        out.write('for ({arg} = {start}; {arg} < {options}; {arg}++) {{\n'.format(
          arg=arg.name,
          options=len(arg.options),
          start=('0' if p2 != arg_id else inst_config.args[p1].name)
          ))

        ref_counter = get_arg_counter(arg.name)
        # declare it
        out.write('int *%s;\n' % ref_counter)
        out.write(arg.switch)
        # decrement the counter once we are done using a value
        scope_stack.append(
            "if ({counter}) (*{counter})--;\n}}\n"
            .format(counter=ref_counter))

      # after we've selected the configs for this node run the preceding nodes
      out.write(iter_next_level)

      # decrement the reference count and close the braces
      while len(scope_stack) > 0:
        out.write(scope_stack.pop())

      # indicate that we are not enumerating this group anymore
      out.write('active_%d_%d = 0;\n' % (
        inst_config.node_id, inst_config.group_id))

    # still enumerate the next level even if the whole level is dead
    out.write('if (!(%s)) { %s }\n' % (
      '||'.join(liveness_flags), iter_next_level
      ))

    out.write('}\n') # close the function for this node

  next_node_id = None
  first_node_id = None
  for node_configs, node_evals in reversed(list(zip(configs, inst_evaluations))):
    node_id = node_configs[0].node_id
    if first_node_id is None:
      first_node_id = node_id
    out.write('static void run_node_%d_aux(int num_tests) {\n' % node_id)

    active_flags = []

    for inst_eval, inst_config in zip(reversed(node_evals), reversed(node_configs)):
      # flag indicating whether a group is active
      af = 'active_%d_%d' % (inst_config.node_id, inst_config.group_id)
      out.write('if (%s)\n' % af)
      active_flags.append(af)

      out.write('for ({op} = 0; {op} < {options}; {op}++) {{\n'.format(
        op=inst_config.name, options=len(inst_config.options)))

      # evaluate the inst
      out.write(inst_eval)

      # now check the result
      out_sizes = sketch_nodes[inst_config.node_id].inst_groups[inst_config.group_id].output_sizes
      for i, y_size in enumerate(out_sizes):
        if y_size == target_size:
          output_name = 'y_%d_%d_%d' % (inst_config.node_id, inst_config.group_id, i)
          out.write('if (memcmp(target, {y}, {y_size}*num_tests) == 0) handle_solution(num_evaluated, {v});\n'.format(
            y=output_name, y_size=bits2bytes(y_size), v=inst_config.node_id
            ))

      if next_node_id is None:
        # at the leaf node 
        out.write('num_evaluated += 1;\n')
      else:
        out.write('run_node_%d_aux(num_tests);\n' % next_node_id)

      out.write('}\n') # close the loop

    # in the case that none of the groups are active,
    # we still want to enumerate the next level
    if next_node_id is not None:
      out.write('if (!({flags})) run_node_{next_node_id}_aux(num_tests);\n'
          .format(
            flags='||'.join(active_flags),
            next_node_id = next_node_id
            ))

    out.write('}\n') # close the function for this node
    next_node_id = node_id


  out.write('void enumerate(int num_tests) { run_node_%d(num_tests); }\n' % first_node_id)
  # FIXME: make this real...
  out.write('int main() { init(); enumerate(32); } \n')

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

def prune_graph(target_size, sketch_graph, sketch_nodes):
  # do top sort
  sorted_nodes = []
  visited = set()
  def topsort(v):
    if v in visited:
      return
    visited.add(v)

    is_leaf = v not in sketch_graph
    if is_leaf:
      sorted_nodes.append(v)
      return

    for w in sketch_graph[v]:
      topsort(w)

    sorted_nodes.append(v)

  for v in sketch_graph:
    topsort(v)

  # after this the src nodes of the dep graph will show up first
  sorted_nodes.reverse()

  # mapping nodes -> users
  users = defaultdict(list)
  for v in sorted_nodes:
    is_leaf = v not in sketch_graph
    # nothing to prune
    if is_leaf:
      continue

    # the target output is always useful
    useful_bitwidths = {target_size}
    # figure out outputs that are potentially useful
    for u in users[v]:
      for inst_group in sketch_nodes[u].inst_groups:
        for bitwidth in inst_group.input_sizes:
          useful_bitwidths.add(bitwidth)
    # drop an inst group if none of its output produce useful bitwidths
    filtered_groups = [inst_group
        for inst_group in sketch_nodes[v].inst_groups
        if any(bw in useful_bitwidths for bw in inst_group.output_sizes)]
    debug('DROPPED', len(sketch_nodes[v].inst_groups) - len(filtered_groups), 'NODES')
    sketch_nodes[v].inst_groups[:] = filtered_groups
    for w in sketch_graph[v]:
      users[w].append(v)

def make_fully_connected_graph(liveins, insts, num_levels, constants=constant_pool):
  # categorize the instructions by their signature first
  sig2insts = defaultdict(list)

  # FIXME: we need to instantiate an instruction for every possible imm8 if it uses imm8
  for inst in insts:
    input_types, out_sigs = sigs[inst.name]
    comm_pairs = commutative_params.get(inst.name, [])
    comm_pair = (None,)
    if len(comm_pairs) >= 1:
      comm_pair = (tuple(comm_pairs[0]),)
    sig_without_imm8 = (tuple(ty.bitwidth for ty in input_types if not ty.is_constant), out_sigs)
    sig2insts[sig_without_imm8 + comm_pair].append(inst)

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
    for (input_sizes, output_sizes, comm_pair), insts in sig2insts.items():
      usable = all(bitwidth in available_bitwidths for bitwidth in input_sizes)
      if not usable:
        continue

      # we can use this category of instructions. make a group for it
      inst_groups.append(InstGroup(insts, input_sizes, output_sizes, comm_pair))

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
  num_bytes = bits2bytes(bitwidth)
  for j in range(num_bytes):
    byte = val & mask
    out.write('((uint8_t *){var})[{i} * {num_bytes} + {j}] = {byte};\n'.format(
      var=var, i=i, j=j, num_bytes=num_bytes, byte=byte
      ))
    val >>= 8

def emit_init(target, liveins, out, test_inputs={}):
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
    #inputs = [
    #  # but use the fixed input val if it's a constant
    #  const_val if const_val is not None else random.randint(0, (1<<size)-1)
    #  for _, size, const_val in liveins]
    inputs = []
    for var, size, const_val in liveins:
      if const_val is not None:
        inputs.append(const_val)
      else:
        counter_examples = test_inputs.get(var, [])
        if i < len(counter_examples):
          inputs.append(counter_examples[i])
        else:
          inputs.append(random.randint(0, (1<<size)-1))

    if i < len(test_inputs.get('target', [])):
      soln = test_inputs['target'][i]
    else:
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

def emit_everything(target, sketch_graph, sketch_nodes, out, test_inputs={}):
  '''
  target is an smt formula
  '''
  target_size = target.size()

  prune_graph(target_size, sketch_graph, sketch_nodes)

  emit_includes(out)
  out.write('#include "insts.h"\n')
  insts = set()
  inst_evaluations, liveins, configs = emit_inst_evaluations(
      target_size, sketch_graph, sketch_nodes, out)
  emit_init(target, liveins, out, test_inputs=test_inputs)
  emit_solution_handler(configs, out)
  emit_enumerator(target_size, sketch_nodes, inst_evaluations, configs, out)

if __name__ == '__main__':
  import sys
  insts = []

  bw = 32

  for inst, (input_types, _) in sigs.items():
    #if sigs[inst][1][0] != 256:
    #  continue

    #if str(bw) not in inst or 'llvm' not in inst:
    #  continue

    if 'llvm' not in inst:
      continue

    if 'Div' in inst or 'Rem' in inst:
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

  import random
  random.seed(42)
  random.shuffle(insts)

  liveins = [('x', bw), ('y', bw)]#, ('z', bw)]
  x, y, z = z3.BitVecs('x y z', bw)
  target = z3.If(x >= y, x , y)

  g, nodes = make_fully_connected_graph(
      liveins=liveins,
      constants=[],
      insts=insts,
      num_levels=4)
  emit_everything(target, g, nodes, sys.stdout)
