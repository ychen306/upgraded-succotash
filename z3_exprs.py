import z3
from z3_utils import z3op_names, askey, get_z3_app

# z3op_names doesn't include extract, zext, or sext
ops = list(z3op_names.keys())
ops.extend([z3.Z3_OP_EXTRACT, z3.Z3_OP_ZERO_EXT,
  z3.Z3_OP_SIGN_EXT, z3.Z3_OP_UNINTERPRETED, z3.Z3_OP_BNUM])
# plus 1 for unknown
num_ops = len(ops) + 1

ops2ids = { op : i for i, op in enumerate(ops) } 

def get_op_id(op):
  id = ops2ids.get(op, None)
  if id is not None:
    return id
  return num_ops-1

max_num_params = 3

bitwidths = [1,2,4,8,16,32,64,128,256,512]
# plus one for ``others''
num_bitwidths = len(bitwidths) + 1

bws2ids = { bw : i for i, bw in enumerate(bitwidths) }
def get_bitwidth_id(bw):
  id = bws2ids.get(bw, None)
  if id is not None:
    return id
  return num_bitwidths-1

def get_canonicalized_params(expr):
  params = [get_bitwidth_id(bw) for bw in expr.params()]
  while len(params) < max_num_params-1:
    params.append(num_bitwidths-1)
  if z3.is_bv(expr):
    params.append(get_bitwidth_id(expr.size()))
  else:
    assert z3.is_bool(expr)
    params.append(get_bitwidth_id(1))
  return params

def serialize_expr(*exprs):
  '''
  <z3 expr> -> <edges>, <ops>, <params>, <node id of exprs>
  '''
  # <z3 expr> -> <id> 
  exprs2ids = {}
  edges = []

  ops = []
  # list of params
  params = [[] for _ in range(max_num_params)]

  def populate_graph(e):
    '''
    polulate subgraph reachable from `e', also return id of `e'
    '''
    k = askey(e)
    if k in exprs2ids:
      return exprs2ids[k]

    id = len(exprs2ids)
    exprs2ids[k] = id

    ops.append(get_op_id(get_z3_app(e)))
    for i, p in enumerate(get_canonicalized_params(e)):
      params[i].append(p)

    for e2 in e.children():
      id2 = populate_graph(e2)
      edge = id, id2
      edges.append(edge)
    return id

  expr_ids = [populate_graph(e) for e in exprs]
  assert len(ops) == len(params[0]) == len(exprs2ids)
  return edges, ops, params, expr_ids
