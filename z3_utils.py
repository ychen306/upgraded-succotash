import z3
import operator
import functools

z3op_names = {
    z3.Z3_OP_AND: 'and',
    z3.Z3_OP_OR: 'or',
    z3.Z3_OP_XOR: 'xor',
    z3.Z3_OP_FALSE: 'false',
    z3.Z3_OP_TRUE: 'true',
    z3.Z3_OP_NOT: 'not',
    z3.Z3_OP_ITE: 'ite',
    z3.Z3_OP_BAND : 'bvand',
    z3.Z3_OP_BOR : 'bvor',
    z3.Z3_OP_BXOR : 'bvxor',
    z3.Z3_OP_BNOT : 'bvnot',
    z3.Z3_OP_BNEG : 'bvneg',
    z3.Z3_OP_CONCAT : 'concat',
    z3.Z3_OP_ULT : 'bvult',
    z3.Z3_OP_ULEQ : 'bvule',
    z3.Z3_OP_SLT : 'bvslt',
    z3.Z3_OP_SLEQ : 'bvsle',
    z3.Z3_OP_UGT : 'bvugt',
    z3.Z3_OP_UGEQ : 'bvuge',
    z3.Z3_OP_SGT : 'bvsgt',
    z3.Z3_OP_SGEQ : 'bvsge',
    z3.Z3_OP_BADD : 'bvadd',
    z3.Z3_OP_BMUL : 'bvmul',
    z3.Z3_OP_BUDIV : 'bvudiv',
    z3.Z3_OP_BSDIV : 'bvsdiv',
    z3.Z3_OP_BUREM : 'bvurem', 
    z3.Z3_OP_BSREM : 'bvsrem',
    z3.Z3_OP_BSMOD : 'bvsmod', 
    z3.Z3_OP_BSHL : 'bvshl',
    z3.Z3_OP_BLSHR : 'bvlshr',
    z3.Z3_OP_BASHR : 'bvashr',
    z3.Z3_OP_BSUB : 'bvsub',
    z3.Z3_OP_EQ : '=',
    z3.Z3_OP_DISTINCT : 'distinct',

    z3.Z3_OP_BSDIV_I:  'bvsdiv',
    z3.Z3_OP_BUDIV_I:  'bvudiv',
    z3.Z3_OP_BSREM_I:  'bvsrem',
    z3.Z3_OP_BUREM_I:  'bvurem',
    z3.Z3_OP_BSMOD_I:  'bvsmod',

    ## z3.Z3_OP_SIGN_EXT: lambda args, expr: self.mgr.BVSExt(args[0], z3.get_payload(expr, 0)),
    ## z3.Z3_OP_ZERO_EXT: lambda args, expr: self.mgr.BVZExt(args[0], z3.get_payload(expr, 0)),
    ## z3.Z3_OP_EXTRACT: lambda args, expr: self.mgr.BVExtract(args[0],
    }

def assoc_op(op):
  return lambda *xs: functools.reduce(op, xs)

class AstRefKey:
    def __init__(self, n):
        self.n = n
    def __hash__(self):
        return self.n.hash()
    def __eq__(self, other):
        return self.n.eq(other.n)
    def __repr__(self):
        return str(self.n)

def askey(n):
    assert isinstance(n, z3.AstRef)
    return AstRefKey(n)

def get_z3_app(e):
  decl = z3.Z3_get_app_decl(z3.main_ctx().ref(), e.ast)
  return z3.Z3_get_decl_kind(z3.main_ctx().ref(), decl)
