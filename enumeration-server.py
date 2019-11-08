from flask import Flask, request, jsonify
import subprocess
from tempfile import NamedTemporaryFile
import os
import json
from gen_enumerator import emit_everything, make_fully_connected_graph, ConcreteInst
from synth import check_synth_batched

from z3_utils import deserialize_z3_expr

app = Flask(__name__)

@app.route("/", methods=['POST'])
def hello():
  job = json.loads(request.data)
  insts_batch = job['insts_batch']
  target = deserialize_z3_expr(job['target'])
  timeout = job['timeout']

  liveins = job['liveins']
  results = check_synth_batched(insts_batch, target, liveins, timeout)
  return jsonify(results)
  

if __name__ == "__main__":
  import sys
  port = sys.argv[1]
  app.run(host='0.0.0.0', port=port)
