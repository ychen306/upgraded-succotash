from flask import Flask, request, jsonify
import subprocess
from tempfile import NamedTemporaryFile
import os
import json
from gen_enumerator import emit_everything, make_fully_connected_graph, ConcreteInst

from z3_utils import deserialize_z3_expr

app = Flask(__name__)

def check_synth_batched(insts_batch, target, liveins, timeout):
  batch_size = len(insts_batch)
  c_files = [NamedTemporaryFile(mode='w', suffix='.c') for _ in range(batch_size)]
  exe_files = [NamedTemporaryFile(delete=False) for _ in range(batch_size)]

  compilations = []
  for insts, f, exe in zip(insts_batch, c_files, exe_files):
    g, nodes = make_fully_connected_graph(
        liveins=liveins,
        insts=[ConcreteInst(inst, imm8=None) for inst in insts],
        num_levels=4)
    try:
      emit_everything(target, g, nodes, f)
    except:
      compilations.append(None)
      continue
    f.flush()
    exe.close()
    compilations.append(subprocess.Popen('cc %s insts.o -o %s -I. -no-pie 2>/dev/null' % (f.name, exe.name), shell=True))

  synth_jobs = []
  for compilation, exe in zip(compilations, exe_files):
    if compilation is None:
      synth_jobs.append(None)
      continue
    compilation.communicate()
    synth_jobs.append(subprocess.Popen(['timeout', timeout, exe.name], stdout=subprocess.PIPE))

  results = []
  for i, synth_job in enumerate(synth_jobs):
    if synth_job is None:
      results.append(0)
    else:
      out = synth_job.stdout.readline()
      synthesized = len(out) > 1
      results.append(1.0 if synthesized else 0.0)
      synth_job.kill()
    c_files[i].close()
    exe_files[i].close()

  return results

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
