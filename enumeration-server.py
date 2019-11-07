from flask import Flask, request, jsonify
import subprocess
from tempfile import NamedTemporaryFile
import os
import json

app = Flask(__name__)

@app.route("/", methods=['POST'])
def hello():
  job = json.loads(request.data)
  contents = job['files']
  timeout = job['timeout']
  dir = job['dir']
  exes = [NamedTemporaryFile(delete=False) for _ in contents]
  c_files = [NamedTemporaryFile(mode='w', suffix='.c') for _ in contents]

  for content, f in zip(contents, c_files):
    f.write(content)
    f.flush()

  lib = dir + '/' + 'insts.o'
  compilations = [subprocess.Popen(['cc', '-o', exe.name, f.name, lib, '-I'+dir, '-no-pie'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      for exe, f in zip(exes, c_files)]

  enumerations = []
  for compilation, exe in zip(compilations, exes):
    compilation.communicate()
    exe.close()
    enumerations.append(subprocess.Popen(['timeout', timeout, exe.name], stdout=subprocess.PIPE))

  results = []
  for enum, exe in zip(enumerations, exes):
    out = enum.stdout.readline()
    synthesized = len(out) > 1
    enum.kill()
    if os.path.exists(exe.name):
      os.unlink(exe.name)
    results.append(synthesized)

  return jsonify(results)

if __name__ == "__main__":
  import sys
  port = sys.argv[1]
  app.run(host='0.0.0.0', port=port)
