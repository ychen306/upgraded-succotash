from flask import Flask, request, jsonify
import subprocess
from tempfile import NamedTemporaryFile

app = Flask(__name__)

@app.route("/")
def hello():
  files = request.args['files'].split(',')
  timeout = request.args['timeout']
  dir = request.args['dir']
  exes = [NamedTemporaryFile() for _ in files]
  lib = dir + '/' + 'insts.o'
  compilations = [subprocess.Popen(['cc', '-o', exe.name, f, lib, '-I'+dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      for exe, f in zip(exes, files)]

  enumerations = []
  for compilation, exe in zip(compilations, exes):
    compilation.communicate()
    enumerations.append(subprocess.Popen(['timeout', timeout, exe.name], stdout=subprocess.PIPE))

  results = []
  for enum in enumerations:
    out = enum.stdout.readline()
    synthesized = len(out) > 1
    enum.kill()
    results.append(synthesized)

  return jsonify(results)

if __name__ == "__main__":
  import sys
  port = sys.argv[1]
  app.run(port=port)