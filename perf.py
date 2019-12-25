import glob
import os
import argparse
import subprocess
import getpass
import re
import tzlocal
from datetime import datetime
from datetime import timezone

FMT = '%Y-%m-%dT%H:%M:%S%z'

class Bench:
  def __init__(self, header, table):
    self.header = header
    self.table = table

  def to_s(self):
    return "{} ({})".format(self.filename(), self.ts())

  def set(self, key, value):
    self.header[key] = value

  def get(self, key):
    return self.header[key] if key in self.header else None

  def ts(self): return self.get("ts")
  def filename(self): return self.get("filename")

  def text(self):
    s = ""
    for k, v in self.table.items():
      s += "{:<80} {:8.3f} ms\n".format(k, v)
    for k, v in self.header.items():
      if isinstance(v, datetime):
        s += "# {}: {}\n".format(k, v.strftime(FMT))
      else:
        s += "# {}: {}\n".format(k, v)
    return s

  def print(self):
    print(self.text(), end="")

  @classmethod
  def parse(cls, text):
    header = {}
    t = {}
    for line in text.splitlines():
      if line[0] == "#":
        k, v = line[1:].strip().split(":", 1)
        header[k.strip()] = v.strip()
      else:
        name, time, *others = line.strip().split()
        t[name] = float(time)

    if "ts" in header:
      #header["ts"] = datetime.strptime(header["ts"], '%Y-%m-%dT%H:%M:%S.%f%z')
      header["ts"] = datetime.strptime(header["ts"], FMT)
    return Bench(header, t)

  @classmethod
  def parse_file(cls, filename):
    with open(filename) as f:
      bench = Bench.parse(f.read())
      bench.set("filename", filename)
      return bench

class Diff:
  def __init__(self, a, b, k):
    self.key = k
    self.va = a.table[k]
    if b and k in b.table:
      self.vb = b.table[k]
      self.diff = self.va - self.vb
      self.ratio = self.diff / self.vb * 100
    else:
      self.vb = None
      self.diff = 0
      self.ratio = 0

  def to_s(self):
    if self.vb:
      return "{:<80} {:8.3f} {:8.3f} {:8.3f} {:8.3f} %".format(self.key, self.va, self.vb, self.diff, self.ratio)
    else:
      return "{:<80} {:8.3f}".format(self.key, self.va)

class CompareResult:
  def __init__(self, a, b, diffs, threshold):
    self.a = a
    self.b = b
    self.diffs = diffs
    self.threshold = threshold

  def is_ok(self, threshold):
    return len(self.errors(threshold)) == 0

  def errors(self, threshold):
    e = []
    for d in self.diffs:
      if d.ratio > threshold:
        e.append(d)
    return e

  def print(self, print_all=False):
    header ="{:<80} {:<8} {:<8} {:<8} {:<8}".format("benchmark", 
                                                    "a", "b", "diff", "diff%")
    print("a: {}".format(self.a.to_s()))
    print("b: {}".format(self.b.to_s()))

    if print_all:
      print("All Results:")
      print(header)
      for d in self.diffs:
        print(d.to_s())

    errors = self.errors(self.threshold)
    print("{} errors found".format(len(errors)))
    if len(errors) > 0:
      print(header)
      for d in errors:
        print(d.to_s())

def load(args):
  files = args.files
  if args.dir:
    files.extend(glob.glob(args.dir + "/*"))

  ary = []
  for f in files:
    ary.append(Bench.parse_file(f))

  return sorted(ary, key=lambda x : x.ts()) 

def compare(args, a=None, b=None):
  if not a or not b:
    ary = load(args)
    if a:
      b = ary[-1]
    else:
      a = ary[-1]
      b = ary[-2]

  if not a or not b:
    raise("no enough result")
  else:
    return CompareResult(a, b, [Diff(a, b, k) for k in a.table.keys()], 
                         args.threshold)

def run(args):
  exe = args.exe
  try:
    ret = subprocess.run(exe, check=True, stdout=subprocess.PIPE, 
                         universal_newlines=True)
    text = ret.stdout
    text += "# ts: {}\n".format(datetime.now(tz=tzlocal.get_localzone()).strftime(FMT))
    text += "# hostname: {}\n".format(os.uname().nodename)
    text += "# user: {}\n".format(getpass.getuser())
    return Bench.parse(text)
  except subprocess.CalledProcessError as e:
    print(e)

def save(args, bench, filename):
  with open(filename, "w") as f:
    f.write(bench.text())

# Command

def test(args):
  bench = run(args)
  if not bench:
    return
  result = compare(args, bench)
  result.print(True)
  if args.output:
    print("write result to {}".format(args.output))
    save(args, bench, args.output)

def store(args):
  if len(args.files) == 0:
    return # do nothing

  frm = args.files[0]

  bench = Bench.parse_file(args.files[0])
  to = "{}/{}".format(args.dir, 
                      bench.ts().astimezone(timezone.utc).strftime(FMT))
  print("Copy {} to {}".format(frm, to))
  save(args, bench, to)

def check(args):
  a = None
  b = None
  if args.a:
    a = Bench.parse_file(args.a)
  if args.b:
    b = Bench.parse_file(args.b)
  compare(args, a, b).print(args.verbose > 0)

def show(args):
  ary = load(args)
  #print("show: len={}".format(len(ary)))
  if len(ary) < 0:
    return
  tmp = list(reversed(ary))[0:args.l]
  latest = tmp[0]

  for i, b in enumerate(tmp):
    print("{} {}".format(i, b.to_s()))

  print("{:80}".format(""), end="")
  for i in range(len(tmp)):
    print(" {:<8}".format(i), end="")
  print()

  for k in latest.table.keys():
    print("{:<80}".format(k), end="")
    for b in tmp:
      if k in b.table:
        print(" {:8.3f}".format(b.table[k]), end="")
      else:
        print(" {:8}".format(""), end="")
    print()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--threshold", type=float, default=5.0)
  parser.add_argument("-v", "--verbose", action="count", default=0)
  parser.add_argument("-d", "--dir", type=str, default="perfdb")
  parser.add_argument("-e", "--exe", type=str, default="build/test/bench")
  parser.add_argument("-l", type=int, default=5)
  parser.add_argument("-a", type=str)
  parser.add_argument("-b", type=str)
  parser.add_argument("-o", "--output", type=str)
  parser.add_argument("command", type=str, nargs="?")
  parser.add_argument("files", nargs="*")
  args = parser.parse_args()

  #print(args.command)
  if args.command == "check":
    check(args)
  elif args.command == "parse": # check if file can be parsed
    for f in args.files:
      Bench.parse_file(f)
  elif args.command == "run":
    b = run(args)
    b.print()
  elif args.command == "test": # run and check
    test(args)
  elif args.command == "store":
    store(args)
  elif args.command == "show":
    show(args)

if __name__ == "__main__":
  main()
