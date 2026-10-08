#!/usr/bin/env python3
"""Compare LP decomposition and reconstruction with pinned GLOP."""
import random, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'target/native/lp_decomposer_reference_adapter'
RUST=ROOT/'target/debug/examples/lp_decomposer_trace'
def run(exe,data): return subprocess.run([exe],input=data,text=True,capture_output=True,check=True).stdout
def generate(r):
 n=r.randrange(0,30); m=r.randrange(0,30); maximize=r.randrange(2); entries=[]
 variables=[f'x{c} {r.randrange(2)} {-r.randrange(5)} {r.randrange(5,11)} {(c+1)*.25}' for c in range(n)]
 constraints=[f'r{i} {-r.randrange(5)} {r.randrange(5,11)}' for i in range(m)]
 for row in range(m):
  for col in range(n):
   if r.random()<.12: entries.append(f'{row} {col} {r.choice([-2,-1,.5,1,3])}')
 return f'{n} {m} {len(entries)} {maximize}\n'+'\n'.join(variables+constraints+entries)+'\n'
def main():
 r=random.Random(0xDEC0)
 for case in range(1000):
  data=generate(r); a=run(NATIVE,data); b=run(RUST,data)
  if a!=b: raise AssertionError(f'case {case}:\n{data}\n{a}\n!=\n{b}')
 print('1000 LP-decomposer traces agree')
if __name__=='__main__': main()
