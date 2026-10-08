#!/usr/bin/env python3
"""Compare LP constraint and model parsing with pinned GLOP."""
import random, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'target/native/lp_parser_reference_adapter'; RUST=ROOT/'target/debug/examples/lp_parser_trace'
def run(exe,mode,text): return subprocess.run([exe],input=mode+'\n'+text+'\n',text=True,capture_output=True,check=True).stdout
def compare(mode,text):
 a=run(NATIVE,mode,text); b=run(RUST,mode,text)
 if a!=b: raise AssertionError(f'{mode} {text!r}:\n{a!r}\n!=\n{b!r}')
def main():
 constraints=['x<=1','0<=x<=1','x=2','2=x','r: -inf < -2*x + 3*y >= inf','x+x<=1','',
  'x','1 x <= 2','1 <= <= 2','1 = x <= 2','x <= nope','x <= 1 junk','r[]: .5*x > -1e2']
 for text in constraints: compare('C',text)
 models=['','min: x;','MAX: 1 + 2*x - y; x>=0; int x,y;','min: x; bin x; x<=.5;',
  'min: x+x;','min: x; 1<=x<=0;','r: x+y<=1; max: y;','min: 1 1*e2;','min: 1 1e2;',
  'min: x; a: x<=1; a: x>=0;','int: x, y; min: x+y;']
 for text in models: compare('L',text)
 r=random.Random(0x1F)
 names=['x','y','z','a[1]','foo_bar','q)']
 for _ in range(1000):
  terms=[]
  for name in r.sample(names,r.randrange(0,4)):
   terms.append(r.choice(['','+','-','2*','-3 '])+name)
  left=r.choice(['','-inf <= ','0 < ','2 >= ']); right=r.choice([' <= 5',' > -2',' = 1',''])
  compare('C',r.choice(['','r: '])+left+' '.join(terms)+right)
 for _ in range(1000):
  objective=r.choice(['min: ','max: '])+r.choice(['','1 + '])+' + '.join(r.sample(names,r.randrange(0,4)))
  lines=[objective]
  for _ in range(r.randrange(0,5)):
   name=r.choice(names); lines.append(r.choice([f'{name} >= -2',f'0 <= {name} <= 3',f'int {name}',f'bin: {name}']))
  compare('L','; '.join(lines)+';')
 print('2024 LP-parser traces agree')
if __name__=='__main__': main()
