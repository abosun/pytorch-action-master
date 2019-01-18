#! /usr/bin/env python
import os
import sys
path = sys.argv[1]
res = []
with open(path) as f:
    for line in f.readlines():
        res.append(os.path.basename(line.split()[0])+' '+str(1+int(line.split()[-1])))
print(res[:2],res[-2:])
print(len(res))
res = '\n'.join(res)
with open(sys.argv[2],'w') as f:
    f.write(res)
