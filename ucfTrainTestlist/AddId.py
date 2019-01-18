labelFile = 'classInd.txt'
tarfile = 'testlist01.txt'

Dict = {}
with open(labelFile) as F:
  for line in F.readlines():
    if len(line)>5:
      line = line.strip()
      Dict[line.split(' ')[1]] = str(int(line.split(' ')[0]))
res = []
print(Dict)
with open(tarfile) as F:
  for line in F.readlines():
    if len(line)>5:
      line = line.strip()
      res.append(line+' '+Dict[line.split('_')[1]])
with open(tarfile+'.1','w') as F:
  F.write('\n'.join(res))
