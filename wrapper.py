import os
import sys
args = ' '.join(map(str,sys.argv[1:]))
command = f'mpirun -np 4 python 19_regularization.py {args}'
print(command)
os.system(command)
