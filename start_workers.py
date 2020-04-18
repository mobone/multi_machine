import os
import multiprocessing


os.chdir('/home/nick/multi_machine')
print(os.getcwd())
cores = multiprocessing.cpu_count()
for i in range(int(cores)):
    os.system('rq worker --url redis://192.168.1.127 &')