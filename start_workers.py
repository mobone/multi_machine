import os
import multiprocessing

os.system('pkill -9 rq')
os.system('cd /home/nick/multi_machine/')
cores = multiprocessing.cpu_count()
for i in range(int(cores)):
    os.system('rq worker --url redis://192.168.1.127 &')