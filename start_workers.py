import os
import multiprocessing
import subprocess

os.chdir('/home/nick/multi_machine')
print(os.getcwd())
cores = multiprocessing.cpu_count()
for i in range(int(cores)):
    #os.system('rq worker --url redis://192.168.1.127 &')
    #subprocess.Popen(['rq','worker','--url','redis://192.168.1.127', '>', '/tmp/worker.log', '2>&1'])
    #> /tmp/worker.log 2>&1 
    subprocess.call(['rq worker --url redis://192.168.1.127 &'])