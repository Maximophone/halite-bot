from subprocess import call
import os
import time

n_hlt_to_keep = 100

while True:
	result = call(['./halite_mod', '-t', '-d', '15 15', 'python RLBot1.py', 'python3 RandomBot2.py'])
	hlt_files = [x for x in os.listdir('.') if x[-4:]=='.hlt']
	while len(hlt_files) > n_hlt_to_keep:
		os.remove(hlt_files.pop(0))
	time.sleep(3)
