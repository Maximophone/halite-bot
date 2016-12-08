from subprocess import call

while True:
	result = call(['./halite', '-t', '-d', '15 15', 'python RLBot1.py', 'python3 RefactoredBot.py'])
