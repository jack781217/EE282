import os

print '>>>>> EE282 PA2 Part 1 <<<<<'
apps = ['blackscholes', 'fluidanimate', 'streamcluster', 'swaptions', 'art']


for a in apps:
	for c in range(1,9):
		if a == 'fluidanimate' and not c in [1,2,4,8]:
			continue
		else:
			os.system('zsim.sh -B -a %s -c %d' % (a, c))
			print 'app: %s, core: %d' %(a,c)


