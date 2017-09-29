import os

for i in range(0, 30):
	str_exec = 'python3 logreg.py --fn logreg_eta_test.csv --eta ' + str(.03*i)
	os.system(str_exec)
