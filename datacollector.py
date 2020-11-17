## Simple script that compiles the results in a csv file for analysis from output logs of the algorithm
output = open('OUTPUT/RESULTS.csv','w')
output.write('INDEX,METRIC,NOISE TYPE,NOISY,GAUSS FILTER,NLM FILTER\n')
for index in range(1, 12):
    f = open('OUTPUT/LOGS/' + str(index) + '-LOG.csv','r')

    f.readline()
    f.readline()
    f.readline()

    output.write(str(index) + ',GAUSSIAN,PSNR,' + f.readline())
    output.write(str(index) + ',GAUSSIAN,MSE,' + f.readline())
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    output.write(str(index) + ',S&P,PSNR,' + f.readline())
    output.write(str(index) + ',S&P,MSE,' + f.readline())


    f.close()