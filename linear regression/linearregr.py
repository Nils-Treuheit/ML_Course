from pathlib import Path
import os,sys,getopt

#---------------------------------------------------------------------#
#~                      define functions                             ~#
#---------------------------------------------------------------------#
#func: create output file and give path
def getOutputFilePath(filename):
    outputFile = ((str(filename).split("\\")[-1]).split(".")[0]+
                  "SolutionFileNT222402.csv")
    newOutputFile = outputFile
    vn = 2
    while os.path.exists(newOutputFile):
        newOutputFile=outputFile.split(".")[0]+"v"+str(vn)+".csv"
        vn+=1
    return newOutputFile  

#func: read in data from file
def readIn(x,y,m,filename):
    file = open(filename,"r")
    i = 0
    for dataString in file:
        dataTable = dataString.strip("\n").split(",")
        x[i][0] = 1
        for j in range(0,m-1): x[i][j+1] = float(dataTable[j].strip())
        y[i] = float(dataTable[m-1].strip())
        i+=1
    file.close()
    return x,y

#func: vector scalar
def scalar(v,s):
    res = [0 for _ in range(len(v))]
    for i in range(0,len(v)): res[i]=v[i]*s
    return res

#func: vector sum
def vSum(vx,vy):
    res = [0 for _ in range(len(vx))]
    if len(vx)==len(vy):
        for i in range(0,len(vx)): res[i]=vx[i]+vy[i]
    return res

#func: hypothesis function
def f(x,w):
    res = 0
    for i in range(len(w)): res+=w[i]*x[i]
    return res

#func: calc sensitivity value
def sensitivityVal(n):
    res = 0
    val = n
    while val<1:
        val=val*10
        res+=1
    return res
#---------------------------------------------------------------------#


def main():

    #argument input
    data = None
    learningRate = None
    threshold = None
    outputFile = None
    argv = sys.argv[1:]

    try:
        opts,args = getopt.getopt(argv, "d:lR:t:of",
                                  ["data=","learningRate=",
                                   "threshold=","outputFile="])
    except getopt.GetoptError as err:
        print(err)
        opts = []
    
    for opt,arg in opts:
        if opt in ['-d','--data']:
            data = arg
        if opt in ['-lR','--learningRate']:
            learningRate = float(arg)
        if opt in ['-t','--threshold']:
            threshold = float(arg)
        if opt in ['-of','--outputFile']:
            outputFile = arg

    
    #conditional parameter input
    if not data:
        data = input("Enter the location of your data file for training: ")
    if not learningRate:
        learningRate = float(input("Enter the learning rate value: "))
    if not threshold:
        threshold = float(input("Enter the threshold value: "))
    if len(opts)<3 and not outputFile:
            try:
                outputFile = input("Enter the path where you want to"+
                                   " save the training solution file or"+
                                   " skip the option by pressing enter: ")
            except ValueError:
                outputFile = None

    if outputFile: outputFile = outputFile.split(".")[0]+".csv"
    if data: data = data.split(".")[0]+".csv"

    #complete file paths
    if not (":" in data):
        filename = Path(str(os.path.dirname(
                        os.path.realpath(__file__)
                        )+"/"+data))
    else: filename = Path(data)

    if not outputFile:
        outputFile = getOutputFilePath(filename)



    #scan input file for data format
    file = open(filename,"r")
    m = len(file.readline().strip("\n").split(","))
    N = sum(1 for line in file)+1
    file.close()
    

    #initalize variables
    it = 0
    E  = 0
    n = learningRate
    s = sensitivityVal(n)
    gradient = [0]*m
    w = [0]*m
    y = [0]*N
    x = [([0]*m) for _ in range(N)]
    
    x,y = readIn(x,y,m,filename)


    #make sure while loop runs properly
    for i in range(N):
        E += (y[i] - f(x[i],w))*(y[i] - f(x[i],w))
    E = E*2
    Eprev = E*2 

    
    outputString = ""
    print("Training ...")

    #---------------------------------------------------------------#
    #+                        Trainings-Loop                       +#
    #---------------------------------------------------------------#
    while (Eprev-E)>threshold:

        #prepare Error values for next threshold check
        Eprev = E
        E = 0

        #calc gradient and Error
        for i in range(N):
            gradient = vSum(gradient,scalar(x[i],(y[i] - f(x[i],w))))
            E += (y[i] - f(x[i],w))*(y[i] - f(x[i],w))
    
        #create string for output file
        outputString += str(it)+","
        for i in w: outputString += ("{0:."+str(s)+"f}").format(i)+","
        outputString += ("{0:."+str(s)+"f}").format(E)+"\n"

        #train with data input and regression                
        w = vSum(w,scalar(gradient,n))

        #prepare variables for next iteration
        gradient = [0]*m
        it+=1
    
    else:
        #print output file
        file = open(outputFile,"w")
        file.write(outputString)
        file.close()
        if Eprev>=E: print("Training was succesful!")
        else: print("Training was unsuccesful! - "+
                    "Maybe you used a learning rate that is too high!")
        print("Results saved in "+
              str(os.path.dirname(os.path.realpath(__file__)))+
              "\\"+outputFile)
    #---------------------------------------------------------------#
        
main()
