from pathlib import Path
import os,sys,getopt
from matplotlib import pyplot as plt
import numpy as np
xPlot  = []
yPlot  = []

def plotTwoBinClassLinPercep(y,x,xPlot,yPlot,P1,P2,iteration):
    fig, axs = plt.subplots(3)
    plt.tight_layout()
    N = len(xPlot)

    xAPlot,xBPlot = ([] for i in range(2))
    yAPlot,yBPlot = ([] for i in range(2))

    (xAP1SPlot,xBP1SPlot,xAP1TPlot,xBP1TPlot,xAP2TPlot,xBP2TPlot,
     xAP2SPlot,xBP2SPlot) = ([] for i in range(8))
    (yAP1SPlot,yBP1SPlot,yAP1TPlot,yBP1TPlot,yAP2TPlot,yBP2TPlot,
     yAP2SPlot,yBP2SPlot) = ([] for i in range(8))
    
    for i in range(N):
        if(y[i]==1):
            xAPlot.append(xPlot[i])
            yAPlot.append(yPlot[i])
            if(P1.y(P1.o(x[i]))==1):
                xAP1TPlot.append(xPlot[i])
                yAP1TPlot.append(yPlot[i])
            else:
                xBP1TPlot.append(xPlot[i])
                yBP1TPlot.append(yPlot[i])
            if(P2.y(P2.o(x[i]))==1):
                xAP2TPlot.append(xPlot[i])
                yAP2TPlot.append(yPlot[i])
            else:
                xBP2TPlot.append(xPlot[i])
                yBP2TPlot.append(yPlot[i])
        else:
            xBPlot.append(xPlot[i])
            yBPlot.append(yPlot[i])
            if(P1.y(P1.o(x[i]))==1):
                xAP1SPlot.append(xPlot[i])
                yAP1SPlot.append(yPlot[i])
            else:
                xBP1SPlot.append(xPlot[i])
                yBP1SPlot.append(yPlot[i])
            if(P2.y(P2.o(x[i]))==1):
                xAP2SPlot.append(xPlot[i])
                yAP2SPlot.append(yPlot[i])
            else:
                xBP2SPlot.append(xPlot[i])
                yBP2SPlot.append(yPlot[i])

    # 10 linearly spaced numbers
    a = np.linspace(min(xPlot),max(xPlot),abs(max(xPlot)-min(xPlot)))
    # computing the values of (a*P1.w1+P1.w0)/P2.w2
    b = (a*P1.weights[1]+P1.weights[0])/-P1.weights[2]
    # computing the values of (a*P2.w1+P2.w0)/P2.w2
    c = (a*P2.weights[1]+P2.weights[0])/-P2.weights[2]

    axs[0].scatter(xAPlot,yAPlot,c='red',marker='^')
    axs[0].scatter(xBPlot,yBPlot,c='blue',marker='s')
    axs[0].set_title('Data Set')
    
    axs[1].scatter(xAP1SPlot,yAP1SPlot,c='red',marker='s')
    axs[1].scatter(xBP1SPlot,yBP1SPlot,c='blue',marker='s')
    axs[1].scatter(xAP1TPlot,yAP1TPlot,c='red',marker='^')
    axs[1].scatter(xBP1TPlot,yBP1TPlot,c='blue',marker='^')
    axs[1].plot(a,b,'k')
    axs[1].set_title(('After Iteration {0} with a constant'+
                     ' learning rate').format(iteration))
    axs[2].scatter(xAP2SPlot,yAP2SPlot,c='red',marker='s')
    axs[2].scatter(xBP2SPlot,yBP2SPlot,c='blue',marker='s')
    axs[2].scatter(xAP2TPlot,yAP2TPlot,c='red',marker='^')
    axs[2].scatter(xBP2TPlot,yBP2TPlot,c='blue',marker='^')
    axs[2].set_title(('Iteration {0} with a annealing'+
                     ' learning rate').format(iteration))
    axs[1].plot(a,c,'k')
    plt.show()
    

#---------------------------------------------------------------------#
#~                      define functions                             ~#
#---------------------------------------------------------------------#
#func: create output file and give path
def getOutputFilePath(filename):
    outputFile = ((str(filename).split("\\")[-1]).split(".")[0]+
                  "SolutionFileNT222402.tsv")
    newOutputFile = outputFile
    vn = 2
    while os.path.exists(newOutputFile):
        newOutputFile=outputFile.split(".")[0]+"v"+str(vn)+".tsv"
        vn+=1
    return newOutputFile

#func: analyze data structure
def getDatasetInfo(filename,splitChar):
    file = open(filename,"r")
    m = len(file.readline().strip().strip("\n").split(splitChar))
    if (m==0):
        print("Your input file is empty - Sorry, bye!!!")
        file.close()
        return None,None
    N = 1
    for line in file:
         if (len(line.strip("/n").strip().split(splitChar))>=m): N+=1
    file.close()
    return m,N

#func: read in data from file
def readIn(x,y,m,filename,splitChar):
    file = open(filename,"r")
    i = 0
    for dataString in file:
        dataTable = dataString.strip("\n").split(splitChar)
        x[i][0] = 1
        for j in range(1,m): x[i][j] = float(dataTable[j].strip())
        xPlot.append(float(dataTable[1].strip()))
        yPlot.append(float(dataTable[2].strip()))    
        y[i] = float(dataTable[0].strip()=="A")
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

#class: SecondLayerUnit-Class
class LastLayerUnit:
    threshold = 0
    weights   = []
    gradient  = []
    PLU       = []
    E         = 0
    training  = True

    #constructor: second-layer-unit
    def __init__(self,t,PLU):
        self.threshold = t
        self.weights   = [0]*(len(PLU)+1)
        self.gradient  = [0]*(len(PLU)+1)
        self.dValue    = 0
        self.PLU       = PLU
        self.E         = 0

    #func: hypothesis function
    def o(self,x):
        res = 0
        for i in range(len(self.weights)): res+=self.weights[i]*x[i]
        return res

    #func: dValue-update
    def d(self,t,x):
        self.dValue = (self.y(self.o(x))*(1-self.o(x))*
                       (t-self.y(self.o(x))))
        return self.dValue

    #func: output signal
    def y(self,o):
        if o>self.threshold: return 1
        return -1

    #func: train weights with batch approach and backpropagation 
    def train(self,t,x,n):
        self.resetError()
        if(self.training):
            yPLUi = [1]*(len(self.PLU)+1)
            for i in range(len(t)):
                for j in range(len(self.PLU)):
                    yPLUi[j+1]=self.PLU[j].y(self.PLU[j].o(x[i]))
                self.gradient = vSum(self.gradient,scalar(yPLUi,
                                     self.d(t[i],yPLUi)))
                self.E += ((t[i] - self.y(self.o(yPLUi)))*
                           (t[i] - self.y(self.o(yPLUi))))
            self.weights = vSum(self.weights,scalar(self.gradient,n))
            for i in range(len(self.PLU)):
                self.PLU[i].backprop(t,x,n)
        return self.E
        
    #func: reset error vlues for next batch training    
    def resetError(self):
        if(self.training):
            self.E = 0
            for i in range(len(self.gradient)): self.gradient[i]=0
        return


#class: FirstLayerUnit-Class
class FirstLayerUnit:
    weights   = []
    gradient  = []
    NLU       = []
    E         = 0
    training  = True

    #constructor: first-layer-unit
    def __init__(self,w):
        self.weights   = w
        self.gradient  = [0]*len(w)

    #func: connect next-layer-units
    def connect(self,NLU):
        self.NLU = NLU

    #func: hypothesis function
    def o(self,x):
        res = 0
        for i in range(len(self.weights)): res+=self.weights[i]*x[i]
        return res

    #func: d-update
    def d(self,t,x):
        sumD = 0
        j = 0
        for P in self.NLU:
            sumD+=P.dValue*P.weights[j]
            j+=1
        return self.y(self.o(x))*(1-self.o(x))*sumD

    #func: output signal
    def y(self,o):
        return np.tanh(o)

    #func: train weights with batch approach
    def backprop(self,t,x,n):
        self.resetError()
        if(self.training):
            for i in range(len(t)):
                self.gradient = vSum(self.gradient,scalar(x[i],
                                     self.d(t[i],x[i])))
                self.E += ((t[i] - self.y(self.o(x[i])))*
                           (t[i] - self.y(self.o(x[i]))))
            self.weights = vSum(self.weights,scalar(self.gradient,n))        
        return self.E
        
    #func: reset error vlues for next batch training    
    def resetError(self):
        if(self.training):
            self.E = 0
            for i in range(len(self.gradient)): self.gradient[i]=0
        return
    
#---------------------------------------------------------------------#        

def main():

    #argument input
    data = None
    output = None
    iteration = None
    learningRate = None
    threshold = None
    argv = sys.argv[1:]

    try:
        opts,args = getopt.getopt(argv, "hd:o:i:lR:t",
                                  ["help","data=","iterations=",
                                   "output=","learningRate=",
                                   "threshold="])
    except getopt.GetoptError as err:
        print(err)
        opts = []
    
    for opt,arg in opts:
        if opt in ['-h','--help']:
            print("Your argumented python script call should "+
                  "look like one of the examples in the following lines:\n"+
                  "{python} {PythonScriptPath} --data {InputFilePath} "+
                  "--output {OutputFilePath}\n"+
                  "{python} {PythonScriptPath} --data {InputFilePath} \n"+
                  "{python} {PythonScriptPath} -d {InputFilePath} "+
                  "-o {OutputFilePath}\n"+
                  "{python} {PythonScriptPath} -d {InputFilePath} \n"+
                  "{python} {PythonScriptPath}\n"+
                  "\nall optional arguments: \n"+
                  "[path of inputfile with dataset] --data         "+
                  "-d  {String}\n"+
                  "[desired path of outfile]        --output       "+
                  "-o  {String}\n"+
                  "[maximum number of iterations]   --iterations   "+
                  "-i  {int}\n"+
                  "[initial learning rate]          --learningRate "+
                  "-lR {float}\n"+
                  "[perceptron threshold]           --threshold    "+
                  "-t  {float}")
            sys.exit()
        if opt in ['-d','--data']:
            data = arg
        if opt in ['-o','--output']:
            output = arg
        if opt in ['-i','--iterations']:
            iteration = int(arg)
        if opt in ['-lR','--learningRate']:
            learningRate = float(arg)
        if opt in ['-t','--threshold']:
            threshold = float(arg)
        

    
    #conditional parameter input
    if not data:
        data = input("Enter the location of your data file for training: ")
    if len(opts)<1:
        if not output:
            try:
                output = input("Enter the path where you want to"+
                               " save the training solution file or"+
                               " skip the option by pressing enter: ")
            except ValueError:
                output = None
        if not iteration:
            try:
                iteration = int(input("Enter the maximum iteration value"+
                                        " or skip the option by pressing"+
                                        " enter: "))
            except ValueError:
                iteration = None
        if not threshold:
            try:
                threshold = float(input("Enter the initial threshold value"+
                                        " or skip the option by pressing"+
                                        " enter: "))
            except ValueError:
                threshold = None
        if not learningRate:
            try:
                learningRate = float(input("Enter the initial learning rate"+
                                           " value or skip the option by"+
                                           " pressing enter: "))
            except ValueError:
                output = None
        
    if output: output = output.split(".")[0]+".tsv"
    if data:
        splitChar = ""
        dataInfo  = data.split(".")
        if len(dataInfo)>0:
            if dataInfo[-1].strip("\n")=="csv":
                data = dataInfo[0]+".csv"
                splitChar = ","
            else:
                data = dataInfo[0]+".tsv"
                splitChar = "\t"
        else:
            data = dataInfo[0]+".tsv"
            splitChar = "\t"

    #complete file paths
    if not (":" in data):
        filename = Path(str(os.path.dirname(
                        os.path.realpath(__file__)
                        )+"/"+data))
    else: filename = Path(data)

    if not output:
        output = getOutputFilePath(filename)



    #default values
    if not iteration:
        iteration = 100
    if not learningRate:
        learningRate = 1
    if not threshold:
        threshold = 0

    #initialize values
    m,N  = getDatasetInfo(filename,splitChar)    
    x = [([0]*m) for _ in range(N)]
    y = [0]*N

    #read in dataset
    x,y = readIn(x,y,m,filename,splitChar)

    #create neural network
    FLU1 = FirstLayerUnit([0]*m)
    FLU2 = FirstLayerUnit([0]*m)
    FLUs = [FLU1,FLU2]
    LLU1 = LastLayerUnit(threshold,FLUs)
    LLUs = [LLU1]
    for FLU in FLUs:
        FLU.connect(LLUs)


    OutputString = ""
    print("Training ...")

    #train perceptron weigths
    E=LLU1.train(y,x,learningRate)
    print(LLU1.dValue)
    print(LLU1.weights)
    print(FLU1.weights)
    print(FLU2.weights)
    print(FLU1.E)
    print(FLU2.E)
    print(LLU1.E)
    
    #---------------------------------------------------------------#
    #+                        Trainings-Loop                       +#
    #---------------------------------------------------------------#
    for it in range(iteration+1):

        #train perceptron weigths
        E=LLU1.train(y,x,(learningRate/(it+1)))

        #create String for output file
        OutputString += (str("{0:.0f}".format(E))+"//"+
                         str("{0:.0f}".format(FLU1.E))+"//"+
                         str("{0:.0f}".format(FLU2.E))+"\t")

        if (FLU1.E == 0): FLU1.training = False
        if (FLU2.E == 0): FLU2.training = False
        if (E == 0): LLU1.training = False
        if not(FLU1.training or FLU2.training or LLU1.training): break
    #---------------------------------------------------------------#

    #print output file
    file = open(output,"w")
    file.write(OutputString+"\n")
    file.close()
    print("Results saved in "+
              str(os.path.dirname(os.path.realpath(__file__)))+
              "\\"+output)


    plotTwoBinClassLinPercep(y,x,xPlot,yPlot,FLU1,FLU2,iteration)

main()  
