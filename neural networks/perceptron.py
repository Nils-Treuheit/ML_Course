from pathlib import Path
import os,sys,getopt

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

#class: Perceptron-Class
class Perceptron:
    threshold = 0
    weights   = []
    gradient  = []
    E         = 0
    training  = True

    #constructor: perceptron
    def __init__(self,t,w):
        self.threshold = t
        self.weights   = w
        self.gradient  = [0]*len(w)

    #func: hypothesis function
    def o(self,x):
        res = 0
        for i in range(len(self.weights)): res+=self.weights[i]*x[i]
        return res

    #func: output signal
    def y(self,o):
        return o>self.threshold

    #func: train weights with batch approach
    def train(self,t,x,n):
        if(self.training):
            for i in range(len(t)):
                self.gradient = vSum(self.gradient,scalar(x[i],
                                     (t[i] - self.y(self.o(x[i])))))
                self.E += ((t[i] - self.y(self.o(x[i])))*
                           (t[i] - self.y(self.o(x[i]))))
            self.weights = vSum(self.weights,scalar(self.gradient,n))
        output = int(self.E)
        self.resetError()
        return output
        
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
    w = [0]*m

    #read in dataset
    x,y = readIn(x,y,m,filename,splitChar)

    #create Perceptrons
    P1 = Perceptron(threshold,w)
    P2 = Perceptron(threshold,w)


    P1OutputString = ""
    P2OutputString = ""
    print("Training ...")
    
    #---------------------------------------------------------------#
    #+                        Trainings-Loop                       +#
    #---------------------------------------------------------------#
    for it in range(iteration+1):

        #train perceptron weigths
        P1E=P1.train(y,x,learningRate)
        P2E=P2.train(y,x,(learningRate/(it+1)))

        #create String for output file
        P1OutputString += str(P1E)+"\t"
        P2OutputString += str(P2E)+"\t"

        if (P1E == 0): P1.training = False
        if (P2E == 0): P2.training = False
        if not(P1.training or P2.training): break
    #---------------------------------------------------------------#

    #print output file
    file = open(output,"w")
    file.write(P1OutputString+"\n"+P2OutputString+"\n")
    file.close()
    print("Results saved in "+
              str(os.path.dirname(os.path.realpath(__file__)))+
              "\\"+output)

main()  
