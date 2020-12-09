from pathlib import Path
import time,os,sys,getopt,math

#---------------------------------------------------------------------#
#~                      define functions                             ~#
#---------------------------------------------------------------------#
#func: give output file path
def getOutputFilePath(filepath):
    fname = str(str(filepath).split("\\")[-1])
    array = (fname.split("."))
    outputFile = array[0]
    for i in range(1,len(array)-1): string += "."+array[i]
    if (len(array[-1])>4 and len(array)>1): string += "."+array[-1]    
    outputFile += "SolutionFileNT222402.tsv"
    newOutputFile = outputFile
    vn = 2
    while os.path.exists(newOutputFile):
        newOutputFile=outputFile.split(".")[0]+"v"+str(vn)+".tsv"
        vn+=1
    return newOutputFile

#func: complete file path and add right file extension
def getFullFilePath(path):
    array  = path.split(".")
    string = array[0]
    for i in range(1,len(array)-1): string += "."+array[i]
    if (len(array[-1])>4 and len(array)>1): string += "."+array[-1]
    string += ".tsv"
    newPath = (Path(string) if (":" in string) else
               Path(str(os.path.dirname(
                    os.path.realpath(__file__)
                    )+"/"+str(Path(string)))))
    return str(newPath)

#func: convert information to useful data structure
def getDataStructure(InstanceList, ClassifierList, Distribution):
    N  = len(InstanceList)
    c  = len(ClassifierList)
    nc = [0] * c
    xc = [[] for _ in range(c)]
    i = 0
    for classifier in ClassifierList:
        nc[i] = len(Distribution[classifier])
        xc[i] = Distribution[classifier]
        i += 1
    return N,c,nc,xc

#func: give information of base data set
def readIn(filename):
    file = open(filename,"r")
    AttNum = (len(file.readline().strip("\n").strip().split("\t"))-1)
    if (AttNum==0):
        print("Your input file is empty - Sorry, bye!!!")
        time.sleep(2)
        sys.exit()
        return None,None,None,None,None,None,None
    N = 1
    for line in file:
        if (len(line.strip("/n").strip().split("\t"))==(AttNum+1)): N+=1
    file.close()
    InstanceList   = [[] for _ in range(N)]
    Distribution   = {}
    ClassifierList = []
    file = open(filename,"r")
    i = 0
    for line in file:
        dArray = line.strip("\n").strip().split("\t")
        classifier = dArray[0]
        instance = [classifier]
        array = dArray[1:]
        attributes = []
        for e in array:
            attributes.append(float(e))
            instance.append(float(e))
        if AttNum == len(attributes):
            if (classifier not in ClassifierList):
                ClassifierList.append(classifier)
            if not Distribution.get(classifier):
                Distribution[classifier]=[]
            Distribution[classifier].append(attributes)
        InstanceList[i] = instance
        i += 1
    N,c,nc,xc = getDataStructure(InstanceList,ClassifierList,Distribution)
    return InstanceList,N,ClassifierList,c,nc,xc,AttNum
            
#func: calc sigma
def sigSquare(x,u,nci,AttNum):
    res = [0]*AttNum
    for a in range(AttNum):
        sumE = 0
        for k in range(nci):
            sumE += (x[k][a]-u[a])*(x[k][a]-u[a])
        res[a]=sumE/(nci-1)
    return res

#func: calc mu
def mu(x,nci,AttNum):
    res = [0]*AttNum
    for a in range(AttNum):
        sumx = 0
        for k in range(nci):
            sumx += x[k][a]
        res[a] = sumx/nci
    return res

#func: calc P(ci)
def p(c,nci):
    sumnci = 0
    for k in range(c):
        sumnci += nci
    return (nci/sumnci)

#func: P(xa|ci)
def pXaCi(xi,u,s,AttNum):
    res = [0]*AttNum
    for a in range(AttNum):
        res[a] = (math.exp(-1*(((xi[a]-u[a])*(xi[a]-u[a]))/(2*s[a])))/
                  math.sqrt(2*math.pi*s[a]))
    return res

#func: naive bayes
def nb(pXaCi,pci,AttNum):
    res = pci
    for a in range(AttNum):
        res *= pXaCi[a]
    return res

#---------------------------------------------------------------------#
def main():
    
    #argument input
    data = None
    outputFile = None
    argv = sys.argv[1:]

    try:
        opts,args = getopt.getopt(argv, "hd:o:",["help","data=","output="])
    except getopt.GetoptError as err:
        print(err)
        opts = []
    
    for opt,arg in opts:
        if opt in ['-h','--help']:
            print("Your argumented python script call should look like "+
                  "one of the following lines:\n"+
                  "{python} {PythonScriptPath} --data {InputFilePath} "+
                  "--output {OutputFilePath}\n"+
                  "{python} {PythonScriptPath} --data {InputFilePath} \n"+
                  "{python} {PythonScriptPath} -d {InputFilePath} "+
                  "-o {OutputFilePath}\n"+
                  "{python} {PythonScriptPath} -d {InputFilePath} \n"+
                  "{python} {PythonScriptPath}")
            sys.exit()
        elif opt in ['-d','--data']:
            data = arg
        elif opt in ['-o','--output']:
            outputFile = arg

    
    #conditional parameter input
    if not data:
        data = input("Enter the location of your data"+
                     " file for training: ")
    if len(opts)==0 and not outputFile:
        try: outputFile = input("Enter the path where you want to"+
                                " save the training solution file or"+
                                " skip the option by pressing enter: ")
        except ValueError: outputFile = None

    #complete file paths
    if outputFile: outputFile = getFullFilePath(outputFile)
    while True:
        data = getFullFilePath(data)
        if os.path.exists(data): break
        data = input("Please enter a valid path to the location "+
                     "of your data file for training: ")
    
    if not outputFile:
        outputFile = getOutputFilePath(data)

    #initialize the variables
    InstanceList,N,ClassifierList,c,nc,xc,AttNum = readIn(data)
    sigs = [[] for _ in range(c)]
    mus  = [[] for _ in range(c)]
    ps   = [[] for _ in range(c)]
    pCi  = [[] for _ in range(c)]
    NB   = [0]*c

    for i in range(c):
        ps[i]   = [0]*AttNum
        mus[i]  = [0]*AttNum
        sigs[i] = [0]*AttNum
        pCi[i]  = [0]*AttNum
        
    outputString = ''
        
    #---------------------------------------------------------------#
    #+                        Trainings-Loop                       +#
    #---------------------------------------------------------------#
    
    #calc values for eac class
    for i in range(c):
        mus[i]  = mu(xc[i],nc[i],AttNum)
        sigs[i] = sigSquare(xc[i],mus[i],nc[i],AttNum)
        ps[i]   = p(c,nc[i])

        #extend output with a line of all calculated
        #values for this class
        for a in range(AttNum):
            outputString += str(mus[i][a])+"\t"+str(sigs[i][a])+"\t"
        outputString += str(ps[i])+"\n"
        
    #---------------------------------------------------------------#




    #---------------------------------------------------------------#
    #+                      Classification-Loop                    +#
    #---------------------------------------------------------------#

    #count missclassifiactions    
    missclassified = 0   
    for x in InstanceList:
        classifier = 0
        for i in range(c):
            pCi   = pXaCi(x[1:],mus[i],sigs[i],AttNum)
            NB[i] = nb(pCi,ps[i],AttNum)
            if NB[i]>NB[classifier]:
                classifier = i
        if not (x[0]==ClassifierList[classifier]):
            missclassified += 1

    #extend output with the missclassifiaction count
    outputString += str(missclassified)
    
    #---------------------------------------------------------------# 

            
    #print output file
    file = open(outputFile,"w")
    file.write(outputString)
    file.close()
    print("Results saved in "+
          str(os.path.dirname(os.path.realpath(__file__)))+
          "\\"+outputFile)
    time.sleep(2.5)
    sys.exit()
    
main()    
