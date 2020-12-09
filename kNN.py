from pathlib import Path
import time,os,sys,getopt,math

#---------------------------------------------------------------------#
#~                         define class                              ~#
#---------------------------------------------------------------------#
class Point:
    name = ""
    coordinates = [0]*2
    classifier  = 0
    distance = float('inf')
    def __init__(self, xs, c, n):
        self.name = n
        self.classifier  = c
        self.coordinates = xs

    def calcDist(self,point):
        self.distance = 0
        for i in range(len(self.coordinates)):
            self.distance+=((self.coordinates[i]-point[i])*
                            (self.coordinates[i]-point[i]))
        self.distance = math.sqrt(self.distance)
        return self.distance

#---------------------------------------------------------------------#



#---------------------------------------------------------------------#
#~                      define functions                             ~#
#---------------------------------------------------------------------#
#func: give output file path
def getOutputFilePath(filepath,k):
    fname = str(str(filepath).split("\\")[-1])
    array = (fname.split("."))
    outputFile = array[0]
    for i in range(1,len(array)-1): string += "."+array[i]
    if (len(array[-1])>4 and len(array)>1): string += "."+array[-1]    
    outputFile += "-"+str(k)+"NN-SolutionFileNT222402.tsv"
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

#func: gives information about the base data set
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
    InstanceList   = []
    file = open(filename,"r")
    i = 0
    for line in file:
        dArray = line.strip("\n").strip().split("\t")
        classifier = dArray[0]
        array = dArray[1:]
        attributes = []
        for e in array: attributes.append(float(e))
        InstanceList.append(Point(attributes,classifier,str(i)))
        i+=1
    return InstanceList

#func: gives you the k-nearest neighbors to an instance
#      out of a casebase of given instances          
def kNearest(Casebase,Instance,k):
    neighbors = []
    for j in range(k):
        kNearest = [0,float('inf')]
        for i in range(len(Casebase)):
            if ((Casebase[i].calcDist(
                Instance.coordinates)<kNearest[1]) and
                (Casebase[i] not in neighbors)):
                kNearest = [i,Casebase[i].distance]
                
        if(Casebase[kNearest[0]] not in neighbors):
            neighbors.append(Casebase[kNearest[0]])            
    return neighbors

#func: classification decision based on the k-nearest
#      neighbors of an instance
def classify(Neighbors):
    dist = {}
    nearest  = Neighbors[0]
    farthest = Neighbors[-1]
    for instance in Neighbors:
        classifier = instance.classifier
        if not dist.get(classifier):
            dist[classifier] = []
        w=wCalc(instance,farthest,nearest)
        dist[classifier].append(w)
    classifiers = list(dist.keys())
    wSCI = [0,float('-inf')]
    weightSum = 0
    i = 0
    for cl in classifiers:
        weightSum = sum(dist[cl])
        if (weightSum>wSCI[1]):
            wSCI[0] = i
            wSCI[1] = weightSum
        i+=1
    return classifiers[wSCI[0]]

#func: calculates weights for classification
def wCalc(current,farthest,nearest):
    if (farthest==nearest): return 1
    return ((farthest.distance-current.distance)/
     (farthest.distance-nearest.distance))  

#func: trains casebase with the IB2-Algo 
def IB2(Instances,k):
    MCCounter = 1
    CB=[Instances[0]]
    for i in range(1, len(Instances)):
        neighbors  = kNearest(CB,Instances[i],k)
        classifier = classify(neighbors)
        if not (classifier == Instances[i].classifier):
            MCCounter+=1
            CB.append(Instances[i])         
    MCCounter = max(MCCounter-k,0)
    return MCCounter,CB

#func: runs casebase based classification on the data-set
def CBClassify(Instances,k,CB):
    MCCounter = 0
    for i in range(1, len(Instances)):
        neighbors  = kNearest(CB,Instances[i],k)
        classifier = classify(neighbors)
        if not (classifier == Instances[i].classifier):
            MCCounter+=1
    return MCCounter

#---------------------------------------------------------------------#
def main():
    
    #argument input
    data = None
    outputFile = None
    k = None
    argv = sys.argv[1:]

    try:
        opts,args = getopt.getopt(argv, "hd:o:k:",["help",
                                  "data=","output=","k="])
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
                  "{python} {PythonScriptPath}\n"+
                  "additional optional parameter: -k --k [number of "+
                  "nearest neighbors that should be considered]")
            sys.exit()
        elif opt in ['-d','--data']:
            data = arg
        elif opt in ['-o','--output']:
            outputFile = arg
        elif opt in ['-k','--k']:
            k = arg

    
    #conditional parameter input
    if not data:
        data = input("Enter the location of your data"+
                     " file for training: ")
    if len(opts)==0 and not outputFile:
        try: outputFile = input("Enter the path where you want to"+
                                " save the training solution file or"+
                                " skip the option by pressing enter: ")
        except ValueError: outputFile = None
    if len(opts)==0 and not k:
        try: k = int(input("Enter the number of nearest neighbors that "+
                       "should be considered or skip the option "+
                       "by pressing enter: "))
        except ValueError: k = None

    #complete file paths
    if outputFile: outputFile = getFullFilePath(outputFile)
    while True:
        data = getFullFilePath(data)
        if os.path.exists(data): break
        data = input("Please enter a valid path to the location "+
                     "of your data file for training: ")

    if not k: k=4
    if not outputFile:
        outputFile = getOutputFilePath(data,k)

    #initialize the variables
    InstanceList = readIn(data)
        
    outputString = ''
    #---------------------------------------------------------------#
    #+                  Classification-Loop                        +#
    #---------------------------------------------------------------#
    for it in range(2,11,2):
        _,CB   = IB2(InstanceList,it)
        miclco = CBClassify(InstanceList,it,CB)
        outputString+=str(miclco)+"\t"
    outputString=outputString.strip("\t")+"\n"
    
    _,CB = IB2(InstanceList,k)
    for instance in CB:
        outputString+=instance.classifier+"\t"
        for c in instance.coordinates:
            outputString+=str(c)+"\t"
        outputString=outputString.strip("\t")+"\n"
    
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
