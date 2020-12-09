from pathlib import Path
import time,os,sys,getopt,math
from matplotlib import pyplot as plt
import numpy as np
import multiprocessing
xPlot,yPlot,cPlot,pPlot,CBPlot = ([] for _ in range(5))


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
        cPlot.append(classifier)
        xPlot.append(attributes[0])
        yPlot.append(attributes[1])
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
    if (farthest.distance==nearest.distance): return 1
    return ((farthest.distance-current.distance)/
     (farthest.distance-nearest.distance))  

#func: trains casebase with the IB2-Algo 
def IB2(Instances,k,plot=False):
    MCCounter = 1
    CB=[Instances[0]]
    for i in range(1, len(Instances)):
        neighbors  = kNearest(CB,Instances[i],k)
        classifier = classify(neighbors)
        if not (classifier == Instances[i].classifier):
            MCCounter+=1
            CB.append(Instances[i])
            if plot: CBPlot.append(i)
    MCCounter = max(MCCounter-k,0)
    return MCCounter,CB

#func: runs casebase based classification on the data-set
def CBClassify(Instances,k,CB,plot=False):
    MCCounter = 0
    if plot: pPlot.append(Instances[0].classifier)
    for i in range(1, len(Instances)):
        neighbors  = kNearest(CB,Instances[i],k)
        classifier = classify(neighbors)
        if plot: pPlot.append(classifier)
        if not (classifier == Instances[i].classifier):
            MCCounter+=1
    return MCCounter

#func: for parallel processing
def IBp(tuple_input):
    Instances,it,q = tuple_input
    _,CBi  = IB2(Instances,it)
    miclco = CBClassify(Instances,it,CBi)
    q.put((it,len(CBi),miclco))
    return

#---------------------------------------------------------------------#

#---------------------------------------------------------------------#
#           Functions for realizing extra functionality               #
#---------------------------------------------------------------------#
#func: for plotting solutions
def plotKNN(mc):
    fig, axs = plt.subplots(2)
    plt.tight_layout()
    N = len(xPlot)

    xAPlot,xBPlot = ([] for _ in range(2))
    yAPlot,yBPlot = ([] for _ in range(2))

    (xASPlot,xBSPlot,xCSPlot,
     xATPlot,xBTPlot,xCTPlot) = ([] for _ in range(6))
    (yASPlot,yBSPlot,yCSPlot,
     yATPlot,yBTPlot,yCTPlot) = ([] for _ in range(6))

    j = 0
    for i in range(N):
        if(cPlot[i]=="A"):
            xAPlot.append(xPlot[i])
            yAPlot.append(yPlot[i])
            if (j<len(CBPlot)) and (CBPlot[j]==i):
                xCTPlot.append(xPlot[i])
                yCTPlot.append(yPlot[i])
                j+=1
            elif(pPlot[i]=="A"):
                xATPlot.append(xPlot[i])
                yATPlot.append(yPlot[i])
            else:
                xBTPlot.append(xPlot[i])
                yBTPlot.append(yPlot[i])
        else:
            xBPlot.append(xPlot[i])
            yBPlot.append(yPlot[i])
            if (j<len(CBPlot)) and (CBPlot[j]==i):
                xCSPlot.append(xPlot[i])
                yCSPlot.append(yPlot[i])
                j+=1
            elif(pPlot[i]=="A"):
                xASPlot.append(xPlot[i])
                yASPlot.append(yPlot[i])
            else:
                xBSPlot.append(xPlot[i])
                yBSPlot.append(yPlot[i])
                

    axs[0].scatter(xAPlot,yAPlot,c='red',marker='^')
    axs[0].scatter(xBPlot,yBPlot,c='blue',marker='s')
    axs[0].set_title('Data Set')
    
    axs[1].scatter(xASPlot,yASPlot,c='red',marker='s')
    axs[1].scatter(xBSPlot,yBSPlot,c='blue',marker='s')
    axs[1].scatter(xCSPlot,yCSPlot,c='black',marker='s')
    axs[1].scatter(xCTPlot,yCTPlot,c='black',marker='^')
    axs[1].scatter(xATPlot,yATPlot,c='red',marker='^')
    axs[1].scatter(xBTPlot,yBTPlot,c='blue',marker='^')
    axs[1].set_title(('kNN (Error='+str(mc)+
                     ',Accuracy={0:.5f})').format((N-mc)/N))
    plt.show()
    return

#func: run parallel processed func
def run_parallel(InstanceList):
    print("\n\nThese are the results of the optima "+
          "calculation(with multiprocessing):")
    print("----------------------------------------"+
          "-----------------------------")
    start = time.time()
    restups = []
    cores = os.cpu_count()-2
    it = 1
    for _ in range(int(len(InstanceList)/cores)):
        processes = []
        q = multiprocessing.Queue()
        for _ in range(cores):
            p = multiprocessing.Process(target=IBp, args=[(InstanceList,it,q)])
            p.start()
            processes.append(p)
            it+=1

        for process in processes:
            restups.append(q.get())
            process.join()
            
        q.close()
        q.join_thread()

    resIL,res1VL,res2VL = ([] for _ in range(3))      
    for tup in restups:
        Index,Val1,Val2 = tup
        resIL.append(Index)
        res1VL.append(Val1)
        res2VL.append(Val2)
        
    e1,e2 = (min(res1VL),min(res2VL))
    i1,i2 = (res1VL.index(e1),res2VL.index(e2))

    #k1 = k with smallest IB2-Casebase &
    #k2 = k for which IB2-Casebase-Classification has
    #least missclassifications
    k1 = [resIL[i1],res1VL[i1]] 
    k2 = [resIL[i2],res2VL[i2]]
    
    end=time.time()
    print("Smallest casebase at k="+str(k1[0])+" with a size of "+
          str(k1[1]))
    print("Least missclassifications at k="+str(k2[0])+" with "+
          str(k2[1])+" missclassifications")
    print("Calculation took: "+str(end - start)+"secs")
    return

#---------------------------------------------------------------------#

#---------------------------------------------------------------------#
#             Splitter functions of the main Function                 #
#---------------------------------------------------------------------#
#func: to get all the necessary User Inputs
def getUserInput():
    
    data = None
    outputFile = None
    k = None
    kOc = None

    #argument input
    argv = sys.argv[1:]
    try:
        opts,args = getopt.getopt(argv, "hd:o:k:e",["help",
                                  "data=","output=","k=",
                                    "kOptimaCalculation"])
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
                  "nearest neighbors that should be considered] \n"+
                  "-e --kOptimaCalculation {!!!compute-heavy Option!!!}")
            sys.exit()
        elif opt in ['-d','--data']:
            data = arg
        elif opt in ['-o','--output']:
            outputFile = arg
        elif opt in ['-k','--k']:
            k = arg
        elif opt in ['-e','--kOptimaCalculation']:
            kOc = True

    
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
    if len(opts)==0 and not kOc:
        try:
            Input = input("Enter yes[y] if you want to calculate optimal "+
                       "values(!!!Attention: very compute-heavy!!!) - "+
                       "Skip that option by enter no[n] or pressing enter: ")
            if (Input == "y") or (Input == "yes"): kOc = True
            else: kOc = False 
        except ValueError: kOc = False

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

    return k,kOc,data,outputFile


#func: to solve the main task
def solve(k,data,outputFile):
    
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
    
    _,CB = IB2(InstanceList,k,True)
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

    #additional calculation in preparation of plotting results
    mcc = CBClassify(InstanceList,k,CB,True)

    return mcc,InstanceList  
#---------------------------------------------------------------------#

def main():

    #get the users Input
    k,kOc,data,outputFile = getUserInput()

    #do the main task
    MiClCounter,InstanceList = solve(k,data,outputFile)

    #plot results
    plotKNN(MiClCounter)
    
    #do paralell computing to find optimal ks if wanted
    if kOc:
        run_parallel(InstanceList)
        input("Press any key to exit...")
    else:
        time.sleep(2.5)
        
    #exit
    time.sleep(0.5)
    sys.exit()
    return


if __name__ == "__main__": main()
