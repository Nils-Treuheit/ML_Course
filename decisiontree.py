from pathlib import Path
from xml.etree.ElementTree import ElementTree
import xml.etree.ElementTree as etree
import copy,os,sys,getopt,math

#---------------------------------------------------------------------#
#~                      define functions                             ~#
#---------------------------------------------------------------------#
#func: give output file path
def getOutputFilePath(filename):
    outputFile = ((str(filename).split("\\")[-1]).split(".")[0]
                  +"SolutionFileNT222402.xml")
    newOutputFile = outputFile
    vn = 2
    while os.path.exists(newOutputFile):
        newOutputFile=outputFile.split(".")[0]+"v"+str(vn)+".xml"
        vn+=1
    return newOutputFile  

#func: give information of base data set
def readIn(filename):
    file = open(filename,"r")
    AttNum = len(file.readline().strip("\n").split(","))-1
    if (AttNum==0):
        print("Your input file is empty - Sorry, bye!!!")
        return None,None,None,None,None
    N = 1
    for line in file:
        if not (len(line.strip("/n").strip())==0):
            N+=1
    file.close()
    InstanceList=[{} for _ in range(N)]
    AFList=[[] for _ in range(AttNum)]
    ClassifierList = {}
    file = open(filename,"r")
    i = 0
    for line in file:
        array = (line.strip("\n")).split(",")
        classifier = array[-1]
        attributes = array[:-1]
        if AttNum == len(attributes):
            if ClassifierList.get(classifier):
                ClassifierList[classifier]+=1
            else: ClassifierList[classifier]=1 
            j = 0
            for feature in attributes:
               InstanceList[i][("att"+str(j))] = feature
               if feature not in AFList[j]:
                   AFList[j].append(feature)
               j+=1
            InstanceList[i]["classifier"] = classifier
            i+=1       
    file.close()
    return AttNum,len(ClassifierList),AFList,InstanceList,ClassifierList

'''LockedFeatures = [""]*attNum'''
#func: gives a subset of instances for a definite branch
def getSubset(LockedFeatures, InstanceList):
    Subset = InstanceList
    i = 0
    for feature in LockedFeatures:
        if not (feature==""):
            newSubset = []
            for instance in Subset:
                if (instance[("att"+str(i))]==feature):
                    newSubset.append(instance)
            Subset = newSubset
        i+=1   
    Classdistribution = {}
    for instance in Subset:
        classifier = instance["classifier"]
        if Classdistribution.get(classifier):
            Classdistribution[classifier]+=1
        else: Classdistribution[classifier]=1
    #print(Classdistribution)
    return Subset

#func: get Attribute-Feature Distribution of a Subset based
#      on the Classification of instances
def infoSubset(LockedFeatures, Subset, AttNum):  
    AFCDList = {}
    Classdistribution = {}
    for i in range(AttNum):
        if (LockedFeatures[i]==""):
            AFCDList[("att"+str(i))]={}
    for instance in Subset:
        classifier = instance["classifier"]
        if Classdistribution.get(classifier):
            Classdistribution[classifier]+=1
        else: Classdistribution[classifier]=1
        for att in instance:
            if att in AFCDList:
                feature = instance[att]
                if AFCDList[att].get(feature):
                    if AFCDList[att][feature].get(classifier):
                        AFCDList[att][feature][classifier]+=1
                    else: AFCDList[att][feature][classifier]=1
                else:
                    AFCDList[att][feature]={}
                    AFCDList[att][feature][classifier]=1
    return AFCDList,Classdistribution

#func: to calculate entropy of each feature and attribute in a subset 
def calcEntropy(AFDBCList,N,A,C):
    attNum = 0
    eA  = [0]*A
    eAF = [{} for _ in range(A)]
    for att in AFDBCList:
        #print("\n")
        for feat in AFDBCList[att]:
            eFD = [0]*C
            eF  = 0
            fSum = 0 
            cNum = 0
            for Class in AFDBCList[att][feat]:
                fSum += AFDBCList[att][feat][Class]
            for Class in AFDBCList[att][feat]:
                eFD[cNum]=((-1)*((AFDBCList[att][feat][Class])/fSum)*
                    math.log((AFDBCList[att][feat][Class])/fSum,C))
                cNum += 1
            for e in eFD: eF+=e
            #print(str(feat)+" Entropy:"+str(eF))
            eAF[attNum][feat]=eF
            eA[attNum] += (fSum/N)*eF
        #print(att+" Entropy:"+str(eA[attNum]))
        attNum+=1
    return eA,eAF

#func: to calculate information gain of a subset 
def calcInformationGain(eA,ES,A):
    IG=[0]*A
    for attNum in range(A):
        IG[attNum] = ES-eA[attNum]
    return IG   
    
#func: to calculate entropy of a subset 
def calcDataSetEntropy(CDist,N,C):
    ES = 0
    for c in CDist:
        ES+=(-1)*(CDist[c]/N)*math.log(CDist[c]/N,C)
    return ES

#func: to initialize root node
def initRoot(LockedFeatures,InstanceList,AttNum,C):
    Subset = getSubset(LockedFeatures, InstanceList)
    N = len(Subset)
    AFDBCList,CDist = infoSubset(LockedFeatures, Subset, AttNum)
    ES = calcDataSetEntropy(CDist,N,C)
    attrib={"entropy":str(ES)}
    root = etree.Element("tree",attrib)
    return root
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

    if outputFile: outputFile = outputFile.split(".")[0]+".xml"
    if data: data = data.split(".")[0]+".csv"

    #complete file paths
    filename = ( Path(data)
                 if (":" in data) else
                 Path(str(os.path.dirname(
                        os.path.realpath(__file__)
                        )+"/"+data)))

    if not outputFile:
        outputFile = getOutputFilePath(filename)

    #initialize the variables
    AttNum,C,AFList,InstanceList,ClassifierList = readIn(filename)
    LockedFeatures = [""]*AttNum
    root = initRoot(LockedFeatures,InstanceList,AttNum,C)
    
    #get stacks for the loop ready
    stack = []
    stack.append(LockedFeatures)
    treeStack = []
    treeStack.append(root)

    #---------------------------------------------------------------#
    #+                        Trainings-Loop                       +#
    #---------------------------------------------------------------#
    while (len(stack)>0):
        A = AttNum
        LockedFeatures =stack.pop()
        for feat in LockedFeatures:
            if not (feat==""): A-=1 
        Subset = getSubset(LockedFeatures, InstanceList)
        N = len(Subset)
        AFDBCList,CDist = infoSubset(LockedFeatures, Subset, AttNum)
        ES = calcDataSetEntropy(CDist,N,C)
        EA,EAF = calcEntropy(AFDBCList,N,A,C)
        IG = calcInformationGain(EA,ES,A)
        splitAtt = 0
        for i in range(len(IG)):
            if IG[i]>IG[splitAtt]: splitAtt = i
        sAtt = int(str(list(AFDBCList.keys())[splitAtt])[-1:])
        mainTree = treeStack.pop()
        for i in range(len(AFList[sAtt])):
            feat = AFList[sAtt][i]
            eF = EAF[splitAtt][feat]
            newLockedFeatures = copy.deepcopy(LockedFeatures)
            attrib={
                "entropy":str(eF),
                "feature":("att"+str(sAtt)),
                "value":feat
                }
            currentTree=etree.SubElement(mainTree,"node",attrib)
            newLockedFeatures[sAtt]=feat
            if (eF==0):
                subset = getSubset(newLockedFeatures,InstanceList)
                _,cDist = infoSubset(newLockedFeatures,subset,AttNum)
                currentTree.text = list(cDist.keys())[0]
            else:
                stack.append(newLockedFeatures)
                treeStack.append(currentTree)
    else:
        #print tree to xml File
        Tree = ElementTree(root)
        Tree.write(open(outputFile,"w"), encoding='unicode')
        print("Results saved in "+
          str(os.path.dirname(os.path.realpath(__file__)))+
          "\\"+outputFile)
    #---------------------------------------------------------------#
    
main()    
