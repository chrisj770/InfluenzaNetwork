from nn import InfluenzaNetwork 
import sys, datetime
import itertools, statistics
import pickle

fields = ["EP_POV", "EP_UNEMP", "EP_PCI", "EP_NOHSDP", "EP_AGE65", "EP_AGE17", "EP_DISABL", "EP_SNGPNT", "EP_MINRTY", "EP_LIMENG", "EP_MUNIT", "EP_MOBILE", "EP_CROWD", "EP_NOVEH", "EP_GROUPQ", "EP_UNINSUR"]

def updateRunningAverages(runningAverages, activeFields, error):
    for field in activeFields: 
        if not (field in runningAverages): 
            runningAverages[field] = [error]
        else: 
            runningAverages[field].append(error)
    return runningAverages  
            
def calculateRunningAverages(runningAverages):
    fieldDict = {}
    for fieldList in runningAverages:
        fieldDict[fieldList] = statistics.mean(runningAverages[fieldList])
    return fieldDict
    
def getAllSubsets(myList, startCount, stopCount):
    allSubsets = []
    for i in range(1, len(myList)+1):        
        if i < startCount: continue
        allSubsets.extend(itertools.combinations(myList, i))
        if i >= stopCount: break
    return allSubsets  
        
def testLinearElasticNet(alphaMin, alphaMax, alphaMultip, l1Min, l1Max, l1Step, startCount=1, stopCount=3, dumpLimit=30):
    lowestError, modelTypeString = 10000000, "ElasticNet"
    stats = []
    runningAverages = {}
    outputFilename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + modelTypeString + "_output.txt"
    for shortFields in getAllSubsets(fields, startCount, stopCount):
        alpha = alphaMin
        while alpha <= alphaMax: 
            l1 = l1Min
            while l1 <= l1Max:
                object = InfluenzaNetwork(shortFields, 0.2)
                object.trainLinearElasticNet(alpha, l1)
                error = object.testModel_statistics()
                runningAverages = updateRunningAverages(runningAverages, shortFields, error)
                if (error < lowestError): 
                    stats = [shortFields, alpha, l1]
                    lowestError = error
                    sys.stdout = open(outputFilename, "a")
                    print("Lower Error for Linear Elastic", shortFields, "alpha =", alpha, "l1 =", l1, ":", error)
                    sys.stdout.close()
                if error < dumpLimit: InfluenzaNetwork.exportModel(object)
                l1 += l1Step
            alpha *= alphaMultip
    sys.stdout = open(outputFilename, "a")
    print("LOWEST Error for Linear Elastic", stats[0], "alpha =", stats[1], "l1 =", stats[2], ":", lowestError)
    runningAverageResults = sorted(calculateRunningAverages(runningAverages).items() ,  key=lambda x: x[1]) 
    for key in runningAverageResults:
        print (key[0], ":", key[1])
    sys.stdout.close()

def testLinearRegression(startCount=1, stopCount=3, dumpLimit=30):
    lowestError, modelTypeString = 10000000, "LinearRegression"
    stats = []
    runningAverages = {}
    outputFilename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + modelTypeString + "_output.txt"
    for shortFields in getAllSubsets(fields, startCount, stopCount):
        object = InfluenzaNetwork(shortFields, 0.2)
        object.trainLinearRegression()
        error = object.testModel_statistics()
        runningAverages = updateRunningAverages(runningAverages, shortFields, error)
        if (error < lowestError): 
            stats = [shortFields]
            lowestError = error
            sys.stdout = open(outputFilename, "a")
            print("Lower Error for Linear Regression", shortFields, ":", error)
            sys.stdout.close()
        if error < dumpLimit: InfluenzaNetwork.exportModel(object)
    sys.stdout = open(outputFilename, "a")
    print("LOWEST Error for Linear Regression", stats[0], ":", lowestError)
    runningAverageResults = sorted(calculateRunningAverages(runningAverages).items() ,  key=lambda x: x[1]) 
    for key in runningAverageResults:
        print (key[0], ":", key[1])
    sys.stdout.close()
    
def testSVRLinear(cValueMin, cValueMax, cValueStep, gammaMin, gammaMax, gammaStep, startCount=1, stopCount=3, dumpLimit=30):
    lowestError, modelTypeString = 10000000, "SVRLinear"
    stats = []
    runningAverages = {}
    outputFilename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + modelTypeString + "_output.txt"
    for shortFields in getAllSubsets(fields, startCount, stopCount):
        cValue = cValueMin 
        while cValue <= cValueMax:
            gamma = gammaMin
            while gamma <= gammaMax:    
                object = InfluenzaNetwork(shortFields, 0.2)
                object.trainSVRLinear(cValue, gamma)
                error = object.testModel_statistics()
                runningAverages = updateRunningAverages(runningAverages, shortFields, error)
                if (error < lowestError): 
                    stats = [shortFields, cValue, gamma]
                    lowestError = error
                    sys.stdout = open(outputFilename, "a")
                    print("Lower Error for SVRLinear", shortFields, "C =", cValue, "gamma =", str("{:.2f}".format(gamma)), ":", error)
                    sys.stdout.close()
                if error < dumpLimit: InfluenzaNetwork.exportModel(object)
                gamma += gammaStep
            cValue += cValueStep
    sys.stdout = open(outputFilename, "a")
    print("LOWEST Error for SVRLinear", stats[0], "C =", stats[1], "gamma =", str("{:.2f}".format(stats[2])), ":", lowestError)
    runningAverageResults = sorted(calculateRunningAverages(runningAverages).items() ,  key=lambda x: x[1]) 
    for key in runningAverageResults:
        print (key[0], ":", key[1])
    sys.stdout.close()
    
def testSVRRadial(cValueMin, cValueMax, cValueStep, gammaMin, gammaMax, gammaStep, epsilonMin, epsilonMax, epsilonStep, startCount=1, stopCount=3, dumpLimit=30):
    lowestError, modelTypeString = 10000000, "SVRRadial"
    stats = []
    runningAverages = {}
    outputFilename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + modelTypeString + "_output.txt"
    for shortFields in getAllSubsets(fields, startCount, stopCount):
        cValue = cValueMin 
        while cValue <= cValueMax:
            gamma = gammaMin
            while gamma <= gammaMax:   
                epsilon = epsilonMin
                while epsilon <= epsilonMax:
                    object = InfluenzaNetwork(shortFields, 0.2)
                    object.trainSVRRadial(cValue, gamma, epsilon)
                    error = object.testModel_statistics()
                    runningAverages = updateRunningAverages(runningAverages, shortFields, error)
                    if (error < lowestError): 
                        stats = [shortFields, cValue, gamma, epsilon]
                        lowestError = error
                        sys.stdout = open(outputFilename, "a")
                        print("Lower Error for SVRRadial", shortFields, "C =", cValue, "gamma =", str("{:.2f}".format(gamma)), "epsilon =", str("{:.2f}".format(epsilon)), ":", error)
                        sys.stdout.close()
                    if error < dumpLimit: InfluenzaNetwork.exportModel(object)
                    epsilon += epsilonStep
                gamma += gammaStep 
            cValue += cValueStep
    sys.stdout = open(outputFilename, "a")
    print("LOWEST Error for SVRRadial", stats[0], "C =", stats[1], "gamma =", str("{:.2f}".format(stats[2])), "epsilon =", str("{:.2f}".format(stats[3])), ":", lowestError) 
    runningAverageResults = sorted(calculateRunningAverages(runningAverages).items() ,  key=lambda x: x[1]) 
    for key in runningAverageResults:
        print (key[0], ":", key[1])
    sys.stdout.close()
        
def testLinearRidge(alphaMin, alphaMax, alphaMultip, startCount=1, stopCount=3, dumpLimit=30):
    lowestError, modelTypeString = 10000000, "LinearRidge"
    stats = []
    runningAverages = {}
    outputFilename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + modelTypeString + "_output.txt"
    for shortFields in getAllSubsets(fields, startCount, stopCount):
        alpha = alphaMin   
        while alpha <= alphaMax:
            object = InfluenzaNetwork(shortFields, 0.2)
            object.trainLinearRidge(alpha, False)
            error = object.testModel_statistics()
            runningAverages = updateRunningAverages(runningAverages, shortFields, error)
            if (error < lowestError): 
                stats = [shortFields, alpha]
                lowestError = error
                sys.stdout = open(outputFilename, "a")
                print("Lower Error for Ridge", shortFields, "alpha =", alpha, ":", error)
                sys.stdout.close()
            if error < dumpLimit: InfluenzaNetwork.exportModel(object)
            alpha *= alphaMultip
    sys.stdout = open(outputFilename, "a")
    print("LOWEST Error for Ridge", stats[0], "alpha =", stats[1], ":", lowestError)
    runningAverageResults = sorted(calculateRunningAverages(runningAverages).items() ,  key=lambda x: x[1]) 
    for key in runningAverageResults:
        print (key[0], ":", key[1])
    sys.stdout.close()
    
def testLars(startCount=1, stopCount=3, dumpLimit=30):
    lowestError, modelTypeString = 10000000, "Lars"
    stats = []
    runningAverages = {}
    outputFilename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + modelTypeString + "_output.txt"
    for shortFields in getAllSubsets(fields, startCount, stopCount):
        object = InfluenzaNetwork(shortFields, 0.2)
        object.trainLars()
        error = object.testModel_statistics()
        runningAverages = updateRunningAverages(runningAverages, shortFields, error)
        if (error < lowestError): 
            stats = [shortFields]
            lowestError = error
            sys.stdout = open(outputFilename, "a")
            print("Lower Error for Lars", shortFields, ":", error)
            sys.stdout.close()
        if error < dumpLimit: InfluenzaNetwork.exportModel(object)
    sys.stdout = open(outputFilename, "a")
    print("LOWEST Error for Lars", stats[0], ":", lowestError)
    runningAverageResults = sorted(calculateRunningAverages(runningAverages).items() ,  key=lambda x: x[1]) 
    for key in runningAverageResults:
        print (key[0], ":", key[1])
    sys.stdout.close()
    
def testLinearOrthogonalMatchingPursuit(startCount=1, stopCount=3, dumpLimit=30):
    lowestError, modelTypeString = 10000000, "OrthogonalMatchingPursuit"
    stats = []
    runningAverages = {}
    outputFilename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + modelTypeString + "_output.txt"
    for shortFields in getAllSubsets(fields, startCount, stopCount):
        object = InfluenzaNetwork(shortFields, 0.2)
        object.trainLinearOrthogonalMatchingPursuit()
        error = object.testModel_statistics()
        runningAverages = updateRunningAverages(runningAverages, shortFields, error)
        if (error < lowestError): 
            stats = [shortFields]
            lowestError = error
            sys.stdout = open(outputFilename, "a")
            print("Lower Error for Orthogonal Matching Pursuit", shortFields, ":", error)
            sys.stdout.close()
        if error < dumpLimit: InfluenzaNetwork.exportModel(object)
    sys.stdout = open(outputFilename, "a")
    print("LOWEST Error for Orthogonal Matching Pursuit", stats[0], ":", lowestError)
    runningAverageResults = sorted(calculateRunningAverages(runningAverages).items() ,  key=lambda x: x[1]) 
    for key in runningAverageResults:
        print (key[0], ":", key[1])
    sys.stdout.close()
    
def testMLPRegressor(numLayers, layerMin, layerMax, toleranceMin, toleranceMax, toleranceMultip, iterationsMin, iterationsMax, iterationsMultip, activationFunction='relu', startCount=1, stopCount=3, dumpLimit=30):
    lowestError, modelTypeString = 10000000, "MLPRegressor"
    stats = []
    runningAverages = {}
    outputFilename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + modelTypeString + "_output.txt"
    for shortFields in getAllSubsets(fields, startCount, stopCount):
        layerCurrent = layerMin
        while layerCurrent <= layerMax: 
            layerTuple = ()
            for x in range(numLayers):
                layerTuple += (layerCurrent, )
            tolerance = toleranceMin
            while tolerance <= toleranceMax: 
                iterations = iterationsMin
                while iterations <= iterationsMax:
                    object = InfluenzaNetwork(shortFields, 0.2)
                    object.trainMLPRegressor(layerTuple, tolerance, iterations, activationFunction=activationFunction)
                    error = object.testModel_statistics()
                    runningAverages = updateRunningAverages(runningAverages, shortFields, error)
                    if (error < lowestError): 
                        stats = [shortFields, layerCurrent, tolerance, iterations]
                        lowestError = error
                        sys.stdout = open(outputFilename, "a")
                        print("Lower Error for MLPRegressor", shortFields, "layers =", layerCurrent, "tolerance =", tolerance, "max_iterations =", iterations, "error =", error)
                        sys.stdout.close()
                    if error < dumpLimit: InfluenzaNetwork.exportModel(object)
                    iterations *= iterationsMultip
                tolerance *= toleranceMultip
            layerCurrent += 1
    sys.stdout = open(outputFilename, "a")
    print("LOWEST Error for MLPRegressor", stats[0], "layers =", stats[1], "tolerance =", stats[2], "max_iterations =", stats[3], "error =", lowestError)
    runningAverageResults = sorted(calculateRunningAverages(runningAverages).items() ,  key=lambda x: x[1]) 
    for key in runningAverageResults:
        print (key[0], ":", key[1])
    sys.stdout.close()
        
if __name__=="__main__":    
    testLinearRegression()
    testLinearElasticNet(0.000000001, 100000, 10, 0.1, 1, 0.1)
    testSVRLinear(0.1, 10, 0.5, 0.1, 1, 0.05)