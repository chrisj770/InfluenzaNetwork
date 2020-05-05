import numpy as np 
import matplotlib.pyplot as plt
from nn import InfluenzaNetwork
import random

def manualInput(networkObject):
    fieldPrompts = ["Population"]
    fieldPrompts.extend(networkObject.fields)
    fieldInputList = []
    for field in fieldPrompts:
        fieldInput = None
        while fieldInput is None:
            try:
                fieldInput = float(input("Enter " + field + ": "))
                fieldInputList.append(fieldInput)
            except Exception: 
                print("Input must be a floating-point digit!")
                fieldInput = None
    population = fieldInputList[0] 
    fieldInputList = fieldInputList[1:]
    fieldPrompts = fieldPrompts[1:]
    results = networkObject.testModel_custom(np.array([fieldInputList]))
    print("Results for Inputs: " + str(results[0]))
    print("Number of Infections: " + str(float(results[0]) * population))
    
def modelData_barGraph(networkObject, numPoints, bestFit=False):
    allPredictedOutput = networkObject.testOutput
    allRealOutput = networkObject.testModel_output()
    assert len(allPredictedOutput) == len(allRealOutput)
    if bestFit: 
        predicted, real, metadata = _modelData_barGraph_bestFit(networkObject,allPredictedOutput, allRealOutput, numPoints)
    else: 
        predicted, real, metadata = _modelData_barGraph_random(networkObject,allPredictedOutput, allRealOutput, numPoints)
    print(predicted, real, metadata)
    rangePerPoint, increment = np.arange(numPoints), 0.00
    figure = plt.figure()
    #ax = figure.add_axes([1,1,1,1])
    axes = figure.add_axes([0.1,0.2, 0.8, 0.7])
    if hasattr(networkObject, 'testInfo') and (networkObject.testInfo is not None): 
        labels = [index[1] + "_" + str(index[0]) for index in metadata]
        print(predicted, metadata)
        predicted = [predicted[x]*metadata[x][2] for x in range(len(predicted))]
        real = [real[x]*metadata[x][2] for x in range(len(real))]
        axes.set_xticks(range(numPoints))
        axes.set_xticklabels(labels, rotation=90)
    else:
        axes.set_xticks(range(numPoints))
    axes.bar(rangePerPoint, predicted, color = 'g', width=0.25)
    axes.bar(rangePerPoint + 0.25, real, color = 'b', width=0.25)
    plt.show() # Hangs until interrupted
    return predicted, real
        
        
def _modelData_barGraph_bestFit(o, p, r, n):
    va, ml = {}, []
    for i in range(len(p)):
        d = abs(p[i]-r[i])
        if not(d in va): va[d] = i
    print(va)
    s = sorted(va.keys())
    if o.testInfo is not None: ml = [o.testInfo[va[z]] for z in s[0:n]]
    return [p[va[z]] for z in s[0:n]], [r[va[z]] for z in s[0:n]], ml
    
def _modelData_barGraph_random(o, p, r, n):
    pr, rr, ml, rl = [], [], [], []
    p = p.tolist(); r = r.tolist()
    lp = len(p)
    for i in range(n): 
        rl.append(random.randint(0, lp-1)); lp-=1
    for i in rl:
        pr.append(p[i])
        rr.append(r[i])
        if o.testInfo is not None: ml.append(o.testInfo[i])      
    return pr, rr, ml
    
def _modelData_barGraph_arrange(p, r):
    toReturn = [] 
    for i in range(len(p)):
        toReturn.append([r[i], p[i]])
    return toReturn
    
    