import csv
import sys
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression, Lars, OrthogonalMatchingPursuit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statistics, random, datetime
import pickle

class InfluenzaNetwork:

    def __init__(self, fields, testPercentage): 
        self.data = self.getDataFromFile("influenza_data_by_year_by_county.csv")
        self.fields = fields
        self.model = None
        self.trainingInput = None 
        self.trainingOutput = None
        self.trainingInfo = None 
        self.testInput = None
        self.testOutput = None
        self.testInfo = None
        self.testPercentage = testPercentage
    
    def getDataFromFile(self, fileName):
        '''
        setData(): Sets field "self.data" with dictionary parsed from CSV File; 
        dictionary in form {year : {county: {...} } }
        fileName: Relative Path to "influenza_data_by_year_by_county.csv"        
        '''
        yearSet = {}
        with open(fileName, 'r') as rp: 
            csvreader = csv.reader(rp)
            fieldDictionary = {} 
            fields = next(csvreader) 
            for i in range(len(fields)):
                if not (fields[i] in fieldDictionary): 
                    fieldDictionary[fields[i]] = i          
            for row in csvreader: 
                if len(row) == 0: continue
                year = row[fieldDictionary["Year"]]
                county = row[fieldDictionary["County"]]
                if not (year in yearSet): 
                    yearSet[year] = {} 
                if not (county in yearSet[year]):
                    yearSet[year][county] = {}
                for parsedField in list(fieldDictionary.keys()): 
                    if parsedField in ["Year", "County"]: continue
                    yearSet[year][county][parsedField] = float(row[fieldDictionary[parsedField]])
        return yearSet
        
    def getIOFromData(self, testPercentage):
        '''
        Assumes existence of self.data formatted as "{year : {county: {...} } }"
        '''
        inputList, outputList, trainingInput, trainingOutput = [], [], [], []
        IOMetadata, trainingMetadata = [], []
        for yearKey in self.data: 
            for countyKey in self.data[yearKey]:
                outputList.append(self.data[yearKey][countyKey]["Percent"])
                singleInput = []
                for field in self.fields:
                    singleInput.append(self.data[yearKey][countyKey][field])
                inputList.append(singleInput)
                IOMetadata.append((yearKey, countyKey, self.data[yearKey][countyKey]["Population"]))
                
        # Split into test and training sets based on "testPercentage"
        if testPercentage > 1 or testPercentage < 0: 
            testPercentage = 0.25
        trainingSplit = int(float(len(inputList)) * (1-testPercentage))
        while len(trainingInput) < trainingSplit: 
            randomPos = random.randint(0, len(inputList)-1)
            trainingInput.append(inputList[randomPos])
            trainingOutput.append(outputList[randomPos])
            inputList.pop(randomPos)
            outputList.pop(randomPos)
            trainingMetadata.append(IOMetadata[randomPos])
            IOMetadata.pop(randomPos)
        self.trainingInput = np.array(trainingInput)
        self.trainingOutput = np.array(trainingOutput)
        self.testInput = np.array(inputList)
        self.testOutput = np.array(outputList)
        self.trainingInfo = trainingMetadata
        self.testInfo = IOMetadata
    
    def trainLinearElasticNet(self, alpha, l1): 
        self.getIOFromData(self.testPercentage)
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1)
        self.model.fit(self.trainingInput, self.trainingOutput)
        
    def trainLinearRegression(self):
        self.getIOFromData(self.testPercentage)
        self.model = LinearRegression()
        self.model.fit(self.trainingInput, self.trainingOutput)
    
    def trainSVRLinear(self, cValue, gammaValue):
        self.getIOFromData(self.testPercentage)
        self.model = SVR(kernel='linear', C=cValue, gamma=gammaValue)
        self.model.fit(self.trainingInput, self.trainingOutput)    
        
    def trainSVRRadial(self, cValue, gammaValue, epsilonValue):
        self.getIOFromData(self.testPercentage)
        self.model = SVR(kernel='rbf', C=cValue, gamma=gammaValue, epsilon=epsilonValue)
        self.model.fit(self.trainingInput, self.trainingOutput)
        
    def trainLinearRidge(self, alpha, fit_intercept):
        self.getIOFromData(self.testPercentage)
        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        self.model.fit(self.trainingInput, self.trainingOutput)
        
    def trainLars(self):
        self.getIOFromData(self.testPercentage)
        self.model = Lars()
        self.model.fit(self.trainingInput, self.trainingOutput)
        
    def trainLinearOrthogonalMatchingPursuit(self):
        self.getIOFromData(self.testPercentage)
        self.model = OrthogonalMatchingPursuit()
        self.model.fit(self.trainingInput, self.trainingOutput)
        
    def trainMLPRegressor(self, layerSizes, tolerance, max_iterations, activationFunction='relu'):
        self.getIOFromData(self.testPercentage)
        self.model = make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=layerSizes,tol=tolerance, max_iter=max_iterations, random_state=0, activation=activationFunction))
        self.model.fit(self.trainingInput, self.trainingOutput)
        
    def testModel_statistics(self):
        #print(self.trainingInput, self.testInput, self.trainingOutput, self.testOutput)
        results = self.model.predict(self.testInput)
        #print(results)
        percentErrors = []
        for resultIndex in range(len(results)): 
            # Calculate percent error
            PE = abs((self.testOutput[resultIndex] - results[resultIndex])/self.testOutput[resultIndex]) * 100
            percentErrors.append(PE)
        return statistics.mean(percentErrors)
        
    def testModel_output(self):
        results = self.model.predict(self.testInput)
        return results
        
    def testModel_custom(self, customInputList):
        results = self.model.predict(customInputList)
        return results
        
    # Static Methods to load/dump models from/into files
    def importModel(filename):
        with open(filename, 'rb') as fp:
            model = pickle.load(fp)
            toReturn = InfluenzaNetwork(model.fields, model.testPercentage)
            toReturn.model = model.model
            toReturn.trainingInput = model.trainingInput 
            toReturn.trainingOutput = model.trainingOutput
            toReturn.testInput = model.testInput
            toReturn.testOutput = model.testOutput
            if hasattr(model, 'trainingInfo'): toReturn.trainingInfo = model.trainingInfo
            if hasattr(model, 'testInfo'): toReturn.testInfo = model.testInfo
            return toReturn

    def exportModel(influenzaNetworkInstance, filename=None):
        if (filename is None) or (len(filename) == 0) or (".pickle" not in filename):
            filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_model.pickle"
        with open(filename, 'wb') as wp: 
            pickle.dump(influenzaNetworkInstance, wp, protocol=pickle.HIGHEST_PROTOCOL)
            
   