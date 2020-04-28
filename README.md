# Influenza Network

Influenza Network is a Python-based Machine Learning library createrd to predict effects of influenza outbreaks in the United States on a county-level. The software employs several techniques including Linear Regression, Multi-Layer Perceptron (Neural Network), and Support Vector Regression (SVR) to correlate statistics on social vulnerability with infection quantities -- with an overall goal of predicting future disease figures and saving lives.

The library uses [Scikit-Learn], an educational Machine Learning library provided under New BSD License, as its ML backbone. Inputs undergo several processes before training/testing models including (1) Preprocessing and Normalization, (2) Model and Algorithm Selection, and (3) formation of Testing/Training Datasets. 

Data is retrieved in CSV format from the following sources...
  - [svi20xx.csv] -- Social Vulnerability Index Annual Data (Center for Disease Control) 
  - [influenza.csv] -- Influenza Laboratory (State of New York) 
  - [census.csv] -- US Census Data 

#### Dependencies

  - Python (3.8.0 or later)
  - matplotlib (3.2.1) and dependencies
  - numpy (1.18.3) and dependencies
  - scikit-learn (0.22.2) and dependencies
  - sklearn (0.0) and dependencies

   [Scikit-Learn]: <https://scikit-learn.org/stable/>
   [census.csv]: <https://github.com/chrisj770/InfluenzaNetwork/blob/master/datafiles/census.csv>
   [svi20xx.csv]: <https://github.com/chrisj770/InfluenzaNetwork/tree/master/datafiles>
   [influenza.csv]: <https://github.com/chrisj770/InfluenzaNetwork/blob/master/datafiles/influenza.csv> 
