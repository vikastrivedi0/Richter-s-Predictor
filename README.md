# Richter-s-Predictor
A machine learning model to predict the Danger level for the earthquake using multiple attributes of both the land and the building.

This model uses data provided by Driven Data (https://www.drivendata.org/competitions/57/nepal-earthquake/) to analyze the propertiesof the land along with features of the building in Nepal classifying the damage in 3 different classes i.e 1,2 & 3 namely.

This is a Random Forrest model with no default hyper-parameters which uses 100 estimators and the average of all the 100 estimators is used for one observation. It is a type of Ensemble Learning method. The training of the model takes some time as the quantity of data is vast, adding the computation time taken for the ensemble learning method.

The Data Pre-processing steps include conversion of categorical dependent varibles to numerical variable using the One-Hot Encoding method. Also to avoid Dummy Variable Trap, one dummy variable from each set is removed. Standard scaling is also applied as dimensionality reduction methods need to be applied.

As the data contains 61 features after dummy variable generation, these are to be reduced using Kernel Principle Component Analysis(KPCA) which identifies and selects 20(according to code) best features for the data which provide same variance as the original data and minimal information loss occures.

This data is then fitted to the Random Forrest model with 100 individual trees. The Prediction is generated and stored in a "Submission.csv" file.
