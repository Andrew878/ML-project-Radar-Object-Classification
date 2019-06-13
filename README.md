# RadarCATClassification

EXECUTIVE SUMMARY:

A master's machine learning project that classifies objects/materials based on how they interact with radar signals. 

Within the provided python scripts there are:

◦ The training and calibration of four classification models:
        
 ▪ Logistic Regression;
            
 ▪ K-Nearest Neighbours;
            
 ▪ Single layer neural network.
            
 ▪ Gradient Boosted Trees;
            
◦ Aggregation of above models into an ensemble voting classifier that achieves 100% accuracy on unseen test data.
        

Full credit of original idea and data source:

https://sachi.cs.st-andrews.ac.uk/research/interaction/radarcat-exploits-googles-soli-radar-sensor-for-object-and-material-recognition/


OBJECTIVE:

Here, the ultimate desired outcome is not a better understanding of radar’s reflection and refraction properties to further physics, but rather it is the fast, accurate classification of objects. As such, when selecting a model, I did not restrict myself to choosing interpretable models only, and will focussed on accuracy and speed to classify.

An ensemble classifier was considered given: 
    
 • With the small amount of data per category suggests it could be difficult to train a single accurate model. The aggregated classifications of several different models with ‘OK’ accuracy might allow us overcome this by diversification of errors; 
    
 • Evidence in both academia (Kittler, Hatef, Duin, & Matas, 1998)⁠ and actual data-science competitions (Géron, 2017; Murray, 2018)⁠ suggest that ensemble methods can work well for classification problems;
 

GENERAL APPROACH:

 • Step 1: Trained a regularised logistic regression model. Measure performance on training and validation sets to obtain an initial accuracy benchmark.
            a) Adjust regularisation parameter (L2) as appropriate if under or over-fitting occurs.
           
 • Step 2: Trained the following models we get satisfiable accuracy. Use grid-search with cross-validation to tune hyper-parameters.
            a) A gradient boosted tree (‘GBT’) classifier;
            b) K-Nearest Neighbour (‘KNN’) classifier (note: KNN doesn’t require training, only hyper-parameter tuning);
            c) A feed-forward neural network classifier;
			
 • Step 3: Combined all four models into a voting ensemble classifier and test performance on training and validation sets.
      
 • Step 4: Combined training and validation data and retrain model parameters and predict against test set. This step will provide guidance if our ensemble approach is appropriate.
      
 • Step 5: Use all data points (training, validation and test), retrain the model parameters and then use this to test the unlabelled ‘XToClassifyData’. This last step is to ensure we make our final predictions with the most amount of data as possible.


JUSTIFICATION FOR MODEL SELECTION:

Per guidance in Kittler et al, combinations of classifiers are “particularly useful if they are different” (Kittler et al., 1998)⁠. As such, each of the proposed models was chosen because it offers different qualities (strengths/weaknesses) to the ensemble voter:

Logistic regression:

 • Provides linear decision boundaries and so for linearly separable data, this approach should work well.
 • Takes the entire dataset into consideration when creating decision boundaries. 
 • Because “logistic regression wants to create a large margin between boundary and (all of) the data”, it can result in decision boundaries tilting in odd angles to accommodate outliers near the decision boundary (Minka, 2003)⁠ .

Gradient Boosted Trees:

 • The GBT model is itself an ensemble learner that gradually adds and trains multiple decision-tree models where each incremental tree is trained on the residuals of the prior tree. It is somewhat similar to forward stage-wise liner regression (Hastie et al, Chapter 4) in the sense that the models next tree is determined by the residuals of the model’s prior iteration.
 • Tree methods such as GBT are robust to outliers (Hastie, Tibshirani, & Friedman, 2008)⁠.
 • The combination of these trees will produce linear decision boundaries that are orthogonal to the PCA axes. This will be different to the decision boundaries of the other models.
 • Lastly, it is difficult to ignore that GBT approaches tend to feature in many winning data-science entries such as Kaggle and the Netflix Prize (Koren, 2009)⁠. 

K-Nearest Neighbour:

 • KNN is a probabilistic classifier that formulates classifications based on the ‘K’ closest data points and nothing more (in this sense, the ‘K’ majority voting system shares similarities to an ensemble classifier). 
 • Per Hastie et al, KNN’s produces piece-wise linear decision boundaries which can be used to represent “irregular” and non-linear class boundaries (Hastie et al., 2008)⁠. 
 • Hastie et al also details how KNN are successful in many real world applications including electrocardiography signals. From my naive perspective, pictures of electrocardiography waves don’t look dissimilar to the waveforms in the RadarCat article. This leads me to believe KNN could be worth attempting.
 • Russell and Norvig also note that “in low-dimensional spaces with plenty of data, nearest neighbours works very well”(Russell & Norvig, 2016)⁠. Due to the curse of dimensionality, KNN might have been infeasible in the original, larger feature set, but with the smaller PCA features, the data-to-feature ratio is much higher, and so there is a possibility it will perform well. 
 • Unfortunately, as the size of the data set gets very large, KNN will take longer and longer to classify (as it calculates the distance from every data point) but given the small data set, it is sufficient.

Neural network:

 • NN’s can represent complex non-linear decision boundaries between classes and work well in instances where there is high-signal-to-noise ratios (Hastie et al., 2008)⁠.
 • NN’s are well suited to classification tasks, particularly when “modelling real world complex relationships” (G. P. Zhang, 2000)⁠ (which this problem is considered as).
    
    
