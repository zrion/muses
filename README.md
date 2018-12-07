# MusEs: *MU*sic genre cla*S*sification with *E*n*S*emble learning

We focus on music genre classification, which is important for music information retrieval and is an interesting problem. 

- Python version: Python 3.4+ (*This is to take advantages of parallel processing with GridSearchCV, which has some problems with multiprocessing that is hard to solve in lower Python version.*)
- Libraries used: 
	- For data manipulation and visualization: Numpy, scipy, pandas, matplotlib, imbalanced-learn
	- For learning and prediction: Scikit-learn, xgboost
- Dataset: Free Music Archive (FMA) https://github.com/mdeff/fma

Models that I used:
- Baseline models: Logistic regression, Decision Tree, SVM kernel RBF/Linear, Neural Network
- **Ensemble models**: XGBoost (Extreme Gradient Boosting) and ExtraTrees (Extremely Randomized Trees)

Further techniques that I used:
- Principal Component Analysis (PCA) for dimensionality reduction.
- Synthetic Minority Over-sampling (SMOTE) for balancing the dataset.

Possible directions for future work:
- Feature engineering for selecting best subset of features
- Relabeling data to alleviate the effect of imbalanced dataset
- Cost-Sensitive Learning is another approach for imbalanced dataset (Problem: Cannot wrap around for all models! (http://storm.cis.fordham.edu/~gweiss/papers/dmin07-weiss.pdf)
- Classification with automatic feature extraction using Convolutional Neural Network with strong classification models: XGBoost, Deep Neural Network

**Notes** if you want to use the code:
- Make sure that you have all the required library installed. The easiest way is by using *pip install*
- Make sure that you understand how to tune hyperparameters using GridSearchCV. You should see in the code that I've done the tuning extensively to many models and that could cause confusion if you are not familiar with using *scikit-learn*
- Please check the path to recording files.



