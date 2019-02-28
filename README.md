# MusEs: *MU*sic genre cla*S*sification with *E*n*S*emble learning

<span style="font-family:Papyrus; font-size:4em;">Hai N. Nguyen, PhD student</span>\
<span style="font-family:Papyrus; font-size:4em;">Khoury College of Computer Sciences, Northeastern University</span>\
<span style="font-family:Papyrus; font-size:4em;">In the scope of CS6140 Machine Learning final project</span>

We focus on music genre classification, which is important for music information retrieval and is an interesting problem. 

- Python version: Python 3.4+ (*This is to take advantages of parallel processing with GridSearchCV, which has some problems with multiprocessing that is hard to solve in lower Python version.*)
- Libraries used: 
	- For data manipulation and visualization: Numpy, scipy, pandas, matplotlib, imbalanced-learn
	- For learning and prediction: Scikit-learn, xgboost
- Dataset: Free Music Archive (FMA) https://github.com/mdeff/fma

Models that I used:
- Baseline models: Logistic regression, Decision Tree, SVM kernel RBF/Linear, Neural Network
- **Ensemble models**: XGBoost (Extreme Gradient Boosting) and ExtraTrees (Extremely Randomized Trees)
Hyper-parameters have been tuned with 5-fold cross validation.

Further techniques that I used:
- Principal Component Analysis (PCA) for dimensionality reduction.
- Synthetic Minority Over-sampling (SMOTE) for balancing the dataset.

Possible directions for future work:
- Feature engineering for selecting best subset of features
- Relabeling data to alleviate the effect of imbalanced dataset
- Cost-Sensitive Learning is another approach for imbalanced dataset (Problem: Cannot wrap around for all models! (http://storm.cis.fordham.edu/~gweiss/papers/dmin07-weiss.pdf)
- Classification with automatic feature extraction using Convolutional Neural Network with strong classification models: XGBoost, Deep Neural Network

**Notes**
- Make sure that you have all the required library installed. The easiest way is by using `pip install`
- Make sure that you are comfortable with tuning hyperparameters using `GridSearchCV`.
- Please check the path to recording files.



