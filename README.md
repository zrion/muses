# MusEs: *Mu*sic genre cla*s*sification with *E*n*s*emble learning

We focus on music genre classification, which is a crucial problem for music recommender systems.

- Python version: Python 2.7
- Libraries used: 
	- For data manipulation and visualization: Numpy, scipy, pandas, matplotlib
	- For learning and prediction: Scikit-learn, xgboost


We will evaluate through various learning models for the task. Planned models to use:
- Baseline models: Softmax regression, Decision tree
- Advanced models: SVM kernel RBF, Neural Network
- **Ensemble models**: XGBoost (Extreme Gradient Boosting) and ExtraTrees (Extremely Randomized Trees)

Our techniques we intent to use:
- Unbalanced dataset: Resampling (Problem: Overfitting!!), Cost-Sensitive Learning (Problem: Cannot wrap around for all models! (http://storm.cis.fordham.edu/~gweiss/papers/dmin07-weiss.pdf)
- Overfitting: Regularization (L2, LASSO), Dropout (for neural network)
- Feature selection: PCA, any other (e.g. pre-analyze data to see if features are discriminate)?

Our strategy:
- Cleaning up dataset: Basically dataset is clean already, no missing values and all values are numerical so appropriate for any models. So what we need to do is FEATURE SELECTIONS. One obvious approach is dimensionality reduction. Another "exhaustive" approach is manual selection. We investigate data based on the kind of features, e.g. 'mfcc', 'tonnetz', etc. Due to time limit, I would not suggest to do this.
- Parameter tuning: One way to apply CV is to train n_estimators and max_depth. Other parameters like gamma, learning_rate, alpha, beta can be set after that.  
