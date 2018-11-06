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

To be updated...
