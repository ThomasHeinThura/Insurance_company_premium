# Insurance_company_premium

This projects mainly focus on imbalanced classification datasets which is fraud detection. And, testing all sorts of machine learning models on the datasets.
Datasets are obtained from Kaggle.
1. [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?datasetId=310&sortBy=voteCount&sort=votes)
2. [Fraud e-commerce ](https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce )
3. [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
4. [IEEE-CIS Fraud Detection]( https://www.kaggle.com/competitions/ieee-fraud-detection/data )  
* For now, I just write a project on Credit Card Fraud Detection, because the dataset contains numbers and there is no need to feature selection and data encoding.     
* In this project, there are a total of 20 Machine Learning Models, 1 Stacking model, 2 ANN(TensorFlow DNN) models and 1 CNN models.
* Also, I add pyspark model for who wanted to test and try out. 
* There are example notebooks from Kaggle. If you want to know further details and try something, you can read them.   

### Challenges and Notes
* In this project, you have to know Datasets are imbalanced in both classifications. Normally, the classification classes need to have a nearly 50-50 ratio. If classification classes are imbalanced, the main problem is you will get high accuracy score on majority class prediction but poor performance on the detection of minority class(in this case, is_fraud). So, you need to understand undersampling, oversampling and mix sampling.
* Another challenge is hyperparameter tuning and long training time. For testing purposes, I use mainly the undersampling dataset that is modified by Near Miss. 

### Finding 
* I found oversampling data using SMOTE is quite good. Stacking and ANN(TensorFlow DNN) models show nearly 99% accurate results and No False Negatives on the testing dataset. 
* Undersampling data tends to overfit, and sometimes it is underfitting due to a low dataset.


## Result Table
Title             |  With SMOTE         |  Without SMOTE
:-------------------------:|:-------------------------:|:-------------------------:
All model f1 score  | ![](result_png/all_model_f1_score_SMOTE.png)| ![](result_png/all_model_f1_socre.png)
All model f1 score plot|![](/result_png/all_model_f1_score_plot_SMOTE.png)|:![](result/../result_png/all_model_f1_score_plot.png):
logistic | ![](result_png/logistic_matrix_SMOTE.png) | ![](result_png/logistic_matrix.png) 
Bagging | ![](result_png/Bagging_martix_SMOTE.png) | ![](result_png/Bagging_martix.png)
CatBoost | ![](result_png/Cat_matrix_smote.png) | ![](result_png/Cat_matrix.png)
Extra Tree | ![](result_png/Extra_tree_matrix_SMOTE.png) | ![](result_png/Extra_tree_matrix.png)
Random forest Tree | ![](result_png/Random_forest_matrix_SMOTE.png)| ![](result_png/Random_forest_matrix.png)
Stacking model | ![](result_png/stacking_matrix_SMOTE.png) | ![](result_png/stacking_matrix.png)
ANN Matrix | ![](result_png/ANN_matrix_SMOTE.png) | ![](result_png/ANN_matrix.png)
ANN Accuracy | ![](result_png/ANN_accurary_SMTOE.png) | ![](result_png/ANN_accuray.png)
ANN Loss | ![](result_png/ANN_loss_SMOTE.png) | ![](result_png/ANN_loss.png)


## Further Study 
* try to test other datasets (2, 3, 4).
* Run tests on oversampling datasets and models.
* Add LSTM models because there is a time frame.

