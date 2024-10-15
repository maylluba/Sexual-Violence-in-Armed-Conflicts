# A data-driven analysis of Conflict Related Sexual Violence in Armed Conflicts(CRSV) 📊📈 

For our analysis, we utilized the **Sexual Violence in Armed Conflict (SVAC) datase**t, from the
The Peace Research Institute Oslo and the John F. Kennedy School of Government at
Harvard University. The Sexual Violence in Armed Conflict (SVAC) dataset measures reports
of the conflict-related sexual violence committed by armed actors during the years
1989-2021.

We have a total of 19 columns and 11911 entries (rows). Memory usage: 1.7 MB.

Our aim was to use this dataset to build an algorithm that could make predictions about the
likelihood of sexual violence in conflicts, leveraging the detailed data on actors, their behavior, and the context in which they operate.

The data is **highly imbalanced**, with far more observations for no sexual violence (class 0)
than for occurrence of sexual violence (class 1). This imbalance can severely affect the
performance of machine learning models, which tend to favor the majority class.

We leveraged the feature importance mechanisms provided by
both Random Forest and XGBoost (eXtreme Gradient Boosting) to gain insights into which
variables had the greatest impact on our predictions. These algorithms were chosen because they are known to handle imbalanced classification tasks well, especially when
combined with techniques like oversampling and class weighting.

We first trained a **Random Forest** model to predict the occurrence of sexual violence. It is a
bagging algorithm that reduces overfitting by averaging multiple decision trees. It is robust to
imbalanced datasets because it can handle both categorical and continuous variables and
has mechanisms to deal with imbalance through techniques like class weighting and
oversampling (which we implemented with RandomOverSampler) provides a built-in method
to measure feature importance based on the Gini impurity or information gain.

We then trained a **XGBoost** model. This boosting algorithm builds models sequentially and focuses on improving errors from
previous models. It’s particularly good for our imbalanced datasets because it allows you to
weight classes or use custom loss functions to emphasize the minority class (sexual violence
cases). XGBoost is often more efficient than Random Forest when dealing with complex,
imbalanced data. 

While this project delivered significant insights, we encountered several challenges that
required careful adjustments to improve model performance: class imbalance (that we addressed by using RandomOverSampler), limitations by default
hyperparameters (optimized with GridSearchCV), and overfitting (avoided by the use of Stratified Cross-Validation).

This project successfully applied machine learning models—Random Forest and
XGBoost—to predict sexual violence in conflict zones. **The results validate the ability of machine learning models to assist in predicting
and understanding conflict-related sexual violence.**

*Future Research Directions:* Despite the successes of this project, there are areas for
future improvement. More advanced sampling techniques, such as SMOTE
(Synthetic Minority Over-sampling Technique) or ADASYN, could further enhance
model performance by addressing class imbalance more effectively. Additionally,
exploring temporal features—such as conflict timelines or evolving actor behaviors
over time—could add depth to the model’s predictions. Future models could also
benefit from integrating additional data sources, such as social media analysis or
geospatial data, to create more comprehensive predictive tools.


