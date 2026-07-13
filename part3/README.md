### part3 Readme
### task1
Report training accuracy and test accuracy:
Training Set Accuracy: 1.0000 (100.0%)
Testing Set Accuracy: 0.6400 (64.0%)

a perfect 100% accuracy score on our training data proves that the unconstrained tree did not actually learn
it completely panics when faced with the unseen testing set
In data science, single decision trees are notoriously categorized as high-variance models due to their core architectural logic. They are what we call greedy learners
Once it makes that split and divides the data into branches, it can never go back in time to change, adjust, or revisit that earlier decision. It locks itself into that path permanently.

### task2
Report training and test accuracy again:
Training Set Accuracy: 0.7163 (71.63%)
Testing Set Accuracy: 0.6700 (67.00%)
explain the role of max_depth:
This parameter puts a strict structural ceiling on how deep the flowchart can grow from top to bottom.
min_samples_split : a new branch/rule unless you are looking at a group of at least 20 customer records who share that pattern." If a weird quirk only affects 2 or 3 rows, the tree is forced to ignore it, preventing it from turning tiny, random coincidences into official business rules.
Compare the controlled tree's train/test gap to the unconstrained tree:
The Unconstrained Tree: Had a massive, highly unstable its training score (100%) and test score (64%). It looked perfect in practice but collapsed on the actual exam.
The Controlled Tree : Shrunk that performance gap down to a tight, incredibly safe 

### task3:
max depth =5
Gini Impurity Test Accuracy: 0.6750 (67.50%)
Entropy Test Accuracy: 0.6800 (68.00%)

Gini Impurity Formula: Gini =  1 - Σ pi²
he probability or proportion of customers belonging to a specific class i  inside that group. Gini measures the likelihood that a randomly chosen customer would be mislabeled if you guessed their status completely at random. It uses basic squaring, making it computationally very fast and light on your computer's processor

Entropy Formula: -Σ pi log2(pi)
he proportion of customers in class i, calculated using base-2 logarithms. Entropy comes from information theory and measures the absolute level of chaos, surprise, or unpredictability inside a group. If a customer bucket is a perfect 50/50 chaotic mix of staying and leaving, Entropy hits its absolute maximum score of 1.0

Gini = 0 (or Entropy = 0), it means the bucket of data has achieved Absolute Purity
For example, if a branch filters out a group of 50 customers, and all 50 of them are Loyal (0) while 0 of them cancelled (1), there is zero mixture, zero mess, and zero doubt remaining in that room.

### task4:
Report training accuracy, test accuracy, and ROC-AUC:
Training Set Accuracy: 0.9137 (91.37%)
Testing Set Accuracy: 0.6950 (69.50%)
Testing ROC-AUC Score: 0.7414 (74.14%)

A Random Forest relies on an engineering technique called Bagging (Bootstrap Aggregating) to build its power.
Instead of letting all 100 trees study the exact same dataset, the ensemble uses bootstrap sampling, meaning each individual tree is given a random sample drawn with replacement from our training data. This means some customer rows are repeated, and some are left out completely for each tree.
By forcing 100 deep trees to study different combinations of customers and features, you create an incredibly diverse committee. When a new customer profile walks in, all 100 trees vote, and the ensemble averages their predictions. This averaging process dramatically reduces variance compared to a single deep decision tree; if five trees get tricked by random noise or data anomalies, the other 95 trees outvote them, stabilizing the system and pulling your final test scores higher

model tracked down our most critical columns and generated a clear business leaderboard (TotalCharges, tenure, Contract_two year, Contract_one year, PaperlessBilling_yes).
How Random Forest computes importance: Every time an individual tree uses a feature column (like TotalCharges) to split a group of customers and successfully lowers the group's messiness, scikit-learn records how much the Gini impurity dropped. The model tallies up this reduction across every single split point that used that feature, across all 100 trees, and averages it. Features that consistently clean up messy data pools rise straight to the top of the leaderboard

 Linear Regression: A Linear Regression coefficient is a rigid, global multiplier. It says: For every 1 unit this column goes up, churn risk goes up or down by exactly X amount in a straight line. It assumes a fixed, steady relationship. Random Forest importance doesn't care about straight lines. It measures pure predictive utility. It tells you how useful a column is at segmenting complex, curved, non-linear customer behaviours—even if a column's risk goes up in some situations and down in others

### 4 a): Gradient Boosting
GradientBoostingClassifier. Instead of building 100 trees all at once in parallel like the Random Forest, this algorithm builds a highly disciplined sequential team. It trains trees one after the other in an assembly line, where each new tree is custom-engineered to correct the exact classification mistakes made by the tree before it.
Training Set Accuracy: 0.8450 (84.50%)
Testing Set Accuracy: 0.6850 (68.50%)
Testing ROC-AUC Score: 0.7464 (74.64%)

### 4 b):  Feature ablation study
To test the streamlined efficiency of our system, we conducted a structural stress test called a Feature Ablation Study. We used our logical negation operator (~) to completely chop away the 5 weakest, lowest-importance columns discovered by our Random Forest model and evaluated how the network behaved. The head-to-head performance scorecard on our unseen testing data generated the following results
(a) Full Feature Set Test ROC-AUC (All Features): 0.7414 (74.14%)
(b) Reduced Feature Set Test ROC-AUC (5 Removed): 0.7271 (72.71%)

the script flagged several duplicate categorical columns contaminated with trailing text spaces (such as 'gender_  Male  ' and 'PaymentMethod_  Electronic check  ') as low-importance clutter.
model's diagnostic capability experienced a very minor degradation—with the ROC-AUC score dropping by a tiny 1.43% (from 0.7414 down to 0.7271).

### task5: Cross-validated comparison
remove all chance of luck and scientifically identify our most stable model, we subjected all four candidate algorithms to a rigorous 5-Fold Stratified Cross-Validation tournament. The evaluation referee tracked the Area Under the ROC Curve (ROC-AUC) across five separate rounds, generating the following final scoreboard metrics:

Logistic Regression: Mean AUC = 0.7020 | Std = 0.0203
Random Forest Classifier: Mean AUC = 0.6978 | Std = 0.0354
Gradient Boosting Classifier: Mean AUC = 0.6782 | Std = 0.0546
Controlled Decision Tree: Mean AUC = 0.6767 | Std = 0.0271

Cross-Validation completely destroys this randomness. By cutting the dataset into 5 equal, stratified blocks, we force every model to repeat its exam five separate times. In each round, the data blocks shift seamlessly

### task6: Hyperparameter tuning with GridSearchCV
deployed a robotic hyperparameter auto-tuner loop (GridSearchCV) to find the absolute best structural recipe for our forest. The final tuning metrics generated the following output
Best Hyperparameters Found: {'max_depth': 5, 'min_samples_leaf': 5, 'n_estimators': 50}
Best Cross-Validated ROC-AUC Score: 0.7056

### task7: Manual learning curve
evaluate the return on investment of our data size, we ran a manual Learning Curve Analysis on our champion tuned Random Forest pipeline
(i)
20% Fraction: Training AUC = 0.9064 | Testing AUC = 0.6827
40% Fraction: Training AUC = 0.8797 | Testing AUC = 0.7359
60% Fraction: Training AUC = 0.8448 | Testing AUC = 0.7412
80% Fraction: Training AUC = 0.8303 | Testing AUC = 0.7297
100% Fraction: Training AUC = 0.8262 | Testing AUC = 0.7585

When you give a high-variance model a tiny, 20% snapshot of customer data, the dataset is so small that the trees can easily cheat. They use their structural power to hyper-memorize individual rows and random coincidences word-for-word, resulting in a near-perfect practice test score.
However, as you systematically add more customer records toward 100%, you flood the training room with a massive, diverse crowd of unique customer profiles. Rote memorization suddenly becomes impossible because the model cannot over-complicate its logic to fit every single person
(ii)
the second column of our results table, we see that our Test AUC steadily increases as more training data is added, climbing from a weak 0.6827 at the 20% mark all the way to its peak of 0.7585 at full capacity.
the pipeline was trained on only 20% of the customer database, it was too uneducated and ignorant to pass the final exam, leading to low predictive accuracy. However, as we systematically expanded the training data volume to 40%, 60%, and 100%, the model's predictive radar grew significantly sharper. It successfully transformed those extra data points into real, actionable insights.
(iii)
You would see the Test AUC flatten out and stop changing early on (for example, hitting a ceiling at 60% or 80% data and staying identical all the way to 100%). That would tell you that adding more data is a complete waste of corporate money because the model has already maxed out its intelligence.
it experienced its sharpest upward surge—jumping from 0.7297 at the 80% milestone straight to its absolute project peak of 0.7585 (75.85%) at the 100% full dataset boundary.

### remaing taskes:
we stacked our historical baseline from Part 2 head-to-head against all of our advanced individual and ensemble models from Part 3. The consolidated evaluation scoreboard across all validation testing methods generated the following metrics


Model Description  5-Fold CV Mean AUC  5-Fold CV Std AUC  Test-SetAUC    

Logistic              0.7020             0.0203            0.7165
Regression(part2)

Controlled            0.6767             0.0271            0.6700
Decision Tree
(Task2)

Random Forest         0.6978             0.0354            0.7414
Classifier
(Task 4)

Gradient 
Boosting Classifier   0.6782             0.0546            0.7464
(task 4a)

Tuned Random          0.7056             0.0284            0.7585
Forest Pipeline
(task 6 winner)


Recommendation & Justification
I highly recommend deploying the Tuned Random Forest Pipeline (from Task 6) into production for the client.

While the Logistic Regression model from Part 2 put up an incredibly stable fight, our automated hyperparameter tuning loop successfully optimized the Random Forest's structure—allowing the ensemble to achieve our project's peak performance with a Test-Set AUC of 0.7585 and a Cross-Validation Mean AUC of 0.7056.



