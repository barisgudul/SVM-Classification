# SVM Classification

## Project Overview
This project applies **Support Vector Machine (SVM)** for binary classification using the **sklearn.svm.SVC** module with a linear kernel. The goal is to classify data based on two features: `Age` and `Estimated Salary`. The model performance is evaluated using metrics like a **confusion matrix** and **accuracy score**.

## What is a Support Vector Machine?
**Support Vector Machine (SVM)** is a supervised machine learning algorithm used primarily for classification tasks. It works by finding the optimal hyperplane that best separates data points belonging to different classes in the feature space.

### Key Concepts:
1. **Hyperplane**: A decision boundary that separates different classes in the dataset.
2. **Support Vectors**: Data points closest to the hyperplane that influence its position.
3. **Margin**: The distance between the hyperplane and the nearest support vectors. SVM aims to maximize this margin to ensure better classification.

SVM can work with:
- **Linear Data**: Using a linear kernel (as in this project)
- **Non-Linear Data**: Using kernels like polynomial, radial basis function (RBF), etc.

## Dataset
The project uses the dataset: **Social_Network_Ads.csv**
- **Target Variable**: Binary class (0 or 1)
- **Features**: 
  - Age
  - Estimated Salary

## Code Workflow
1. **Data Preprocessing**
   - Import libraries
   - Load the dataset
   - Split data into training and test sets

2. **Model Training**
   - Support Vector Classification (SVC) with a **linear kernel**
   - Random State: 42 for reproducibility

3. **Model Evaluation**
   - Predictions on test data
   - Confusion matrix
   - Accuracy score

4. **Visualization**
   - Visualize the **decision boundary** for training and test sets

## Key Code Snippet
```python
from sklearn.svm import SVC
classifier = SVC(kernel="linear", random_state=42)
classifier.fit(X_train, y_train)

# Predicting Test Set Results
y_pred = classifier.predict(X_test)

# Evaluate the Model
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
```

### Confusion Matrix Output:
```
[[50  2]
 [ 9 19]]
```

### Accuracy:
```
0.8625
```

## Visualizations
1. **Train Set**:
   ![Train Set Visualization](Support%20Vector%20Machine%20Classification/train.png)

2. **Test Set**:
   ![Test Set Visualization](Support%20Vector%20Machine%20Classification/test.png)

## Results
The SVM classifier achieved an accuracy of **86.25%** on the test set, demonstrating good performance for a linear kernel in this binary classification task.

## Files Included
- **SVM_classification.ipynb**: Main code implementation
- **Social_Network_Ads.csv**: Dataset
- **train.png**: Training set visualization
- **test.png**: Test set visualization

## Author
- **Mehmet Barış Güdül**
