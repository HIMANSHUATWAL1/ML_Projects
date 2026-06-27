
# <----------------------------------------------------------- Model for Diabetes Detection ----------------------------------------------------->

# Why we are using svm -->

Binary Classification Problem:
Your target variable Outcome has two classes (diabetic vs. non-diabetic). SVM is well-suited for binary classification tasks.

Decision Boundary:
SVM tries to find the optimal hyperplane that separates the two classes with the maximum margin. This makes it robust in handling overlapping data points.

Linear Kernel Choice:
You used kernel='linear'. This means the model assumes the data can be separated (at least approximately) by a straight line (or hyperplane in higher dimensions).

Linear kernels are faster and work well when features are already informative.

For more complex patterns, you could experiment with rbf or poly kernels.

# Why Standard scaler -->

SVM is sensitive to the scale of features because it relies on distances (dot products, norms) to construct the decision boundary.

Example:-
Glucose values might range in the hundreds.

BMI values are typically between 15–50.

Without scaling, glucose would dominate the distance calculations, skewing the model.

# outcome -->
SVM provides a strong classifier for your binary outcome.

StandardScaler ensures that every feature contributes fairly to the decision boundary.
