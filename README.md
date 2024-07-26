# Machine Learning Roadmap

**This is a compilation of the resources I gathered while learning the skills required to build machine learning projects**

**I originally made this to serve as a roadmap for me to follow; but I then thought, why not make it publicly viewable? It is still a work in progress since I am continuously learning new skills, and it is by no means an exhaustive list. Anyways, hope you have fun with this repo!**

**PS: This is targeted at individuals that already understand basic programming concepts and things like version control.**

**🔗 Connect with me:** [![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/Mohamed-Al-Mamari) 

## Topics of Study

- [Statistics](#statistics)
- [Linear Algebra](#linear-algebra)
- [Calculus](#calculus)
- [Programming](#programming)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Pt.1: Fundamentals](#machine-learning-pt1-fundamentals)
- [Machine Learning Pt.2: Algorithms](#machine-learning-pt2-algorithms)
- [Model Evaluation and Validation](#model-evaluation-and-validation)
- [Deep Learning](#deep-learning)
- [Natural Language Processing (NLP)](#natural-language-processing-nlp)
- [Computer Vision (CV)](#computer-vision-cv)
- [Data Niches](#data-niches)
- [Deployment](#deployment)

---

## Statistics

Statistics can be thought of as the "foundational toolkit" for understanding and interpreting data. It's crucial for tasks like data analysis, building probabilistic models, and evaluating model performance and so on. To build a general intuition for the statistics required for data science, focus on the following areas:


![Illustration-of-a-bivariate-Gaussian-distribution-The-marginal-and-joint-probability_W640](https://github.com/user-attachments/assets/32db0836-d58e-40e5-978a-f8c55539e52a)


- **Descriptive Statistics**: Measures of Central Tendency (mean, median, mode), Measures of Dispersion (range, variance, standard deviation).
- **Inferential Statistics**: Hypothesis testing, Confidence intervals, P-values, t-tests, chi-square tests.
- **Probability Theory**: Basic probability, Conditional probability, Bayes' theorem.
- **Distributions**: Normal distribution, Binomial distribution, Poisson distribution.
- **Bayesian Statistics**: Bayesian inference, Prior and posterior probabilities.


📚 **References:**

- [Stat Quest with Josh Starmer](https://www.youtube.com/user/joshstarmer)
- [Datatab's guide to statistics for data science](https://www.youtube.com/watch?v=Ym1iH8-GQOE&t=456s)
- [Zedstatistics](https://www.youtube.com/@zedstatistics/)
- [Khan Academy](https://www.khanacademy.org/math/statistics-probability/)
- [Harvard statistics 110](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo)

<details>
<summary>Code Example: Calculating Mean, Median, and Mode in Python</summary>
    
```
import numpy as np
from scipy import stats

data = [1, 2, 2, 3, 4, 7, 9]

mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data)
```

</details>

---

## Linear algebra

Linear algebra is fundamental for understanding how data is represented and manipulated in machine learning algorithms. It forms the backbone of many ML techniques, especially in deep learning where operations on large matrices are common.

![Linear_subspaces_with_shading](https://github.com/user-attachments/assets/6c1ca9a4-c4dd-4cb5-ad60-af7ea59815ad)


- **Vectors & Vector Spaces**: Addition, subtraction, scalar multiplication, dot products, linear transformations.
- **Matrices & Matrix Operation**s: Basic concepts, addition, subtraction, multiplication, transposition, inversion.
- **Eigenvectors & Eigenvalues**: Understanding their significance in data transformations and dimensionality reduction (PCA).

📚 **References:**

- [3Blue1Brown's "Essence of Linear algebra"](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [MIT Linear algebra course](https://www.youtube.com/playlist?list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k)
- [Brown University slides & computational applications of linear algebra](https://codingthematrix.com/)

<details>
<summary>Basic matrix operations illustrated</summary>
    
![Matrix-rules](https://github.com/user-attachments/assets/e98a2856-49df-47f3-b0d5-aeb5658c681a)

</details>
<details>
<summary>Code Example: Matrix Multiplication in Python</summary>
    
```
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = np.dot(A, B)
```
</details>

---

## Calculus 

Calculus is essential for understanding the optimization techniques used in machine learning, particularly for gradient-based learning algorithms.

![matlab](https://github.com/user-attachments/assets/65cb4b6a-15f7-4d46-92a7-e9cef469f250)


- **Differential Calculus**: Derivatives, Partial derivatives, Gradients.
- **Integral Calculus**: Integration, Area under curves, Summations.
- **Multivariable Calculus**: Functions of multiple variables.

📚 **References:**

- [3Blue1Brown's Essense of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [Khan academy's Multivariable calculus](https://www.khanacademy.org/math/multivariable-calculus)

<details>
<summary> Code Example: Calculating Gradient Descent in Python</summary>
    
```
import numpy as np

def gradient_descent(x, y, learning_rate=0.01, epochs=1000):
    w = 0
    b = 0
    n = len(x)
    
    for _ in range(epochs):
        y_pred = w * x + b
        dw = -2/n * sum(x * (y - y_pred))
        db = -2/n * sum(y - y_pred)
        w -= learning_rate * dw
        b -= learning_rate * db
    
    return w, b

**Example data**

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

w, b = gradient_descent(x, y)
print(f"Slope: {w}, Intercept: {b}")
```

</details>

--- 

## Programming 

The importance of programming is probably self explanatory. Programming skills are essential for implementing machine learning algorithms, working with data and everything else (This roadmap mainly focuses on Python).

- **Python**: General syntax, Control flow, Functions, Classes, File handling, Loops and so on.
- **Libraries**: NumPy, pandas, matplotlib, seaborn, Scikit-learn, OpenCV, NLTK, Spacy, TensorFlow, Keras, PyTorch etc..
- **Tools**: Jupyter Notebooks, git/github, package/environment managers.

📚 **References:**

### Python
- [Freecode camp's python for beginners](https://www.youtube.com/watch?v=rfscVS0vtbw&t=3182s)
- [W3 schools python](https://www.w3schools.com/python/default.asp)

### Essential Libraries
- [Keith Galli's complete Pandas tutorial](https://youtu.be/2uvysYbKdjM?si=rUAbKPo86p2SQqLW)
<details>
<summary>Basic Pandas cheat sheet</summary>
    
<img width="880" alt="Screen Shot 2024-07-26 at 11 27 33 AM" src="https://github.com/user-attachments/assets/e5d5784a-c0c0-4771-a778-130f96f6dc01">

</details>

- [Patrick loeber's Numpy crash course](https://www.youtube.com/watch?v=9JUAPgtkKpI&t=398s)

<details>
<summary>Basic Numpy cheat sheet</summary>

<img width="880" alt="Screen Shot 2024-07-26 at 11 28 08 AM" src="https://github.com/user-attachments/assets/945bd494-d8a4-4bd2-809a-e39ed4a18c01">

</details>

- [Neural nine's matplotlib crash course](https://www.youtube.com/watch?v=OZOOLe2imFo)

### Tools
- [Conda (package manager) tutorial](https://www.youtube.com/watch?v=sDCtY9Z1bqE)
- [Jupyter Notebooks](https://www.youtube.com/watch?v=IMrxB8Mq5KU)
- [Mosh Hamadani's github tutorial](https://www.youtube.com/watch?v=8JJ101D3knE&t=1001s)

---

## Data Preprocessing

A simple analogy would be to think of data as the ingredients required to cook a meal. The better the quality the better the meal. "Your ML model is only as good as your data".

- **Data Cleaning**: Handling/Imputing missing values and Removing duplicates.
- **Detecting & Handling Outliers**: Visual Methods (Eg: Box Plots), Statistical Methods (Eg: Std, Z-score etc.), capping/flooring etc..
- **Data Transformation**: Normalization (Feature scaling), log normalization, Standardization, Encoding categorical variables.
- **Feature Engineering**: Creating new features (Domain Knowledge), Selecting relevant features.
- **Handling Imbalanced Data**: Resampling (Over-Sampling, Under-Sampling), SMOTE, Class weight assignments
- **Data Splitting**: Train-test split, Cross-validation.

📚 **References:**

### EDA & Cleaning
- [Data Exploration](https://www.youtube.com/watch?v=OY4eQrekQvs&t=735s)
- [Data Cleaning](https://www.youtube.com/watch?v=qxpKCBV60U4&t=69s)

### Handling outliers
- [outliers (visual approach)](https://www.youtube.com/watch?v=T-ubh8EWpTg)
- [outliers (statistical approach)](https://www.youtube.com/playlist?list=PLeo1K3hjS3ut5olrDIeVXk9N3Q7mKhDxO)

### Data transformation/feature engineering
- [Feature scaling](https://www.youtube.com/watch?v=P3xPA7XFGCQ)
- [log normalization](https://www.youtube.com/watch?v=xtTX69JZ92w)
- [Encoding (one-hot, label)](https://www.youtube.com/watch?v=12Z6JBLKpts&t=432s)

### Handling imbalanced data
- [Handling imbalanced data](https://www.youtube.com/watch?v=JnlM4yLFNuo&t=46s)

### Data splits
- [Data splits (train-test-validation)](https://www.youtube.com/watch?v=dSCFk168vmo)


---

## Machine Learning Pt.1: Fundamentals

Before going into machine learning algorithms, it is important to undertand the different terminologies and concepts that underly these algorithms (note that this section only introduces these concepts)

- **Types of Models**: Supervised, Unsupervised, Reinforcement 
- **Bias and Variance**: Overfitting and Underfitting
- **Regularization**: L1, L2, Dropout and Early stopping
- **Cost Functions**: MSE (regression), Log loss/binary cross entropy (Binary Classification), categorical cross entropy (Multi-class classification with one-hot encoded features), sparse categorical cross entropy (Multi-class classifcation with integer features)
- **Optimizers**: Gradient descent, Stochastic gradient descent, small batch gradient descent, RMSprop, Adam

📚 **References:**

### Model Types
- [Assembly AI's Supervised learning video](https://youtu.be/Mu3POlNoLdc?si=vV2EPwvFRGlSj19G)
- [Assembly AI's Unsupervised learning video](https://www.youtube.com/watch?v=yteYU_QpUxs)
- [Reinforcement learning basic concepts](https://www.youtube.com/watch?v=nIgIv4IfJ6s)

### Bias & Variance 
- [Bias & Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA)

### Regularization
- [Regularization](https://www.youtube.com/watch?v=EehRcPo1M-Q&t=35s)

### Loss functions
- [Mean Squared Error (Regression)](https://www.youtube.com/watch?v=yt7fzvwfWHs)
- [log loss/binary cross entropy (binary classification)](https://www.youtube.com/watch?v=ar8mUO3d05w)
- [Categorical cross entropy (Multi-class classification)](https://www.youtube.com/watch?v=UlNB4R74z1A)

### Optimizers
- [Optimizers](https://www.youtube.com/watch?v=mdKjMPmcWjY)

---

## Machine Learning Pt.2: Algorithms

The statistical learning algorithms that do all the work we associate "Ai" with

- **Supervised Algorithms**: Linear regression, Logistic regression, Decision trees, Random forests, Support vector machines, k-nearest neighbors
- **Unsupervised Algorithms**: Clustering (k-Means, Hierarchical), Dimensionality reduction (PCA, t-SNE, LDA), recommenders (collaborative and content-based) and anomaly detection (Variational autoencoders)
- **Ensemble Methods**: Stacking, Boosting (adaboost, XGboost etc.) and Bagging (bootstrap aggregation)
- **Reinforcement Learning**: Q-learning, Deep Q-networks, Policy gradient methods, RLHF etc..
- **Extra**: [Choosing a model](https://youtube.com/playlist?list=PLVZqlMpoM6kaeaNvBTyoKdhMpOZzXu6AS&si=EBnWyp1PNv3STXSv)

📚 **References:**
<details>
<summary>Basic sk-learn cheat sheet</summary>
    
<img width="881" alt="Screen Shot 2024-07-26 at 11 28 48 AM" src="https://github.com/user-attachments/assets/528f11e9-8284-4fdf-84fe-887b9b047621">


</details>

<details>
<summary>model choice cheat sheet</summary>

![microsoft-machine-learning-algorithm-cheat-sheet-v7-1](https://github.com/user-attachments/assets/a58e3193-3be8-4e6c-b3dc-3313f7c93817)

</details>

<details>
<summary>Machine learning Tips & Tricks</summary>

[press here](Artifacts/cheatsheet-machine-learning-tips-and-tricks.pdf)

</details>
  
### [Supervised Algorithms](Artifacts/cheatsheet-supervised-learning.pdf)


- [Linear regression](https://www.youtube.com/playlist?list=PLblh5JKOoLUIzaEkCLIUxQFjPIlapw8nU)
  
<details>
<summary>Code Example: Linear regression with sk-learn</summary>

```
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict
predictions = model.predict(X)
print(predictions)
```
<details>
<summary>Example: illustration</summary>
    
![1_Nf2tTTkALYq6RTMQmhjo1A](https://github.com/user-attachments/assets/1ba5a1c2-65cd-4d1f-b856-5d700a62e883)
</details>

</details>

- [Logistic regression](https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe)

<details>
<summary>Code Example: Logistic Regression with sk-learn</summary>

```
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create and fit the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```
<details>
<summary>Example: illustration</summary>
    
![46-4-e1715636469361](https://github.com/user-attachments/assets/7a03c495-72df-4aad-a97b-14a69b7422d0)
</details>

</details>
  
- [Decision Trees](https://www.youtube.com/playlist?list=PLblh5JKOoLUKAtDViTvRGFpphEc24M-QH)
<details>
<summary>Code Example: Decision Trees with sk-learn</summary>
<details>    
<summary>Example: Classifier </summary>
    
```
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create and fit the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

```
</details>
<details>
<summary>Example: Regressor </summary>
    
```
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and fit the model
model = DecisionTreeRegressor()
model.fit(X, y)

# Predict
predictions = model.predict(X)
print(predictions)

```
</details>
<details>
<summary>Example: illustration</summary>

![structure-of-a-decision-tree](https://github.com/user-attachments/assets/6667d51d-6b6b-47a2-837a-87e1d46f56dc)
</details>

</details>

- [Random forest](https://www.youtube.com/playlist?list=PLblh5JKOoLUIE96dI3U7oxHaCAbZgfhHk)
  
<details>
<summary>Code Example: Random forests with sk-learn</summary>
<details>    
<summary>Example: Classifier </summary>
    
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create and fit the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

```
</details>
<details>
<summary>Example: Regressor </summary>
    
```
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and fit the model
model = RandomForestRegressor()
model.fit(X, y)

# Predict
predictions = model.predict(X)
print(predictions)

```
</details>
<details>
<summary>Example: illustration</summary>

![istockphoto-1358738588-612x612](https://github.com/user-attachments/assets/b63eeb1c-3435-483b-8869-3a90cdddd79b)
</details>

</details>

- [Support vector machines (SVM)](https://www.youtube.com/playlist?list=PLblh5JKOoLUL3IJ4-yor0HzkqDQ3JmJkc)
  
<details>
<summary>Code Example: SVMs with sk-learn</summary>
<details>    
<summary>Example: Classifier </summary>
    
```
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create and fit the model
model = SVC()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

```
</details>
<details>
<summary>Example: Regressor </summary>
    
```
from sklearn.svm import SVR
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and fit the model
model = SVR()
model.fit(X, y)

# Predict
predictions = model.predict(X)
print(predictions)

```
</details>

<details>
<summary>Example: illustration</summary>


![Illustration-of-linear-SVM-Classifier-separating-the-two-classes-Illustration-of-linear](https://github.com/user-attachments/assets/60c4f617-a893-4aa3-a769-11c059af63f4)
</details>
</details>

- [K-nearest neighbor (KNN)](https://www.youtube.com/watch?v=HVXime0nQeI&t=21s)

<details>
<summary>Code Example: KNNs with sk-learn</summary>
<details>
<summary>Example: Classifier</summary>
    
```
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create and fit the model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
 
```
</details>

<details>
<summary>Example: Regressor</summary>
    
```
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and fit the model
model = KNeighborsRegressor()
model.fit(X, y)

# Predict
predictions = model.predict(X)
print(predictions)
   
```
</details>
<details>
<summary>Example: illustration</summary>
    
![0_2_qzcm2gSe9l67aI](https://github.com/user-attachments/assets/703e7e16-8835-41b5-8fd2-c4b2c4f1552d)

</details>
</details>

- [Python implementation of ML algorithms (from scratch)](https://www.youtube.com/playlist?list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd)

### [Unsupervised Algorithms](Artifacts/cheatsheet-unsupervised-learning.pdf)

**Clustering**
- [hierarchial clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo)
- [k-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA&t=33s)

**Dimensionality reduction**
- [General overview of dimensionality reduction techniques](https://www.youtube.com/watch?v=jc1_yPYmspk)
- [PCA visually explained](https://www.youtube.com/watch?v=FD4DeN81ODY)
- [t-SNE explained](https://www.youtube.com/watch?v=NEaUSP4YerM)
- [Linear discriminant analysis (LDA)](https://www.youtube.com/watch?v=azXCzI57Yfc)

**Recommenders**
- [Content based recommenders](https://www.youtube.com/watch?v=4sy2tpzlgg4)
- [Collaborative filtering](https://www.youtube.com/watch?v=Fmtorg_dmM0&t=7s)

**Anomaly Detection**
- [Variational Autoencoders (VAEs)](https://www.youtube.com/watch?v=fcvYpzHmhvA)

### Ensemble methods
- [Ensemble learning overview](https://www.youtube.com/watch?v=sN5ZcJLDMaE)

### Reinforcement learning fundamentals
- [Reinforcement learning 101](https://www.youtube.com/playlist?list=PLTl9hO2Oobd9kS--NgVz0EPNyEmygV1Ha)
- [Reinforcement learning playlist 2](https://www.youtube.com/playlist?list=PLzvYlJMoZ02Dxtwe-MmH4nOB5jYlMGBjr)

--- 

## Model Evaluation and Validation

It is important to understand how your model is performing and whether it would be useful in the real world. Eg: A model with high variance would not generalize well and would therefore not perform well on unseen data.

- **Metrics: Accuracy**, Precision, Recall, F1 Score, confusion matrix, ROC-AUC, MSE, R squared etc
- **Validation Techniques**: Cross-validation, Hyperparameter tuning (Grid search, Random search), learning curves
- **Model Explainability (Optional)**: SHAP
 
📚 **References:**

### Classification evaluation metrics
- [Classification metrics (precision, recall, F1 score)](https://www.youtube.com/watch?v=2osIZ-dSPGE&t=586s)
- [Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o&t=323s)
- [ROC-AUC](https://www.youtube.com/watch?v=4jRBRDbJemM&t=77s)

### Regression evaluation metrics
- [R squared](https://www.youtube.com/watch?v=bMccdk8EdGo&t=7s)
- [Mean Squared Error (MSE)](https://statisticsbyjim.com/regression/mean-squared-error-mse/)
- [Mean Absolute Error (MAE)](https://medium.com/@m.waqar.ahmed/understanding-mean-absolute-error-mae-in-regression-a-practical-guide-26e80ebb97df)

### Validation techniques
- [Cross validation](https://www.youtube.com/watch?v=fSytzGwwBVw)
- [Grid search & Randomized search (hyperparameter tuning)](https://www.youtube.com/watch?v=HdlDYng8g9s)
- [Learning curves](https://www.youtube.com/watch?v=nt5DwCuYY5c)

### Model explainability
- [SHAP](https://www.youtube.com/playlist?list=PLqDyyww9y-1SJgMw92x90qPYpHgahDLIK)


---

## Deep Learning

This is where it gets interesting. Ever wondered how tools like GPT or Midjourney work? this is where it all starts. The idea is to build deep networks of "neurons" or "perceptrons" that use non-linear activations to understand abstract concepts that would generally confuse standard statistical models.

![1_ZXAOUqmlyECgfVa81Sr6Ew](https://github.com/user-attachments/assets/0026e041-c729-42e3-b074-a398d6034156)


- **Neural Networks**: architecture, Perceptrons, Activation functions (non-saturated), weights & initialization, optimizers (SGD, RMSprop, Adam etc), cost/loss functions, biases, back/forward propagation, gradient clipping, vanishing/exploding gradients, batch normalization etc..
- **Convolutional Neural Networks (CNNs)**: convolutional layers, pooling layers, kernels, feature maps
- **Recurrent Neural Networks and variants (RNNs/LSTMs/GRUs)**: Working with sequential data like audio, text, time series etc..
- **Transformers**: Text generation, attention mechanism
- **Autoencoders**: Latent variable representation (encoding/decoding) & feature extraction
- **Generative Adversarial Networks (GANs)**: Generators & Discriminators

📚 **References:**

- [3Blue1Brown's Neural Network playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [activation functions](https://www.youtube.com/watch?v=Fu273ovPBmQ&t=324s)

<details>
<summary>Code Example: Building a Simple Neural Network with Keras</summary>

```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
```
</details>


- [Convolutional Neural Networks](https://www.youtube.com/playlist?list=PLuhqtP7jdD8CD6rOWy20INGM44kULvrHu)

- [Recurrent Neural Networks](https://www.youtube.com/playlist?list=PLuhqtP7jdD8ARBnzj8SZwNFhwWT89fAFr)

- [Autoencoders](https://www.youtube.com/watch?v=6-fOJ79HW9U&t=179s)

- [Transformers](https://www.youtube.com/watch?v=_UVfwBqcnbM&t=38s)

- [Generative adversarial networks (GANs)](https://www.youtube.com/watch?v=8L11aMN5KY8)


---

## Natural Language Processing (NLP)

As the name suggests, this part focuses on technologies that deal with natural langauge (primarily text data).

- **Text Preprocessing**: Tokenization, Lemmatization, Stemming, stop-word removal
- **Text Representation (Feature Extraction)**: N-grams, Bag of Words, TF-IDF, Word Embeddings (Word2Vec, GloVe), FastText, Cosine similarity
- **NLP Models**: Sentiment analysis, Named entity recognition (NER), Machine translation, Language models etc.
- **Model Optimization**: Quantization, Pruning, Fine tuning, Prompt tuning
- **Specialized Techniques**: Zero-shot learning, Few-shot learning, Prompt Engineering, sequence/tree of prompts, self supervised learning, semi-supervised learning, RAG, topic modeling (LDA, NMF)
- **Large Language Models(LLMs)**: Working with APIs, local/Open-Source LLMs, huggingface transformers, Langchain etc.

📚 **References:**

### Preprocessing 
- [Tokenization](https://www.youtube.com/watch?v=_lR3RjvYvF4)
- [Stemming and Lemmatization](https://www.youtube.com/watch?v=HHAilAC3cXw)

### Text feature extraction
- [N-grams](https://www.youtube.com/watch?v=nZromH6F7R0)
- [Bag of words](https://www.youtube.com/watch?v=Yt1Sw6yWjlw)
- [TF-IDF](https://www.youtube.com/watch?v=ATK6fm3cYfI)
- [Word Embeddings](https://www.youtube.com/watch?v=Do8cVbx-HOs)
- [Cosine Similarity](https://www.youtube.com/watch?v=m_CooIRM3UI&t=539s)

### Optimizing NLP models
- [Quantization](https://www.youtube.com/watch?v=qqN63hbziaI)
- [Quantization & Pruning explained](https://www.youtube.com/watch?v=UcwDgsMgTu4&t=359s)
- [Fine tuning GPT](https://www.youtube.com/watch?v=y_VtqdK6io0)
- [Fine tuning LLMs](https://www.youtube.com/watch?v=eC6Hd1hFvos&t=18s)
- [Prompt tuning](https://www.youtube.com/watch?v=T_QhHvRxqUg)

- [LLM training simple overview](https://masteringllm.medium.com/llm-training-a-simple-3-step-guide-you-wont-find-anywhere-else-98ee218809e5)
- [RAG](https://www.youtube.com/watch?v=Ylz779Op9Pw)
- [Self-supervised learning](https://www.youtube.com/watch?v=sJzuNAisXHA)
- [Direct Preference Optimization](https://www.youtube.com/watch?v=XZLc09hkMwA)
- [Prompt engineering](https://www.youtube.com/watch?v=aOm75o2Z5-o&t=5s)

### Working with models 
- [local LLMs with LM Studio](https://www.youtube.com/watch?v=yBI1nPep72Q&t=75s)
- [Huggingface transformers library](https://www.youtube.com/watch?v=jan07gloaRg)
- [Langchain](https://www.youtube.com/watch?v=mrjq3lFz23s&t=61s)
- [Open AI API](https://www.youtube.com/watch?v=czvVibB2lRA&t=393s)

---

## Computer Vision (CV)

As the name suggests, this part focuses on technologies that deal with visual data (primarily images and video).

- **Image Preprocessing**: Resizing, Normalization, Augmentation, Noise Reduction
- **Feature Extraction**: Edge Detection, Histograms of Oriented Gradients (HOG), Keypoints and Descriptors (SIFT, SURF)
- **Image Representation**: [Convolutional](https://youtube.com/playlist?list=PLVZqlMpoM6kanmbatydXVhZpu3fkaTcbJ&si=qq9dWdAlIAxePwoF) Neural Networks (CNNs), Transfer Learning, Feature Maps
- **CV Models**: Object Detection (YOLO, Faster R-CNN), Image Classification (ResNet, VGG), Image Segmentation (U-Net, Mask R-CNN), Image Generation (GANs, VAEs), Stable Diffusion
- **Model Optimization**: Quantization, Pruning, Knowledge Distillation, Hyperparameter Tuning
- **Specialized Techniques**: Transfer Learning, Few-Shot Learning, Style Transfer, Image Super-Resolution, Zero-Shot Learning
- **Advanced Topics**:
  - **3D Vision**: Depth Estimation, 3D Reconstruction, and Stereo Vision.
  - **Video Analysis**: Action Recognition, Object Tracking, Video Segmentation.
- **Working with APIs and Tools**: OpenCV, TensorFlow, PyTorch, Pre-trained Models, Deployment (ONNX, TensorRT)

📚 **References:**

### Image Preprocessing
- [Image Preprocessing](https://www.youtube.com/watch?v=kSqxn6zGE0c)

### Feature extraction
- [Edge Detection](https://www.youtube.com/watch?v=4xq5oE9jJZg)
- [Histograms of Oriented Gradients (HOG)](https://www.youtube.com/watch?v=5nZGnYPyKLU)
- [Feature detection overview](https://www.youtube.com/playlist?list=PLSK7NtBWwmpR8VfRwSLrflmmthToXzTe_)
- [SIFT](https://www.youtube.com/watch?v=KgsHoJYJ4S8)

### Object detection models
- [YOLO](https://www.youtube.com/watch?v=ag3DLKsl2vk&t=489s)
- [Faster R-CNN](https://www.youtube.com/playlist?list=PL8hTotro6aVG6prsY92ZNVBNPr1PkXgsP)

### Image classification models
- [ResNet](https://www.youtube.com/watch?v=o_3mboe1jYI&t=392s)

### Image segmentation models
- [U-Net](https://www.youtube.com/watch?v=NhdzGfB1q74&t=81s)
- [Mask R-CNN](https://www.youtube.com/watch?v=4tkgOzQ9yyo)

### Generative models
- [Stable Diffusion](https://www.youtube.com/watch?v=sFztPP9qPRc&t=3s)

### Specialized techniques
- [Style transfer](https://www.youtube.com/playlist?list=PLBoQnSflObcmbfshq9oNs41vODgXG-608)
- [Image super resolution](https://www.youtube.com/watch?v=KULkSwLk62I)

### 3D vision
- [Depth estimation](https://www.youtube.com/watch?v=CpA2WAvynb0)
- [3D reconstruction](https://www.youtube.com/watch?v=tqBD6rxiul4&t=71s)
- [Stereo Vision](https://www.youtube.com/watch?v=KOSS24P3_fY)

### Working with videos
- [Human-based action recognition in videos](https://www.mdpi.com/2297-8747/28/2/61)
- [Overview of object tracking](https://supervisely.com/blog/complete-guide-to-object-tracking-best-ai-models-tools-and-methods-in-2023/)
- [Single & Multiple object tracking](https://www.youtube.com/@multipleobjecttracking1226/playlists)
- [Video Segmentation](https://www.v7labs.com/blog/video-segmentation-guide#:~:text=Video%20segmentation%20is%20the%20process,texture%2C%20or%20other%20visual%20features.)

---

## Data Niches

This section covers more complex, abstract and niche data types.

- **Time series Data**: Trends, seasonality, noise, cycles, stationarity, decomposition, differencing, autocorrelation, forecasting, libraries (Statsmodels, Prophet, PyFlux etc..) and models (ARIMA, Exponential smoothing, SARIMA etc)

- **Geospatial Data**: Coordinate systems (latitude, longitude), projections, shapefiles, GeoJSON, spatial joins, distance calculations, mapping libraries (GeoPandas, Folium, Shapely, etc.), and spatial analysis techniques (buffering, overlay, spatial clustering, etc.)

- **Network/graph Data**: Graph representation, nodes, edges, adjacency matrices, graph traversal algorithms (BFS, DFS), centrality measures (degree, closeness, betweenness, eigenvector),link measures, community detection, network visualization (NetworkX, Gephi, Cytoscape), deep graph library and PyTorch Geometric

- **Audio Data**: Feature extraction (MFCCs, spectrograms), audio preprocessing (normalization, noise reduction), signal processing, audio classification, libraries (Librosa, PyDub), and techniques (Fourier Transform, Short-Time Fourier Transform)

- **Multimodal Data**: Data integration (e.g., text and images, audio and video), feature fusion, modality alignment, joint representation learning, applications in recommendation systems, multi-modal sentiment analysis, and data fusion techniques.

- **Biological Data**: DNA sequences, protein structures, genomic variations, biological networks, molecular dynamics, libraries (Biopython, PySCeS), and techniques (sequence alignment, structural bioinformatics).

- **Astronomical Data**: Star catalogs, galaxy surveys, light curves, cosmic microwave background data, celestial object tracking, libraries (AstroPy, HEAsoft), and techniques (source extraction, cosmic event detection).


📚 **References:**

**(Yet to add references)**

---

## Deployment

Deploying machine learning models to production.

- **Model Serving**: Flask, FastAPI, Streamlit, TensorFlow Serving
- **Persisting Models**: Pickle & Joblib
- **Containerization**: Docker, Kubernetes
- **Cloud Services**: AWS, Google Cloud, Azure and IBM Watson

📚 **References:**

### Model serving
- [Flask tutorial](https://www.youtube.com/watch?v=Z1RJmh_OqeA)
- [Flask ML web app](https://www.youtube.com/watch?v=qNF1HqBvpGE&t=2657s)
- [Fast API](https://www.youtube.com/watch?v=tLKKmouUams)
- [Streamlit tutorial series](https://www.youtube.com/playlist?list=PLtqF5YXg7GLmCvTswG32NqQypOuYkPRUE)
- [Tensorflow model serving](https://www.youtube.com/watch?v=P-5sMcpTE0g)

### Model persistence
- [Model persistence](https://www.youtube.com/watch?v=jwhSLGmBEpU&t=20s)

### Containerization
- [Docker tutorial](https://www.youtube.com/watch?v=pTFZFxd4hOI&t=159s)
- [Kubernetes](https://www.youtube.com/watch?v=X48VuDVv0do)


---

<details>
  <summary>Click here if you're ready for more</summary>
  
  ## Notable Extras (Random fun stuff)

  - **GPU Acceleration Libraries**: CUML and CUDF from RAPIDs (SkLearn)
  - **CI/CD for ML**: Automated Pipelines (enkins, GitLab CI, and CircleCI), Versioning (DVC, MLflow), experiment/metadata tracking (MLflow).
  - **Model Monitoring**: Tracking model performance and detecting [drift](https://www.youtube.com/watch?v=uOG685WFO00) using tools like Prometheus, Grafana, and custom monitoring solutions.
  - **System Design**: [Freecodecamp's video](https://www.youtube.com/watch?v=F2FmTdLtb_4)
  - **Database design**: [Freecodecamp's video](https://www.youtube.com/watch?v=ztHopE5Wnpc), [Practice SQL](https://sqlbolt.com/)
  - **OCR Libraries**: [Tesseract OCR](https://www.youtube.com/watch?v=PY_N1XdFp4w&t=32s), [EasyOCR](https://www.youtube.com/watch?v=GboDfGzkRsQ)
    
</details>
<details>
<summary>extra resources (Books, newsletters etc)</summary>
    
- [ML Blogs & Newsletters](https://github.com/josephmisiti/awesome-machine-learning/blob/master/blogs.md)
- [ML Books](https://github.com/josephmisiti/awesome-machine-learning/blob/master/books.md)
- [List of ML courses (free and paid)](https://github.com/josephmisiti/awesome-machine-learning/blob/master/courses.md)

</details>
