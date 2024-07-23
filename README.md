# Machine Learning Roadmap

**This is a compilation of all the resources I gathered while learning the skills required to build machine learning projects**

**I originally made this to serve as a roadmap for me to follow; but I then thought, why not make it publicly viewable? Anyways, hope you have fun with this repo!**

**PS: This is targeted at individuals that already understand basic programming concepts and things like version control.**

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
- [Deployment](#deployment)

---

## Statistics

Statistics can be thought of as the "foundational toolkit" for understanding and interpreting data. It's crucial for tasks like data analysis, building probabilistic models, and evaluating model performance and so on. To build a general intuition for the statistics required for data science, focus on the following areas:

- **Descriptive Statistics**: Measures of Central Tendency (mean, median, mode), Measures of Dispersion (range, variance, standard deviation).
- **Inferential Statistics**: Hypothesis testing, Confidence intervals, P-values, t-tests, chi-square tests.
- **Probability Theory**: Basic probability, Conditional probability, Bayes' theorem.
- **Distributions**: Normal distribution, Binomial distribution, Poisson distribution.
- **Bayesian Statistics**: Bayesian inference, Prior and posterior probabilities.


📚 **References:**

---

## Linear algebra

Linear algebra is fundamental for understanding how data is represented and manipulated in machine learning algorithms. It forms the backbone of many ML techniques, especially in deep learning where operations on large matrices are common.


- **Vectors & Vector Spaces**: Addition, subtraction, scalar multiplication, dot products, linear transformations.
- **Matrices & Matrix Operation**s: Basic concepts, addition, subtraction, multiplication, transposition, inversion.
- **Eigenvectors & Eigenvalues**: Understanding their significance in data transformations and dimensionality reduction (PCA).

📚 **References:**

---

## Calculus 

Calculus is essential for understanding the optimization techniques used in machine learning, particularly for gradient-based learning algorithms.

- **Differential Calculus**: Derivatives, Partial derivatives, Gradients.
- **Integral Calculus**: Integration, Area under curves, Summations.
- **Multivariable Calculus**: Functions of multiple variables.

📚 **References:**

--- 

## Programming 

The importance of programming is probably self explanatory. Programming skills are essential for implementing machine learning algorithms, working with data and everything else (This roadmap mainly focuses on Python).

- **Python**: General syntax, Control flow, Functions, Classes, File handling (pickle/Joblib/Json), Loops and so on.
- **Libraries**: NumPy, pandas, matplotlib, seaborn, Scikit-learn, TensorFlow, Keras, PyTorch etc..
- **Tools**: Jupyter Notebooks, git/github, package/environment managers.

📚 **References:**

---

## Data Preprocessing

A simple analogy would be to think of data as the ingredients required to cook a meal. The better the quality the better the meal. "Your ML model is only as good as your data".

- **Data Cleaning**: Handling/Imputing missing values and Removing duplicates.
- **Detecting & Handling Outliers**: Visual Methods (Eg: Box Plots), Statistical Methods (Eg: Std, Z-score etc.), capping/flooring etc..
- **Data Transformation**: Normalization (Feature scaling), Standardization, Encoding categorical variables.
- **Feature Engineering**: Creating new features (Domain Knowledge), Selecting relevant features.
- **Handling Imbalanced Data**: Resampling (Over-Sampling, Under-Sampling), SMOTE, Class weight assignments
- **Data Splitting**: Train-test split, Cross-validation.

📚 **References:**


---

## Machine Learning Pt.1: Fundamentals

Before going into machine learning algorithms, it is important to undertand the different terminologies and concepts that underly these algorithms

- **Types of Models**: Supervised, Unsupervised, Reinforcement 
- **Bias and Variance**: Overfitting and Underfitting
- **Regularization**: L1, L2, Dropout and Early stopping

📚 **References:**

---

## Machine Learning Pt.2: Algorithms

The statistical learning algorithms that do all the work we associate "Ai" with

- **Supervised Algorithms**: Linear regression, Logistic regression, Decision trees, Random forests, Support vector machines, k-nearest neighbors
- **Unsupervised Algorithms**: Clustering, Dimensionality reduction, recommenders and anomaly detection
- **Ensemble Methods**: Stacking, Boosting and Bagging
- **Reinforcement Learning**: Q-learning, Deep Q-networks


📚 **References:**

--- 

## Model Evaluation and Validation

It is important to understand how your model is performing and whether it would be useful in the real world. Eg: A model with high variance would not generalize well and would therefore not perform well on unseen data.

- **Metrics: Accuracy**, Precision, Recall, F1 Score, confusion matrix, ROC-AUC, MSE, R squared etc
- **Validation Techniques**: Cross-validation, Hyperparameter tuning, Grid search, Random search, learning curves
- **Model Explainability (Optional)**: SHAP
 
📚 **References:**

---

## Deep Learning

This is where it gets interesting. Ever wondered how tools like GPT or Midjourney work? this is where it all starts. The idea is to build deep networks of "neurons" or "perceptrons" that use non-linear activations to understand abstract concepts that would generally confuse standard statistical models.

- **Neural Networks**: architecture, Perceptrons, Activation functions, weights & initialization, biases etc..
- **Convolutional Neural Networks (CNNs)**: Image classification, object detection, image segmentation
- **Recurrent Neural Networks and variants (RNNs/LSTMs/GRUs)**: Working with sequential data like audio, text, time series etc..
- **Transformers**: Text generation, attention mechanism
- **Autoencoders**: Latent variable representation (encoding/decoding) & feature extraction
- **Generative Adversarial Networks (GANs)**: Generators & Discriminators

📚 **References:**

---

## Natural Language Processing (NLP)

As the name suggests, this part focuses on technologies that deal with natural langauge (primarily text data).

- **Text Preprocessing**: Tokenization, Lemmatization, Stemming, stop-word removal
- **Text Representation (Feature Extraction)**: Bag of Words, TF-IDF, Word Embeddings (Word2Vec, GloVe), FastText
- **NLP Models**: Sentiment analysis, Named entity recognition (NER), Machine translation, Language models etc.
- **Model Optimization**: Quantization, Pruning, Fine tuning, Prompt tuning
- **Specialized Techniques**: Zero-shot learning, Few-shot learning, Prompt Engineering, sequence/tree of prompts, self supervised learning, RAG, Langchain
- **Large Language Models(LLMs)**: Working with APIs, local Open-Source LLMs etc.

📚 **References:**

---

## Computer Vision (CV)

As the name suggests, this part focuses on technologies that deal with visual data (primarily images and video).

- **Image Preprocessing**: Resizing, Normalization, Augmentation, Noise Reduction
- **Feature Extraction**: Edge Detection, Histograms of Oriented Gradients (HOG), Keypoints and Descriptors (SIFT, SURF)
- **Image Representation**: Convolutional Neural Networks (CNNs), Transfer Learning, Feature Maps
- **CV Models**: Object Detection (YOLO, Faster R-CNN), Image Classification (ResNet, VGG), Image Segmentation (U-Net, Mask R-CNN), Image Generation (GANs, VAEs), Stable Diffusion
- **Model Optimization**: Quantization, Pruning, Knowledge Distillation, Hyperparameter Tuning
- **Specialized Techniques**: Transfer Learning, Few-Shot Learning, Style Transfer, Image Super-Resolution, Zero-Shot Learning
- **Advanced Topics**:
  - **3D Vision**: Depth Estimation, 3D Reconstruction, and Stereo Vision.
  - **Video Analysis**: Action Recognition, Object Tracking, Video Segmentation.
- **Working with APIs and Tools**: OpenCV, TensorFlow, PyTorch, Pre-trained Models, Deployment (ONNX, TensorRT)

📚 **References:**

---

## Deployment

Deploying machine learning models to production.

**Model Serving**: Flask, FastAPI, Streamlit, TensorFlow Serving
**Containers**: Docker, Kubernetes
**Cloud Services**: AWS, Google Cloud, Azure and IBM Watson

📚 **References:**

---

<details>
  <summary>Click to expand</summary>
  
  ## Extras (Random fun stuff)

  - **GPU Acceleration Libraries**: CUML and CUDF from RAPIDs (SkLearn)
  - **OCR Libraries**: Tesseract OCR, EasyOCR

</details>
