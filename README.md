![Linear_subspaces_with_shading](https://github.com/user-attachments/assets/6c1ca9a4-c4dd-4cb5-ad60-af7ea59815ad)# Machine Learning Roadmap 

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
- [Deployment](#deployment)

---

## Statistics


<img width="493" alt="Screen Shot 2024-07-26 at 12 06 31 PM" src="https://github.com/user-attachments/assets/7d6c8c8e-2fa1-4074-829e-8fdaaa2a06ec">

Statistics can be thought of as the "foundational toolkit" for understanding and interpreting data. It's crucial for tasks like data analysis, building probabilistic models, and evaluating model performance and so on. To build a general intuition for the statistics required for data science, focus on the following areas:

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

![Uploading Lin<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->
<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="325"
   height="235.95"
   id="svg2607"
   sodipodi:version="0.32"
   inkscape:version="0.46"
   version="1.0"
   sodipodi:docbase="C:\Documents and Settings\Kris\My Documents\My Pictures\SVG"
   sodipodi:docname="Linear-subspace-with-shading.svg"
   inkscape:output_extension="org.inkscape.output.svg.inkscape">
  <defs
     id="defs2609">
    <inkscape:perspective
       sodipodi:type="inkscape:persp3d"
       inkscape:vp_x="0 : 117.975 : 1"
       inkscape:vp_y="0 : 1000 : 0"
       inkscape:vp_z="325 : 117.975 : 1"
       inkscape:persp3d-origin="162.5 : 78.649999 : 1"
       id="perspective3352" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="2"
     inkscape:cx="142.70649"
     inkscape:cy="112.44948"
     inkscape:document-units="mm"
     inkscape:current-layer="layer1"
     width="100mm"
     height="99.999997mm"
     units="mm"
     showgrid="false"
     showguides="true"
     inkscape:guide-bbox="true"
     grid_units="mm"
     inkscape:grid-points="true"
     gridtolerance="10000"
     inkscape:window-width="1124"
     inkscape:window-height="732"
     inkscape:window-x="54"
     inkscape:window-y="9">
    <inkscape:grid
       id="GridFromPre046Settings"
       type="xygrid"
       originx="0px"
       originy="0px"
       spacingx="1mm"
       spacingy="1mm"
       color="#0000ff"
       empcolor="#0000ff"
       opacity="0.2"
       empopacity="0.4"
       empspacing="5"
       units="mm"
       visible="true"
       enabled="true" />
  </sodipodi:namedview>
  <metadata
     id="metadata2612">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(-45.92889,-27.636333)">
    <path
       style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1.06544995000000010;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       d="M 261.34375 56.78125 L 156.6875 127.6875 C 156.74255 127.7643 156.75359 127.86709 156.84375 127.90625 C 157.03281 127.98837 157.25126 127.97962 157.40625 127.84375 L 261.25 57.5 C 261.46935 57.3608 261.56922 57.083325 261.46875 56.84375 C 261.44866 56.795835 261.37518 56.819904 261.34375 56.78125 z "
       id="path3244"
       transform="translate(45.92889,27.636333)" />
    <path
       style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1.06544995;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       d="M 157.15625 127.21875 L 58.40625 111.5625 C 58.392262 111.61019 58.3125 111.60422 58.3125 111.65625 C 58.312501 111.93485 58.50345 112.16641 58.78125 112.1875 L 156.53125 127.6875 C 156.82465 127.73065 157.11311 127.51215 157.15625 127.21875 z "
       id="path3236"
       transform="translate(45.92889,27.636333)" />
    <path
       style="fill:#a4a4a4;fill-opacity:0.80232561;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 304.24139,110.79258 L 171.42889,108.13633 L 106.11639,148.48008 L 202.39764,154.98008 L 304.24139,110.79258 z"
       id="path3287" />
    <path
       style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1.06544995;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       d="M 157.15625 127.21875 C 157.1994 126.92534 156.9809 126.66815 156.6875 126.625 L 58.9375 111.125 C 58.8857 111.11732 58.83305 111.11732 58.78125 111.125 C 58.555335 111.14215 58.467157 111.35486 58.40625 111.5625 L 157.15625 127.21875 z "
       id="path3806"
       transform="translate(45.92889,27.636333)" />
    <path
       style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1.06544995000000010;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       d="M 261.34375 56.78125 C 261.21801 56.626632 261.04617 56.484136 260.84375 56.53125 C 260.77566 56.549774 260.71192 56.581641 260.65625 56.625 L 156.8125 126.96875 C 156.62898 127.06258 156.52478 127.23149 156.53125 127.4375 C 156.53464 127.54526 156.62712 127.60327 156.6875 127.6875 L 261.34375 56.78125 z "
       id="path3784"
       transform="translate(45.92889,27.636333)" />
    <path
       style="fill:#00ac51;fill-opacity:0.80232561;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 182.52264,200.82383 L 232.83514,82.980083 L 168.42889,53.136333 L 116.42889,174.63633 L 182.52264,200.82383 z"
       id="path3275" />
    <path
       style="fill:#a4a4a4;fill-opacity:0.80232561;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 202.39764,154.98008 L 106.11639,148.48008 L 45.92889,185.63633 L 103.36639,197.91758 L 202.39764,154.98008 z"
       id="path3291" />
    <path
       style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1.02158522999999990;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       d="M 157.03125 127.78125 L 157.03125 17.03125 C 157.00748 17.030342 156.99272 16.997539 156.96875 17 C 156.69514 17.022364 156.48812 17.256981 156.5 17.53125 L 156.5 127.28125 C 156.5 127.56602 156.74648 127.78125 157.03125 127.78125 z "
       id="path3250"
       transform="translate(45.92889,27.636333)" />
    <path
       style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1.02158522999999990;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       d="M 157.03125 126.8125 C 156.98734 126.81096 156.94999 126.77122 156.90625 126.78125 C 156.66956 126.83076 156.50001 127.03944 156.5 127.28125 L 156.5 219.5 C 156.5 219.78477 156.74648 220 157.03125 220 L 157.03125 126.8125 z "
       id="path3766"
       transform="translate(45.92889,27.636333)" />
    <rect
       style="fill:#fdff23;fill-opacity:0.80232561;stroke:none;stroke-width:1.29192400000000010;stroke-linecap:round;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
       id="rect2421"
       width="140.14482"
       height="153.81146"
       x="172.43365"
       y="151.67435"
       transform="matrix(0.9400316,-0.3410873,0,1,0,0)" />
    <path
       style="fill:#a4a4a4;fill-opacity:0.80232561;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 304.24139,110.79258 L 202.39764,154.98008 L 322.96014,163.10508 L 370.92889,112.13633 L 304.24139,110.79258 z"
       id="path3282" />
    <path
       style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1.02158522999999990;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       d="M 157.03125 126.8125 L 157.03125 220 C 157.31603 220 157.53125 219.78477 157.53125 219.5 L 157.53125 127.28125 C 157.53188 127.12534 157.4655 126.9724 157.34375 126.875 C 157.25704 126.80563 157.13991 126.81631 157.03125 126.8125 z "
       id="path2625"
       transform="translate(45.92889,27.636333)" />
    <path
       style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1.02158522999999990;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       d="M 157.03125 127.78125 C 157.31603 127.78125 157.53125 127.56602 157.53125 127.28125 L 157.53125 17.53125 C 157.53796 17.382812 157.48302 17.227025 157.375 17.125 C 157.28449 17.03952 157.15405 17.03594 157.03125 17.03125 L 157.03125 127.78125 z "
       id="path3757"
       transform="translate(45.92889,27.636333)" />
    <path
       style="fill:#00ac51;fill-opacity:0.80232561;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 182.52264,200.82383 L 256.42889,230.13633 L 304.42889,116.13633 L 232.83514,82.980083 L 182.52264,200.82383 z"
       id="path3280" />
    <path
       style="fill:none;fill-rule:evenodd;stroke:#0000ff;stroke-width:4.07480315000000020;stroke-linecap:round;stroke-linejoin:round;marker-start:none;marker-mid:none;marker-end:none;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       d="M 244.60716,55.30398 L 161.32196,250.2604"
       id="path3465"
       sodipodi:nodetypes="cc" />
    <path
       style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1.06544995000000010;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       d="M 70.5 186.0625 C 70.555048 186.1393 70.566092 186.24209 70.65625 186.28125 C 70.845298 186.36338 71.06376 186.35462 71.21875 186.21875 L 157.40625 127.84375 C 157.6256 127.70455 157.72547 127.42707 157.625 127.1875 C 157.60491 127.13958 157.53143 127.16365 157.5 127.125 L 70.5 186.0625 z "
       id="path3775"
       transform="translate(45.92889,27.636333)" />
    <path
       style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1.06544995;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       d="M 273.0625 145.625 L 156.125 127.0625 C 156.11493 127.10343 156.0625 127.11235 156.0625 127.15625 C 156.0625 127.43485 156.25345 127.66641 156.53125 127.6875 L 272.46875 146.09375 C 272.76215 146.14553 273.04197 145.94965 273.09375 145.65625 C 273.0967 145.63951 273.06113 145.64161 273.0625 145.625 z "
       id="path3788"
       transform="translate(45.92889,27.636333)" />
    <path
       style="fill:#a4a4a4;fill-opacity:0.80232561;fill-rule:evenodd;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"
       d="M 202.39764,154.98008 L 103.36639,197.91758 L 258.92889,231.13633 L 322.96014,163.10508 L 202.39764,154.98008 z"
       id="path3289" />
    <path
       style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1.06544995;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       d="M 273.0625 145.625 C 273.08515 145.35046 272.93291 145.08008 272.65625 145.03125 L 156.6875 126.625 C 156.6357 126.61732 156.58305 126.61732 156.53125 126.625 C 156.29723 126.64276 156.17882 126.84371 156.125 127.0625 L 273.0625 145.625 z "
       id="path3797"
       transform="translate(45.92889,27.636333)" />
    <path
       style="fill:#000000;fill-opacity:1;fill-rule:evenodd;stroke:none;stroke-width:1.06544995000000010;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       d="M 70.5 186.0625 L 157.5 127.125 C 157.37426 126.97038 157.20242 126.82789 157 126.875 C 156.93191 126.89352 156.86817 126.92539 156.8125 126.96875 L 70.625 185.34375 C 70.441476 185.43758 70.33728 185.60649 70.34375 185.8125 C 70.347141 185.92026 70.439625 185.97827 70.5 186.0625 z "
       id="path3372"
       transform="translate(45.92889,27.636333)" />
    <g
       id="g4640"
       transform="translate(35.067499,-54.293637)" />
  </g>
</svg>
ear_subspaces_with_shading.svg…]()


Linear algebra is fundamental for understanding how data is represented and manipulated in machine learning algorithms. It forms the backbone of many ML techniques, especially in deep learning where operations on large matrices are common.


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
- **Reinforcement Learning**: Q-learning, Deep Q-networks
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
  
### Supervised Algorithms
<details>
<summary>Supervised learning cheat sheet</summary>

[press here](Artifacts/cheatsheet-supervised-learning.pdf)

</details>

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

### Unsupervised Algorithms

<details>
<summary>Unsupervised learning cheat sheet</summary>

[press here](Artifacts/cheatsheet-unsupervised-learning.pdf)

</details>

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
- **Specialized Techniques**: Zero-shot learning, Few-shot learning, Prompt Engineering, sequence/tree of prompts, self supervised learning, semi-supervised learning, RAG
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
