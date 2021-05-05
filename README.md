# **Heart-Attack-Classification**

Heart Attack Analysis &amp; Prediction 

<img src = "Heart_Attac/img/Heart-attack-diagram.jpg" height = 100>
(Source: www.anatomynote.com)


# About the Dataset

1. age - age in years

2. sex - sex (1 = male; 0 = female)

3. cp - chest pain type (0 = typical angina; 1 = atypical angina; 2 = non-anginal pain; 3 = asymptomatic)

4. trestbps - resting blood pressure (in mm Hg on admission to the hospital)

5. chol - serum cholestoral in mg/dl

6. fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)

7. restecg - resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy)

8. thalach - maximum heart rate achieved

9. exang - exercise induced angina (1 = yes; 0 = no)

10. oldpeak - ST depression induced by exercise relative to rest

11. slope - the slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)

12. ca - number of major vessels (0-3) colored by flourosopy

13. thal - 0 = NULL; 1 = fixed defect; 2 = normal; 3 = reversable defect

14. output : 0= less chance of heart attack 1= more chance of heart attack



# Problem Definition
Classifiy people who has high or low heart attack risk



<h2>Requirements</h2>

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
```

