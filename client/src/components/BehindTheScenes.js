// dependencies
import React, { useEffect } from 'react'
import Prism from 'prismjs'

// images
import cutaway from '../img/titanic-cutaway-diagram.png'

// styles
import '../css/prism.css'

const BehindTheScenes = () => {
    // run prism when the component mounts
    useEffect(() => {
        Prism.highlightAll()
    }, [])

    return (
        <div className="row">
            <div className="col s12">
                <div>
                    <div>
                        <div>
                            <div>
                                <div id="title" className="scrollspy">
                                    <h1>Predicting the Survival of Titanic Passengers</h1>
                                    <p className="caption">The sinking of the Titanic is one of the most infamous shipwrecks
                                        in history.</p>
                                    <p className="caption">On April 15, 1912, during her maiden voyage, the widely
                                        considered “unsinkable” RMS Titanic sank
                                        after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats
                                        for everyone
                                        onboard, resulting in the death of 1502 out of 2224 passengers and crew.</p>
                                    <p className="caption">While there was some element of luck involved in surviving, it
                                        seems some groups of people were
                                        more likely to survive than others.</p>
                                    <p className="caption">This tutorial will explore the dataset from <a
                                            href="https://www.kaggle.com/c/titanic/">Kaggle's
                                            Titanic machine learning competition</a> and create a trained machine
                                        learning model that will be used in a web app to make survival predictions based
                                        on a user's input.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div id="explore" className="scrollspy">
                                    <h1>Part I: Explore the Data</h1>
                                    <h4>Import dependencies</h4>
                                </div>
                                <div>In [1]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# data visualization
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns

# linear algebra
import numpy as np

# data processing
import pandas as pd

# algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

# serializing
import pickle`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Import the dataset</h4>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [2]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# import training data
train = pd.read_csv('./input/train.csv')

# # import test data
# test = pd.read_csv('./input/test.csv')`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Explore the dataset</h4>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [3]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`list(train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [3]:</div>
                                        <div>
<pre><code className="language-bash">
'
{`['PassengerId',
'Survived',
'Pclass',
'Name',
'Sex',
'Age',
'SibSp',
'Parch',
'Ticket',
'Fare',
'Cabin',
'Embarked']`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <table>
                                        <tbody>
                                            <tr>
                                                <td style={{ "fontWeight": "bold", "textAlign": "left" }}>PassengerId</td>
                                                <td style={{ "textAlign": "left" }}>Unique ID</td>
                                            </tr>
                                            <tr>
                                                <td style={{ "fontWeight": "bold", "textAlign": "left" }}>Survived</td>
                                                <td style={{ "textAlign": "left" }}>Survival status</td>
                                            </tr>
                                            <tr>
                                                <td style={{ "fontWeight": "bold", "textAlign": "left" }}>Pclass</td>
                                                <td style={{ "textAlign": "left" }}>Ticket class</td>
                                            </tr>
                                            <tr>
                                                <td style={{ "fontWeight": "bold", "textAlign": "left" }}>Name</td>
                                                <td style={{ "textAlign": "left" }}>Name</td>
                                            </tr>
                                            <tr>
                                                <td style={{ "fontWeight": "bold", "textAlign": "left" }}>Sex</td>
                                                <td style={{ "textAlign": "left" }}>Genger</td>
                                            </tr>
                                            <tr>
                                                <td style={{ "fontWeight": "bold", "textAlign": "left" }}>Age</td>
                                                <td style={{ "textAlign": "left" }}>Age in years</td>
                                            </tr>
                                            <tr>
                                                <td style={{ "fontWeight": "bold", "textAlign": "left" }}>SibSp</td>
                                                <td style={{ "textAlign": "left" }}>Number of siblings or spouses aboard the
                                                    Titanic</td>
                                            </tr>
                                            <tr>
                                                <td style={{ "fontWeight": "bold", "textAlign": "left" }}>Parch</td>
                                                <td style={{ "textAlign": "left" }}>Number of parents or children aboard the
                                                    Titanic</td>
                                            </tr>
                                            <tr>
                                                <td style={{ "fontWeight": "bold", "textAlign": "left" }}>Ticket</td>
                                                <td style={{ "textAlign": "left" }}>Ticket number</td>
                                            </tr>
                                            <tr>
                                                <td style={{ "fontWeight": "bold", "textAlign": "left" }}>Fare</td>
                                                <td style={{ "textAlign": "left" }}>Price paid</td>
                                            </tr>
                                            <tr>
                                                <td style={{ "fontWeight": "bold", "textAlign": "left" }}>Cabin</td>
                                                <td style={{ "textAlign": "left" }}>Cabin number</td>
                                            </tr>
                                            <tr>
                                                <td style={{ "fontWeight": "bold", "textAlign": "left" }}>Embarked</td>
                                                <td style={{ "textAlign": "left" }}>Port of embarkation</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Summarize the dataframe</h4>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [4]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train.info()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">The training dataset has <strong>891 examples</strong> and
                                        <strong>11 features</strong> plus the
                                        <strong>1 target variable</strong> 'Survived'.</p>
                                    <p className="caption">5 of the features are objects<br />
                                        4 of the features plus the 1 target variable are integers<br />
                                        2 of the features are floats</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Generate descriptive statistics</h4>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [5]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train.describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [5]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th></th>
                                                            <th>PassengerId</th>
                                                            <th>Survived</th>
                                                            <th>Pclass</th>
                                                            <th>Age</th>
                                                            <th>SibSp</th>
                                                            <th>Parch</th>
                                                            <th>Fare</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>count</td>
                                                            <td>891.000000</td>
                                                            <td>891.000000</td>
                                                            <td>891.000000</td>
                                                            <td>714.000000</td>
                                                            <td>891.000000</td>
                                                            <td>891.000000</td>
                                                            <td>891.000000</td>
                                                        </tr>
                                                        <tr>
                                                            <td>mean</td>
                                                            <td>446.000000</td>
                                                            <td>0.383838</td>
                                                            <td>2.308642</td>
                                                            <td>29.699118</td>
                                                            <td>0.523008</td>
                                                            <td>0.381594</td>
                                                            <td>32.204208</td>
                                                        </tr>
                                                        <tr>
                                                            <td>std</td>
                                                            <td>257.353842</td>
                                                            <td>0.486592</td>
                                                            <td>0.836071</td>
                                                            <td>14.526497</td>
                                                            <td>1.102743</td>
                                                            <td>0.806057</td>
                                                            <td>49.693429</td>
                                                        </tr>
                                                        <tr>
                                                            <td>min</td>
                                                            <td>1.000000</td>
                                                            <td>0.000000</td>
                                                            <td>1.000000</td>
                                                            <td>0.420000</td>
                                                            <td>0.000000</td>
                                                            <td>0.000000</td>
                                                            <td>0.000000</td>
                                                        </tr>
                                                        <tr>
                                                            <td>25%</td>
                                                            <td>223.500000</td>
                                                            <td>0.000000</td>
                                                            <td>2.000000</td>
                                                            <td>20.125000</td>
                                                            <td>0.000000</td>
                                                            <td>0.000000</td>
                                                            <td>7.910400</td>
                                                        </tr>
                                                        <tr>
                                                            <td>50%</td>
                                                            <td>446.000000</td>
                                                            <td>0.000000</td>
                                                            <td>3.000000</td>
                                                            <td>28.000000</td>
                                                            <td>0.000000</td>
                                                            <td>0.000000</td>
                                                            <td>14.454200</td>
                                                        </tr>
                                                        <tr>
                                                            <td>75%</td>
                                                            <td>668.500000</td>
                                                            <td>1.000000</td>
                                                            <td>3.000000</td>
                                                            <td>38.000000</td>
                                                            <td>1.000000</td>
                                                            <td>0.000000</td>
                                                            <td>31.000000</td>
                                                        </tr>
                                                        <tr>
                                                            <td>max</td>
                                                            <td>891.000000</td>
                                                            <td>1.000000</td>
                                                            <td>3.000000</td>
                                                            <td>80.000000</td>
                                                            <td>8.000000</td>
                                                            <td>6.000000</td>
                                                            <td>512.329200</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">We can see that 38% of the training set survived the Titanic.
                                        This is roughly inline with the actual
                                        survival rate of 32%.</p>
                                    <p className="caption">We can also make other other early observations such as the 'Age'
                                        feature's values range from 0.4 (ages
                                        less than 1 are represented as a fractional value in the dataset) to 80, the
                                        'Fare' feature's maximum
                                        value is $512.33 but the average value is only $32.20 (and the 75th
                                        percentile value is only $31.00).
                                        In addition there are 177 fewer values for the 'Age' feature than there are any
                                        other feature, so we can
                                        already identify that the dataset contains missing values that we will need to
                                        deal with.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Preview the dataframe</h4>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [6]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train.head(10)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [6]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th>PassengerId</th>
                                                            <th>Survived</th>
                                                            <th>Pclass</th>
                                                            <th>Name</th>
                                                            <th>Sex</th>
                                                            <th>Age</th>
                                                            <th>SibSp</th>
                                                            <th>Parch</th>
                                                            <th>Ticket</th>
                                                            <th>Fare</th>
                                                            <th>Cabin</th>
                                                            <th>Embarked</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Braund, Mr. Owen Harris</td>
                                                            <td>male</td>
                                                            <td>22.0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>A/5 21171</td>
                                                            <td>7.2500</td>
                                                            <td>NaN</td>
                                                            <td>S</td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
                                                            <td>female</td>
                                                            <td>38.0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>PC 17599</td>
                                                            <td>71.2833</td>
                                                            <td>C85</td>
                                                            <td>C</td>
                                                        </tr>
                                                        <tr>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>Heikkinen, Miss. Laina</td>
                                                            <td>female</td>
                                                            <td>26.0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>STON/O2. 3101282</td>
                                                            <td>7.9250</td>
                                                            <td>NaN</td>
                                                            <td>S</td>
                                                        </tr>
                                                        <tr>
                                                            <td>4</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
                                                            <td>female</td>
                                                            <td>35.0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>113803</td>
                                                            <td>53.1000</td>
                                                            <td>C123</td>
                                                            <td>S</td>
                                                        </tr>
                                                        <tr>
                                                            <td>5</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Allen, Mr. William Henry</td>
                                                            <td>male</td>
                                                            <td>35.0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>373450</td>
                                                            <td>8.0500</td>
                                                            <td>NaN</td>
                                                            <td>S</td>
                                                        </tr>
                                                        <tr>
                                                            <td>6</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Moran, Mr. James</td>
                                                            <td>male</td>
                                                            <td>NaN</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>330877</td>
                                                            <td>8.4583</td>
                                                            <td>NaN</td>
                                                            <td>Q</td>
                                                        </tr>
                                                        <tr>
                                                            <td>7</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>McCarthy, Mr. Timothy J</td>
                                                            <td>male</td>
                                                            <td>54.0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>17463</td>
                                                            <td>51.8625</td>
                                                            <td>E46</td>
                                                            <td>S</td>
                                                        </tr>
                                                        <tr>
                                                            <td>8</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Palsson, Master. Gosta Leonard</td>
                                                            <td>male</td>
                                                            <td>2.0</td>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>349909</td>
                                                            <td>21.0750</td>
                                                            <td>NaN</td>
                                                            <td>S</td>
                                                        </tr>
                                                        <tr>
                                                            <td>9</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
                                                            <td>female</td>
                                                            <td>27.0</td>
                                                            <td>0</td>
                                                            <td>2</td>
                                                            <td>347742</td>
                                                            <td>11.1333</td>
                                                            <td>NaN</td>
                                                            <td>S</td>
                                                        </tr>
                                                        <tr>
                                                            <td>10</td>
                                                            <td>1</td>
                                                            <td>2</td>
                                                            <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
                                                            <td>female</td>
                                                            <td>14.0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>237736</td>
                                                            <td>30.0708</td>
                                                            <td>NaN</td>
                                                            <td>C</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">Most machine learning algorithms require numerical input and
                                        output variables. (This limitation may not
                                        be strictly enforced by the machine learning algorithms themselves, but is
                                        generally a more efficient
                                        implementation of the algorithms.)</p>
                                    <p className="caption">From this preview table we can see that we will need to convert
                                        several categorical features into numeric
                                        values (e.g. 'Embarked', 'Sex', 'Pclass'). We can also observe several features
                                        contain continuous data
                                        with a wide range of values (e.g. 'Age', 'Fare') that we'll want to transform
                                        into binned categorical data
                                        to equalize their importance. Furthermore, we can detect additional features
                                        that contain missing values
                                        (e.g. 'Cabin') which we will also need to need to deal with.</p>
                                    <p className="caption">Let's figure out which features contain missing data.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Calculate missing values for each feature</h4>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [7]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# calculate the sum of missing values for each column
total = train.isnull().sum().sort_values(ascending=False)

# calculate the percentage of missing values for each column
percent = round((train.isnull().sum() / train.isnull().count() * 100), 1).sort_values(ascending=False)

# create a new dataframe
missing_data = pd.concat([total, percent], axis=1, keys=['# Missing', '% Missing'])

# preview the dataframe
missing_data.head(12)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [7]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th></th>
                                                            <th># Missing</th>
                                                            <th>% Missing</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>Cabin</td>
                                                            <td>687</td>
                                                            <td>77.1</td>
                                                        </tr>
                                                        <tr>
                                                            <td>Age</td>
                                                            <td>177</td>
                                                            <td>19.9</td>
                                                        </tr>
                                                        <tr>
                                                            <td>Embarked</td>
                                                            <td>2</td>
                                                            <td>0.2</td>
                                                        </tr>
                                                        <tr>
                                                            <td>Fare</td>
                                                            <td>0</td>
                                                            <td>0.0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>Ticket</td>
                                                            <td>0</td>
                                                            <td>0.0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>Parch</td>
                                                            <td>0</td>
                                                            <td>0.0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>SibSp</td>
                                                            <td>0</td>
                                                            <td>0.0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>Sex</td>
                                                            <td>0</td>
                                                            <td>0.0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>Name</td>
                                                            <td>0</td>
                                                            <td>0.0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>Pclass</td>
                                                            <td>0</td>
                                                            <td>0.0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>Survived</td>
                                                            <td>0</td>
                                                            <td>0.0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>PassengerId</td>
                                                            <td>0</td>
                                                            <td>0.0</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">The 'Embarked' feature has 2 missing values, 0.2% of the total
                                        examples. We can drop those entries without having a significant impact on
                                        our model's accuracy to predict
                                        results.</p>
                                    <p className="caption">The 'Age' feature has 177 missing values, 19.9% of the total
                                        examples. We will transform this feature by binning the continues values
                                        into defined cohorts, including a
                                        'missing' cohort, of categorical values.</p>
                                    <p className="caption">The 'Cabin' feature has 687 missing values,
                                        <em><strong>77.7%</strong></em> of the total examples. We will need to take a
                                        closer look at this feature to determine
                                        if it's possible to use this
                                        feature.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h2>Feature selection</h2>
                                    <p className="caption">Let's review our feature set to understand how each feature
                                        impacts survival. The goal is to determine
                                        which features we should include in our machine learning model and which
                                        features we can drop. In some
                                        cases we'll need to preprocess the feature data to better understand its meaning
                                        before deciding how to
                                        proceed. (We'll also do more preprocessing before fitting the data to our
                                        model.)</p>
                                    <p className="caption">We'll generate some quick statistics and a plot simple bar chart
                                        to visualize the relationship between
                                        each feature and our target variable 'Survival'. We can use the <a
                                            href="https://seaborn.pydata.org/index.html">seaborn</a> library (a data
                                        visualization library based on
                                        <a href="https://matplotlib.org/">matplotlib</a> used for creating informative
                                        statistical graphics) to
                                        generate these charts.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Embarked</h4>
                                    <p className="caption">We'll start with the 'Embarked' feature because we want to drop
                                        the 2 rows with missing data before doing
                                        anything else.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [8]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# recreate the dataframe after dropping rows with null 'Embarked' values
train = train[pd.notnull(train['Embarked'])]

# preview the updated dataframe
train.head()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [8]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th>PassengerId</th>
                                                            <th>Survived</th>
                                                            <th>Pclass</th>
                                                            <th>Name</th>
                                                            <th>Sex</th>
                                                            <th>Age</th>
                                                            <th>SibSp</th>
                                                            <th>Parch</th>
                                                            <th>Ticket</th>
                                                            <th>Fare</th>
                                                            <th>Cabin</th>
                                                            <th>Embarked</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Braund, Mr. Owen Harris</td>
                                                            <td>male</td>
                                                            <td>22.0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>A/5 21171</td>
                                                            <td>7.2500</td>
                                                            <td>NaN</td>
                                                            <td>S</td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
                                                            <td>female</td>
                                                            <td>38.0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>PC 17599</td>
                                                            <td>71.2833</td>
                                                            <td>C85</td>
                                                            <td>C</td>
                                                        </tr>
                                                        <tr>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>Heikkinen, Miss. Laina</td>
                                                            <td>female</td>
                                                            <td>26.0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>STON/O2. 3101282</td>
                                                            <td>7.9250</td>
                                                            <td>NaN</td>
                                                            <td>S</td>
                                                        </tr>
                                                        <tr>
                                                            <td>4</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
                                                            <td>female</td>
                                                            <td>35.0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>113803</td>
                                                            <td>53.1000</td>
                                                            <td>C123</td>
                                                            <td>S</td>
                                                        </tr>
                                                        <tr>
                                                            <td>5</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Allen, Mr. William Henry</td>
                                                            <td>male</td>
                                                            <td>35.0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>373450</td>
                                                            <td>8.0500</td>
                                                            <td>NaN</td>
                                                            <td>S</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">Previewing the dataframe doesn't verify that the rows with
                                        missing 'Embarked' values have been dropped,
                                        but we can do a quick calculations to confirm.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [9]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# calculate the sum of missing values in the 'Embarked' column
total = train['Embarked'].isnull().sum()

# count the number of rows in the 'Embarked' column
count = train['Embarked'].isnull().count()

# calculate the percentage of missing values for the 'Embarked' column
percent = round((total / count * 100), 1)

# create a new dataframe
missing_data = pd.DataFrame([[total, percent]], columns=['# Missing', '% Missing'], index =['Embarked'])

# preview the dataframe
missing_data.head()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [9]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th></th>
                                                            <th># Missing</th>
                                                            <th>% Missing</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>Embarked</td>
                                                            <td>0</td>
                                                            <td>0.0</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [10]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# describe the data
train['Embarked'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [10]:</div>
                                        <div>
<pre><code className="language-bash">
{`count     889
unique      3
top         S
freq      644
Name: Embarked, dtype: object`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [11]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# count the unique values
train['Embarked'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [11]:</div>
                                        <div>
<pre><code className="language-bash">
{`S    644
C    168
Q     77
Name: Embarked, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [12]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# plot the feature data as a bar chart
sns.barplot(x='Embarked', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [12]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12a537898>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">Before moving on to the next feature let's convert the 'Embarked'
                                        feature values to lowercase</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [13]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# convert strings in the series to lowercase
train['Embarked'] = train['Embarked'].str.lower()

# count the unique values
train['Embarked'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [13]:</div>
                                        <div>
<pre><code className="language-bash">
{`s    644
c    168
q     77
Name: Embarked, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Cabin</h4>
                                    <p className="caption">Next let's take a look at the 'Cabin' feature and see what to do
                                        about its missing values.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [14]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Cabin'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [14]:</div>
                                        <div>
<pre><code className="language-bash">
{`count             202
unique            146
top       C23 C25 C27
freq                4
Name: Cabin, dtype: object`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [15]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Cabin'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [15]:</div>
                                        <div>
<pre><code className="language-bash">
{`C23 C25 C27    4
G6             4
B96 B98        4
F33            3
D              3
    ..
E38            1
B3             1
D28            1
E46            1
E68            1
Name: Cabin, Length: 146, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [16]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='Cabin', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [16]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12c6dcf98>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">Given the number of missing values, number of unique values and
                                        the categorical nature of this feature we
                                        could be tempted to drop it from our dataset. However, if we take a closer look
                                        at the data we see that
                                        there is a letter in each cabin (e.g. "C123") that corresponds to a deck on the
                                        titanic. If we review a
                                        cutaway diagram of the tianic (see below) it would be interesting to see whether
                                        or not deck location has
                                        an impact on survival.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">
                                        <img alt="" className="cutaway" src={cutaway} />
                                    </p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">Let's extract the deck identifier from the 'Cabin' feature and
                                        use this value to create a new feature
                                        we'll call 'Deck'.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [17]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# create a new 'Deck' column in our dataframe using regex to extract the first letter from the 'Cabin' column
train['Deck'] = train.Cabin.str.extract('([a-zA-Z]+)', expand=False).str.lower()

# replace null values in the 'Deck' column
train['Deck'] = train['Deck'].fillna('unavailable')

# preview the updated dataframe
train.head()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [17]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th>PassengerId</th>
                                                            <th>Survived</th>
                                                            <th>Pclass</th>
                                                            <th>Name</th>
                                                            <th>Sex</th>
                                                            <th>Age</th>
                                                            <th>SibSp</th>
                                                            <th>Parch</th>
                                                            <th>Ticket</th>
                                                            <th>Fare</th>
                                                            <th>Cabin</th>
                                                            <th>Embarked</th>
                                                            <th>Deck</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Braund, Mr. Owen Harris</td>
                                                            <td>male</td>
                                                            <td>22.0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>A/5 21171</td>
                                                            <td>7.2500</td>
                                                            <td>NaN</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
                                                            <td>female</td>
                                                            <td>38.0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>PC 17599</td>
                                                            <td>71.2833</td>
                                                            <td>C85</td>
                                                            <td>c</td>
                                                            <td>c</td>
                                                        </tr>
                                                        <tr>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>Heikkinen, Miss. Laina</td>
                                                            <td>female</td>
                                                            <td>26.0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>STON/O2. 3101282</td>
                                                            <td>7.9250</td>
                                                            <td>NaN</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                        </tr>
                                                        <tr>
                                                            <td>4</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
                                                            <td>female</td>
                                                            <td>35.0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>113803</td>
                                                            <td>53.1000</td>
                                                            <td>C123</td>
                                                            <td>s</td>
                                                            <td>c</td>
                                                        </tr>
                                                        <tr>
                                                            <td>5</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Allen, Mr. William Henry</td>
                                                            <td>male</td>
                                                            <td>35.0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>373450</td>
                                                            <td>8.0500</td>
                                                            <td>NaN</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [18]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Deck'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [18]:</div>
                                        <div>
<pre><code className="language-bash">
{`count             889
unique              9
top       unavailable
freq              687
Name: Deck, dtype: object`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [19]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Deck'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [19]:</div>
                                        <div>
<pre><code className="language-bash">
{`unavailable    687
c               59
b               45
d               33
e               32
a               15
f               13
g                4
t                1
Name: Deck, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [20]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='Deck', y='Survived', data=train, order=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'unavailable'])`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [20]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12cc3e400>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">This looks much better and is something we can use. Let's go
                                        ahead and drop the old 'Cabin' feature from
                                        our dataset.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [21]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# drop the 'Cabin' column from the dataframe
train = train.drop(columns=['Cabin'])

# preview the updated dataframe
train.head()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [21]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th></th>
                                                            <th>PassengerId</th>
                                                            <th>Survived</th>
                                                            <th>Pclass</th>
                                                            <th>Name</th>
                                                            <th>Sex</th>
                                                            <th>Age</th>
                                                            <th>SibSp</th>
                                                            <th>Parch</th>
                                                            <th>Ticket</th>
                                                            <th>Fare</th>
                                                            <th>Embarked</th>
                                                            <th>Deck</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Braund, Mr. Owen Harris</td>
                                                            <td>male</td>
                                                            <td>22.0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>A/5 21171</td>
                                                            <td>7.2500</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                        </tr>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>2</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
                                                            <td>female</td>
                                                            <td>38.0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>PC 17599</td>
                                                            <td>71.2833</td>
                                                            <td>c</td>
                                                            <td>c</td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>Heikkinen, Miss. Laina</td>
                                                            <td>female</td>
                                                            <td>26.0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>STON/O2. 3101282</td>
                                                            <td>7.9250</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                        </tr>
                                                        <tr>
                                                            <td>3</td>
                                                            <td>4</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
                                                            <td>female</td>
                                                            <td>35.0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>113803</td>
                                                            <td>53.1000</td>
                                                            <td>s</td>
                                                            <td>c</td>
                                                        </tr>
                                                        <tr>
                                                            <td>4</td>
                                                            <td>5</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Allen, Mr. William Henry</td>
                                                            <td>male</td>
                                                            <td>35.0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>373450</td>
                                                            <td>8.0500</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Age</h4>
                                    <p className="caption">While we're dealing with missing values let's take a look at the
                                        'Age' feature next.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [22]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Age'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [22]:</div>
                                        <div>
<pre><code className="language-bash">
{`count    712.000000
mean      29.642093
std       14.492933
min        0.420000
25%       20.000000
50%       28.000000
75%       38.000000
max       80.000000
Name: Age, dtype: float64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [23]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Age'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [23]:</div>
                                        <div>
<pre><code className="language-bash">
{`24.00    30
22.00    27
18.00    26
28.00    25
30.00    25
..
55.50     1
70.50     1
66.00     1
23.50     1
0.42      1
Name: Age, Length: 88, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [24]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='Age', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [24]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12ccb0710>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">We see a similar issue with the 'Age' feature that we saw with
                                        the 'Cabin' feature; a significant number
                                        of missing values and unique values. In addition, as our bar chart illustrates,
                                        there is a wide range of
                                        values that are unequally distributed. We'll also need deal with the missing
                                        values. We can solve all
                                        these problems by binning the continuous values, including the missing values,
                                        from the 'Age' feature and
                                        create a new feature we'll call 'AgeCategory'</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [25]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# define bins and labels
bin_ranges = [-1, 0, 12, 20, 30, 45, 60, 80]
range_labels = ['missing', 'child', 'teenager', 'young_adult', 'adult', 'middle_age', 'senior']

# replace NaN values with negative number between 0 and -1
train['Age'] = train['Age'].fillna(-0.5)

# convert values from float to int
train['Age'] = train['Age'].astype(int)

# segment and sort values into labeled bins
train['AgeCategories'] = pd.cut(train['Age'], bin_ranges, labels=range_labels)

# preview the updated dataframe
train.head()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [25]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th></th>
                                                            <th>PassengerId</th>
                                                            <th>Survived</th>
                                                            <th>Pclass</th>
                                                            <th>Name</th>
                                                            <th>Sex</th>
                                                            <th>Age</th>
                                                            <th>SibSp</th>
                                                            <th>Parch</th>
                                                            <th>Ticket</th>
                                                            <th>Fare</th>
                                                            <th>Embarked</th>
                                                            <th>Deck</th>
                                                            <th>AgeCategories</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Braund, Mr. Owen Harris</td>
                                                            <td>male</td>
                                                            <td>22</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>A/5 21171</td>
                                                            <td>7.2500</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                        </tr>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>2</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
                                                            <td>female</td>
                                                            <td>38</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>PC 17599</td>
                                                            <td>71.2833</td>
                                                            <td>c</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>Heikkinen, Miss. Laina</td>
                                                            <td>female</td>
                                                            <td>26</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>STON/O2. 3101282</td>
                                                            <td>7.9250</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                        </tr>
                                                        <tr>
                                                            <td>3</td>
                                                            <td>4</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
                                                            <td>female</td>
                                                            <td>35</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>113803</td>
                                                            <td>53.1000</td>
                                                            <td>s</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                        </tr>
                                                        <tr>
                                                            <td>4</td>
                                                            <td>5</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Allen, Mr. William Henry</td>
                                                            <td>male</td>
                                                            <td>35</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>373450</td>
                                                            <td>8.0500</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>adult</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [26]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['AgeCategories'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [26]:</div>
                                        <div>
<pre><code className="language-bash">
{`count             889
unique              7
top       young_adult
freq              231
Name: AgeCategories, dtype: object`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [27]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['AgeCategories'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [27]:</div>
                                        <div>
<pre><code className="language-bash">
{`young_adult    231
adult          201
missing        184
teenager       111
middle_age      79
child           62
senior          21
Name: AgeCategories, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [28]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='AgeCategories', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [28]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12d002438>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYIAAAEHCAYAAACjh0HiAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAcf0lEQVR4nO3de5gdVZnv8e8vwQgChgNEYCAxUcNgBOTSwOHAcIcTdAwzEoSAlzhodDTiMwoMjD4RUEaFB84RCEhwEAZFbh45ETME5OpEgSQQSAIGYrgkkYzhKrcjNHnPH2t1urKzd++ddFfvdOr3eZ5+dlXtVave3pd6q9batUoRgZmZVdegdgdgZmbt5URgZlZxTgRmZhXnRGBmVnFOBGZmFbdJuwNYV9tuu22MHDmy3WGYmQ0oc+fOfS4ihtV7bsAlgpEjRzJnzpx2h2FmNqBIerrRc24aMjOrOCcCM7OKcyIwM6s4JwIzs4pzIjAzqzgnAjOzinMiMDOrOCcCM7OKG3AXlFljp59+OitWrGD77bfnvPPOa3c4ZjZAOBFsRFasWMHy5cvbHYaZDTBuGjIzqzgnAjOzinMiMDOrOCcCM7OKcyIwM6s4JwIzs4pzIjAzqzgnAjOzinMiMDOrOCcCM7OKKzURSBoraZGkxZLOaFDmE5IelbRQ0rVlxmNmZmsrbawhSYOBqcCRwDJgtqTpEfFoocxo4EzggIh4UdJ7yorHzMzqK/OMYF9gcUQsiYg3geuAY2rKfB6YGhEvAkTEn0qMx8zM6igzEewILC3ML8vLinYGdpY0S9J9ksbWq0jSJElzJM1ZuXJlSeGamVVTuzuLNwFGA4cAE4ArJG1VWygipkVER0R0DBs2rJ9DNDPbuJWZCJYDwwvzO+VlRcuA6RHxVkQ8CTxOSgxmZtZPykwEs4HRkkZJGgKcAEyvKXMz6WwASduSmoqWlBiTmZnVKC0RREQnMBmYCTwG3BARCyWdI2lcLjYTeF7So8BdwGkR8XxZMZmZ2dpKvVVlRMwAZtQsm1KYDuBr+c/MzNqg3Z3FZmbWZk4EZmYV50RgZlZxTgRmZhXnRGBmVnFOBGZmFedEYGZWcU4EZmYV50RgZlZxTgRmZhXnRGBmVnFOBGZmFedEYGZWcU4EZmYV50RgZlZxpd6PwMysP5x++umsWLGC7bffnvPOO6/d4Qw4TgQbgGfO2a1P6ul8YWtgEzpfeLpP6hwxZX7vgzLrBytWrGD58tpbolur3DRkZlZxTgRmZhXnRGBmVnFOBGZmFedEYGZWcU4EZmYVV2oikDRW0iJJiyWdUef5iZJWSpqX/z5XZjxmZra20q4jkDQYmAocCSwDZkuaHhGP1hS9PiImlxWHmZn1rMwzgn2BxRGxJCLeBK4Djilxe2Zmth7KTAQ7AksL88vyslrHSnpE0k2ShpcYj5mZ1dHuzuJfAiMjYnfgduDqeoUkTZI0R9KclStX9muAZmYbuzITwXKgeIS/U162WkQ8HxF/ybM/AvauV1FETIuIjojoGDZsWCnBmplVVZmDzs0GRksaRUoAJwAnFgtI2iEins2z44DHSozHzBrw6J3VVloiiIhOSZOBmcBg4MqIWCjpHGBOREwHTpE0DugEXgAmlhWPmTXm0TurrdRhqCNiBjCjZtmUwvSZwJllxmBmZj1rd2exmZm1mROBmVnFORGYmVWcE4GZWcX5nsW2wfBPGM3aw4nANhj+CaNZe7hpyMys4pwIzMwqzonAzKzinAjMzCrOicDMrOKcCMzMKs6JwMys4pwIzMwqzonAzKzinAjMzCrOicDMrOKcCMzMKs6Dzpn1AY+cagOZE4FZH2jXyKmXfP2XfVLPS8+9tvqxL+qcfMHHel2H9R83DZmZVZzPCMysbc795Pg+qeeFP72cHlc82yd1fuMnN/W6joHEZwRmZhVXaiKQNFbSIkmLJZ3RQ7ljJYWkjjLjMTOztZWWCCQNBqYCRwNjgAmSxtQptyXwVeD+smIxM7PGyjwj2BdYHBFLIuJN4DrgmDrlvg18H/h/JcZiZmYNlJkIdgSWFuaX5WWrSdoLGB4RvyoxDjMz60HbfjUkaRBwITCxhbKTgEkAI0aMKDewAWzbTVcBnfnRzKw1PSYCSa8A0ej5iHh3D6svB4YX5nfKy7psCewK3C0JYHtguqRxETGnZjvTgGkAHR0dDeOpulN3f6ndIZjZANRjIoiILQEkfRt4FrgGEHASsEOTumcDoyWNIiWAE4ATC3W/DGzbNS/pbuDU2iRgZmblarVpaFxEfLgwf5mkh4EpjVaIiE5Jk4GZwGDgyohYKOkcYE5ETF/vqEvi8WLMrIpaTQSvSTqJ9MufACYArzVbKSJmADNqltVNHhFxSIuxlKZd48WYmbVTq4ngROAH+S+AWRSaeczMbP21uzWipUQQEU9R/xoAMzPrpXa3RrSUCCTtDFwGbBcRu0randRv8J1So7MB4YCLD+iTeoa8NIRBDGLpS0v7pM5ZX5nVB1GZbfxavaDsCuBM4C2AiHiE9CsgMzMb4FpNBO+KiAdqlnX2dTBmZtb/Wk0Ez0l6P/niMknjSdcVmJnZANfqr4a+TLqydxdJy4EnSReVmQ1o9xx0cJ/U88Ymg0HijWXL+qzOg++9p0/qMWum1UTwdEQcIWlzYFBEvFJmUGbWvzYf8u41Hq1aWk0ET0q6FbgeuLPEeMysDQ54/8fbHYK1Uat9BLsAvyY1ET0p6RJJB5YXlpmZ9ZeWEkFEvB4RN0TEx4E9gXcDbsA0M9sItHxjGkkHS7oUmAtsCnyitKjMzKzftHpl8VPAQ8ANwGkR0XTAOTMzGxha7SzePSL+XGokZmbWFs3uUHZ6RJwHnCtprTuDRcQppUVmZmb9otkZwWP50XcNMzPbSDW7VeUv8+T8iHiwH+IxM7N+1uqvhi6Q9Jikb0vatdSIzMysX7V6HcGhwKHASuBySfMlfbPUyMzMrF+0fB1BRKyIiIuALwLz6OHG9WZmNnC0eh3BB4HjgWOB50ljDn29xLjWyd6n/Xuf1LPlc68wGHjmuVf6pM6553+690GZmZWs1esIrgSuA/5nRPyxxHjMzKyfNU0EkgYDT0bED/ohHjOzdbbp4EFrPNq6afqqRcTbwHBJQ9a1ckljJS2StFjSGXWe/2LueJ4n6T8ljVnXbZiZ7bnNluz/nqHsuc2W7Q5lQGr5fgTALEnTgdXjDEXEhY1WyGcSU4EjgWXAbEnTI+LRQrFrI+KHufw44EJg7Lr9C2Zm1hutJoI/5L9BQKspd19gcUQsAZB0HXAMsDoR1IxftDn5nshmZtZ/WkoEEXH2etS9I7C0ML8M2K+2kKQvA18DhgCHrcd2zMysF1r9+ehd1Dlaj4he77gjYiowVdKJwDeBz9TZ/iRgEsCIESN6u0mzPrdVxBqPZgNJq01DpxamNyVdT9DZZJ3lwPDC/E55WSPXAZfVeyIipgHTADo6OvxNsw3OJ99e1e4QzNZbq01Dc2sWzZL0QJPVZgOjJY0iJYATgBOLBSSNjogn8uxHgScwM7N+1WrT0NaF2UFABzC0p3UiolPSZGAmMBi4MiIWSjoHmBMR04HJko4A3gJepE6zkJmZlavVpqG5dPcRdAJPASc3WykiZgAzapZNKUx/tcXtWwXEu4JVrCLe5dY/s/7U7A5l+wBLI2JUnv8MqX/gKQo/AzXrC28d8Fa7QzCrpGZXFl8OvAkg6SDgu8DVwMvkzlszMxvYmjUNDY6IF/L08cC0iPg58HNJ88oNzczM+kOzM4LBkrqSxeHAnYXnWu1fMDOzDViznfnPgHskPQe8AfwGQNIHSM1DZmY2wDW7ef25ku4AdgBui1h92eQg4CtlB2dmZuVr2rwTEffVWfZ4OeGYmVl/810czMwqzonAzKzinAjMzCrOPwEtWDVk8zUezcyqwImg4LXRR7U7BDOzfuemITOzinMiMDOrOCcCM7OKcyIwM6s4JwIzs4pzIjAzqzgnAjOzinMiMDOrOCcCM7OKcyIwM6s4JwIzs4orNRFIGitpkaTFks6o8/zXJD0q6RFJd0h6b5nxmJnZ2kpLBJIGA1OBo4ExwARJY2qKPQR0RMTuwE3AeWXFY2Zm9ZV5RrAvsDgilkTEm8B1wDHFAhFxV0S8nmfvA3YqMR4zM6ujzESwI7C0ML8sL2vkZOA/SozHzMzq2CDuRyDpk0AHcHCD5ycBkwBGjBjRj5GZmW38yjwjWA4ML8zvlJetQdIRwDeAcRHxl3oVRcS0iOiIiI5hw4aVEqyZWVWVmQhmA6MljZI0BDgBmF4sIGlP4HJSEvhTibGYmVkDpTUNRUSnpMnATGAwcGVELJR0DjAnIqYD5wNbADdKAngmIsaVFZOZWV967Nw7+6SeN194Y/VjX9T5wW8ctk7lS+0jiIgZwIyaZVMK00eUuX0zM2vOVxabmVWcE4GZWcU5EZiZVZwTgZlZxTkRmJlVnBOBmVnFORGYmVWcE4GZWcU5EZiZVZwTgZlZxTkRmJlVnBOBmVnFORGYmVWcE4GZWcU5EZiZVZwTgZlZxTkRmJlVnBOBmVnFORGYmVWcE4GZWcU5EZiZVZwTgZlZxTkRmJlVnBOBmVnFlZoIJI2VtEjSYkln1Hn+IEkPSuqUNL7MWMzMrL7SEoGkwcBU4GhgDDBB0piaYs8AE4Fry4rDzMx6tkmJde8LLI6IJQCSrgOOAR7tKhART+XnVpUYh5mZ9aDMpqEdgaWF+WV52TqTNEnSHElzVq5c2SfBmZlZMiA6iyNiWkR0RETHsGHD2h2OmdlGpcxEsBwYXpjfKS8zM7MNSJmJYDYwWtIoSUOAE4DpJW7PzMzWQ2mJICI6gcnATOAx4IaIWCjpHEnjACTtI2kZcBxwuaSFZcVjZmb1lfmrISJiBjCjZtmUwvRsUpORmZm1yYDoLDYzs/I4EZiZVZwTgZlZxTkRmJlVnBOBmVnFORGYmVWcE4GZWcU5EZiZVZwTgZlZxTkRmJlVnBOBmVnFORGYmVVcqYPOmZlZc9tsOnSNx/7mRGBm1maT9zyxrdt305CZWcU5EZiZVZwTgZlZxTkRmJlVnBOBmVnFORGYmVWcE4GZWcU5EZiZVZwTgZlZxZWaCCSNlbRI0mJJZ9R5/p2Srs/P3y9pZJnxmJnZ2kpLBJIGA1OBo4ExwARJY2qKnQy8GBEfAP4X8P2y4jEzs/rKPCPYF1gcEUsi4k3gOuCYmjLHAFfn6ZuAwyWpxJjMzKyGIqKciqXxwNiI+Fye/xSwX0RMLpRZkMssy/N/yGWeq6lrEjApz/41sKiUoJNtgeealtpwOf72Gcixg+Nvt7Ljf29EDKv3xIAYfTQipgHT+mNbkuZEREd/bKsMjr99BnLs4PjbrZ3xl9k0tBwYXpjfKS+rW0bSJsBQ4PkSYzIzsxplJoLZwGhJoyQNAU4ApteUmQ58Jk+PB+6MstqqzMysrtKahiKiU9JkYCYwGLgyIhZKOgeYExHTgX8DrpG0GHiBlCzarV+aoErk+NtnIMcOjr/d2hZ/aZ3FZmY2MPjKYjOzinMiMDOruEolAknj6g110cJ6vy0jnvUl6ap8nUbt8r+SdFOePkTSLQ3Wf0rStr3Y/laSvrS+69uGQ9JESZc0KTMyX/ODpD0kfaR/oqsmSR2SLurPbVYqEUTE9Ij43nqs9z/KiKevRcQfI2KtBFGCrYABkwjycCcbtAG0Q94D6LPt9nRwJunVBstXHwhJulvSgL12oJ6ImBMRp7RaPv/0vlc2mkSQvyS/zx+SxyX9VNIRkmZJekLSvsUvm6TjJC2Q9LCke/OyD0l6QNI8SY9IGp2Xv5ofD8kfvJvytn7aNSSGpI/kZXMlXdToaHw9/7dP53gelnRNXnyQpN9KWlL4UqzeUdSsv42k2yQtlPQjoLfDeHwPeH9+nc6XdJqk2TnGswvb/WTh9by8a4cs6VVJ5+b/5z5J2+XlH8uDDz4k6deF5cMk3d4Vv6Snu85ommzjAkkPA/v38v/dELW0Q5Z0c/5MLlS6Qh9Jn83fkQeAAwpl1zjTrN0RK/0M/Bzg+Px6H9/bf2J9D842VJI2l/Sr/NleIOl4SXtLuie/DzMl7ZDL3i3p+/nz+7ikv8nLV5/NS9o6v4eP5O/K7nn5WZKukTQLuKZhQK2KiI3iDxgJdAK7kRLcXOBK0k7vGOBmYCJwSS4/H9gxT2+VHy8GTsrTQ4DN8vSr+fEQ4GXSxXGDgN8BBwKbAkuBUbncz4Bb+uj/+hDwOLBtnt8auAq4MccwhjSmU9drsKAQ6y15+iJgSp7+KBBd9fXite7azlGkn70px3MLcBDwQeCXwDtyuUuBT+fpAD6Wp88Dvpmn/xvdv2T7HHBBnr4EODNPj+2Kv9E2SDurAD6Rl58LfBU4H1iQ3/vja1+nwrYm5umngLOBB/M6u+Tlw4DbgYXAj4Cne3o9SZ+9ubn8pMLyz+b39gHgCro/m1cB4wvlXi2+7qTP5jPASmBe1//SYNtb58fN8ro75nWH5XpmtbrdPD2xq3yLn5Pf53ofB34KHJG3+QRpPLKJhe2PIn2n5gPfKWxf+X1ZBPwamNEVJ3A30FH4LP4uv183Alv0ENsU0rVOC8if37x8H+CR/LqeX/i/B+f52fn5LzSo91jgisL8UOC3wLA8fzzpp/RdsXd9xj8C/LrOd/di4Ft5+jBgXp4+i/SZ2qwv9jMbzRlB9mREzI+IVaQv3R2RXrX5pA9l0SzgKkmfJ73JkD5E/yLpn0njcrxRZxsPRMSyvI15ud5dgCUR8WQu87M+/J8OA26MPP5SRLyQl98cEasi4lFguyZ1HAT8JK//K+DFPozvqPz3EOkLuAswGjgc2BuYLWlenn9fXudNUsKA9GEemad3AmZKmg+cRkqCkJLtdTn+WwvxN9rGlfn5n0saRLo+ZRnpKPrDpJ3R+V1HZk08FxF7AZcBp+Zl3yJd/Pgh0mCJI5rU8Q8RsTfQAZySz9B2ICWZA/L/Vzsyb0ORBnGcAlwfEXtExPU9FD8lnxXdR7qK/1PA3RGxMtfT07p94QPABaTPxS7AiaT/91TgX2rK/gC4LCJ2A54tLP970hhjY0iJfq2m2nyG+E3giPx+zQG+1kNcl0TEPhGxKylJ/m1e/mPSTn4P4O1C+ZOBlyNiH1Ky+LykUXXqnQ8cmY/0/4b0mu8K3J4/o98kfc67/J/8WPweFB1IPuKPiDuBbSS9Oz83vcE+ap0NiLGG1sFfCtOrCvOrqPlfI+KLkvYjHSHPlbR3RFwr6f68bIakL+QXv9E23q6ttx8V42jniK0CvhsRl6+xUPoKcHVEnFlnnbdygoY1X8OLgQsjYrqkQ0hHPc22XXcbkt4GdiclyYdIX6ifRcTbwH9Juof0hf5zk20Uv6gfz9MHknZORMStkpol1lMk/X2eHk5KlNuTd8g53uuBnZvUs07ya3gEsH9EvC7pbtIReqOk00luLs4JdEgfhPFkRMzPda4+OMvJfmRN2QNIR9SQdn5dw9IfRPd790dJtd9JgP9O+r9m5dbaIaQDu0YOlXQ68C7SWfZCSb8BtoyIrvWupTtBHAXsXmg6G0p6H58s1ElEPC5pL9IR/neAO4GFEdGoebLre7w++5LX1rF8QxvbGUHLJL0/Iu6PiCmkU+zhkt5HOrK/CPi/pB1JKxYB71P3jXV63XZacCdwnKRtctxbr0cd95KOxJB0NKkJpjdeAbbM0zOBf5C0Ra5/R0nvAe4AxufprrbO9zapdyjd41F9prB8FvCJXM9Rhfh72kYnqdnhs3SfIdSzeueXbVrzfG++qLU74w+TklLtNhrG1Msd8lDS/T5el7QLaWe5GXBwPit5B3BcofxTpDMsgHHAO+rUWXzvW9HywVm2vle4Crg9nyHtERFjIuLkugWlTUnNiOPz2ccVNH9PBHylUP+oiLitTt1/BbweET8hNSXtBwyTtH9+/h2SPlS7Xg9+A5yU1z2EdIba7OBlnVU2EZCaBuYrda7+FniYtLNZkE/hdgX+vZWK8unZl4BbJc0lfVle7osgI2IhqY37nnyKf+F6VHM2qXN5Iemo9plexvQ86chrAXAk6cjpd/ko7ybSUdWjpNPg2yQ9QmpTb9YUcxZwY34Ni8Pxng0clbd3HLACeKXJNjpJ/Qn7kJLVb0idnIMlDSMdZT5Aat8fo3S3vK1IzUvNNEpM9dTbGQPcT/k75FuBTSQ9Rurgv4/U5HIW6Wh5FvBYofwVOaauDvZ6R5x3kV6vPuksrjGL7mFmTiosv5fu924H4NA6694HHCDpA7C607bRGVbXTv+5fAAzHiAiXgJeyS0FsOaQNzOBf8zvFZJ2lrR5nbp3Ax7I+5BvkZrwxgPfz6/rPOo0bfXgLGDv/Pn+HmseIPWdvuho8F9A7pgiHTlcCvxTu2PaWP6AdwKb5On9yR1mLaz3Q+B7hfdlrc7i/Nx5pM7L20hNQRPz8qfo7qTvIDXlAHSd8Swg7TyfBd7ZQ+z/Qdrh3kzqIDwkP1fsLJ5Gd6fpdqQd28Ok5pF6nbZbkzoue+wsbvP7tjrePH8V3Z28I/PrN5F16yy+ncadxYfR3Zn7CDCuh9i+A/yBlHx+DJyVl+9Hd2fxD4BZefkg4F9zbAtICXFou1/jvvrzWEN9RNI/kbL1ENLp/+cj4vX2RrVxUPoZ7w2kL+ObwJciYnaTdQaROq+Pi4gn+jiedwJvRxpYcX9SB+cefbkNaw9JW0RE18/FzwB2iIivtjms0jkR2EZH6d7YtwC/iIivl1D/OicmGxhyc9eZpP6Lp0lnhyvbG1X5nAjM+kDuzL+jzlOHR+pTsTaQ9AtSk1PRP0fEzHbEs6FyIjAzq7gq/2rIzMxwIjAzqzwnAhvwJP2dpMi/0+9NPacqDRw4T2kQvU83KT8xX0BUKhWGFzcrgxOBbQwmAP+ZH9eLpC+SLo7bN/8U9HCaD90xESg1EUjaJPpveHGrKCcCG9DylaEHkgYFOyEvGyTp0nx0f7ukGeoeqrvukMCkAdD+MfLl+xHx54i4Oq8zJZ8hLJA0Tcl40kVmP81nEJs1qlvSPkrDCHcN2911T4FNJf04X+H+kKRD8/KJkqYrjalzh9a8D8HgXEfXsN9fyMt3kHRv3sYC5SGNzVrhRGAD3THArRHxOPC8pL1Jw2iMJA1C9iny/Qjy8AAXk65K3Zs0BtG5SqM5bhkRSxpsY62RKiPiJtIIlyflM4jOenXn9RuNaPllICKNdzMBuFppHByAvXJdB9fE0mgUzBOBmXkbHyZdGWvWko1t9FGrngmkoQAgDVU9gfS5vjHSUOErJN2Vn/9ruocEhjT8+LM0d6hqRqok3QehqG7defyiRiNaHkhKHkTE7yU9TfcIpLdH95DjRY1GwZwNXJmT3c0R4URgLXMisAFLaSTWw4DdJAVp5xvALxqtQoMhgZXuaPa+2rMCdY9U2RERSyWdRf2RKuvWnRPB+mg0xHDXKJhrXRAl6SDSEOpXSbowIloaNNHMTUM2kI0HromI90bEyIgYThof/gXg2NxXsB3pjk+QBi1rNCTwd4GpuZkISVvkXw3VHakyK44AWrfu6HlEy+IQwzuTbnCzqMn/XHcUTKXht/8rIq4g3TVtryb1mK3mMwIbyCbQffOSLj8n3cJyGfAo6RaiD5La1d/MTSoXSRpK+vz/b1JTz2XAFqS7nb0FvEW6jeBLkq4gjTi5gtQE0+Uq4IeS3iD1QzSq+2TgCkmrgHvoHqL8UuAypeG7O0nj2vwlNy018iNS/8eDSgVXAn9HSnan5dhfJd3Jy6wlHmLCNkrKo0jmMYAeAA6IiBXtjCVPV2ZESxs4fEZgG6tbcvv8EODb7UoC2UclrTGiZRtjMVuLzwjMzCrOncVmZhXnRGBmVnFOBGZmFedEYGZWcU4EZmYV9/8BLF0eHMcAYTYAAAAASUVORK5CYII=" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">This is a big improvement. It might be nice if our bins were more
                                        equal, but I think the age ranges we
                                        defined for our bins make sense in the context of a person's age. Now that we
                                        have our new categorical
                                        feature we can drop the existing 'Age' feature from our dataset.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [29]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# drop the 'Age' column from the dataframe
train = train.drop(columns=['Age'])

# preview the updated dataframe
train.head()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [29]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th></th>
                                                            <th>PassengerId</th>
                                                            <th>Survived</th>
                                                            <th>Pclass</th>
                                                            <th>Name</th>
                                                            <th>Sex</th>
                                                            <th>SibSp</th>
                                                            <th>Parch</th>
                                                            <th>Ticket</th>
                                                            <th>Fare</th>
                                                            <th>Embarked</th>
                                                            <th>Deck</th>
                                                            <th>AgeCategories</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Braund, Mr. Owen Harris</td>
                                                            <td>male</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>A/5 21171</td>
                                                            <td>7.2500</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                        </tr>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>2</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
                                                            <td>female</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>PC 17599</td>
                                                            <td>71.2833</td>
                                                            <td>c</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>Heikkinen, Miss. Laina</td>
                                                            <td>female</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>STON/O2. 3101282</td>
                                                            <td>7.9250</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                        </tr>
                                                        <tr>
                                                            <td>3</td>
                                                            <td>4</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
                                                            <td>female</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>113803</td>
                                                            <td>53.1000</td>
                                                            <td>s</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                        </tr>
                                                        <tr>
                                                            <td>4</td>
                                                            <td>5</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Allen, Mr. William Henry</td>
                                                            <td>male</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>373450</td>
                                                            <td>8.0500</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>adult</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Fare</h4>
                                    <p className="caption">While we're converting continuous values into binned categorical
                                        values we should go ahead and look at
                                        the 'Fare' feature next.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [30]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Fare'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [30]:</div>
                                        <div>
<pre><code className="language-bash">
{`count    889.000000
mean      32.096681
std       49.697504
min        0.000000
25%        7.895800
50%       14.454200
75%       31.000000
max      512.329200
Name: Fare, dtype: float64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [31]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Fare'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [31]:</div>
                                        <div>
<pre><code className="language-bash">
{`8.0500     43
13.0000    42
7.8958     38
7.7500     34
26.0000    31
        ..
7.8000      1
13.8583     1
7.6292      1
15.0500     1
8.6833      1
Name: Fare, Length: 247, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [32]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='Fare', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [32]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12d11efd0>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">The bad news is that it looks like our 'Fare' data has some
                                        similar issues that our 'Age' data had; a
                                        wide range of values that are unequally distributed and a significant number of
                                        unique values. The good
                                        news is that because the 'Fare' data is similar to the 'Age' data we can give
                                        this feature a similar
                                        transformation without too much effort.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [33]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# define bins and labels
bin_ranges = [-1, 0, 7, 14, 35, 70, 140, 525]
range_labels = ['Missing', '0', '1', '2', '3', '4', '5']

# replace NaN values with negative number between 0 and -1
train['Fare'] = train['Fare'].fillna(-0.5)

# convert values from float to int
train['Fare'] = train['Fare'].astype(int)

# segment and sort values into labeled bins
train['FareCategories'] = pd.cut(train['Fare'], bin_ranges, labels=range_labels)

# preview the updated dataframe
train.head()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [33]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th></th>
                                                            <th>PassengerId</th>
                                                            <th>Survived</th>
                                                            <th>Pclass</th>
                                                            <th>Name</th>
                                                            <th>Sex</th>
                                                            <th>SibSp</th>
                                                            <th>Parch</th>
                                                            <th>Ticket</th>
                                                            <th>Fare</th>
                                                            <th>Embarked</th>
                                                            <th>Deck</th>
                                                            <th>AgeCategories</th>
                                                            <th>FareCategories</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Braund, Mr. Owen Harris</td>
                                                            <td>male</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>A/5 21171</td>
                                                            <td>7</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>2</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
                                                            <td>female</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>PC 17599</td>
                                                            <td>71</td>
                                                            <td>c</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>4</td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>Heikkinen, Miss. Laina</td>
                                                            <td>female</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>STON/O2. 3101282</td>
                                                            <td>7</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>3</td>
                                                            <td>4</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
                                                            <td>female</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>113803</td>
                                                            <td>53</td>
                                                            <td>s</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>3</td>
                                                        </tr>
                                                        <tr>
                                                            <td>4</td>
                                                            <td>5</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Allen, Mr. William Henry</td>
                                                            <td>male</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>373450</td>
                                                            <td>8</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>adult</td>
                                                            <td>1</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [34]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['FareCategories'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [34]:</div>
                                        <div>
<pre><code className="language-bash">
{`count     889
unique      7
top         2
freq      240
Name: FareCategories, dtype: object`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [35]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['FareCategories'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [35]:</div>
                                        <div>
<pre><code className="language-bash">
{`2          240
0          226
1          216
3           89
4           72
5           31
Missing     15
Name: FareCategories, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [36]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='FareCategories', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [36]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12d999898>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">Another big improvement, and another situation where it might be
                                        nice if our bins were more equal, but
                                        again I think the ranges we defined for our bins make sense given the context.
                                        Let's drop the existing
                                        'Fare' feature from our dataset.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [37]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# drop the 'Fare' column from the dataframe
train = train.drop(columns=['Fare'])

# preview the updated dataframe
train.head()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [37]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th></th>
                                                            <th>PassengerId</th>
                                                            <th>Survived</th>
                                                            <th>Pclass</th>
                                                            <th>Name</th>
                                                            <th>Sex</th>
                                                            <th>SibSp</th>
                                                            <th>Parch</th>
                                                            <th>Ticket</th>
                                                            <th>Embarked</th>
                                                            <th>Deck</th>
                                                            <th>AgeCategories</th>
                                                            <th>FareCategories</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Braund, Mr. Owen Harris</td>
                                                            <td>male</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>A/5 21171</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>2</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
                                                            <td>female</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>PC 17599</td>
                                                            <td>c</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>4</td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>Heikkinen, Miss. Laina</td>
                                                            <td>female</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>STON/O2. 3101282</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>3</td>
                                                            <td>4</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
                                                            <td>female</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>113803</td>
                                                            <td>s</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>3</td>
                                                        </tr>
                                                        <tr>
                                                            <td>4</td>
                                                            <td>5</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Allen, Mr. William Henry</td>
                                                            <td>male</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>373450</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>adult</td>
                                                            <td>1</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Name</h4>
                                    <p className="caption">Let's continue working with the features that need some extra
                                        work and take a look at the 'Name' feature
                                        next.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [38]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Name'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [38]:</div>
                                        <div>
<pre><code className="language-bash">
{`count                      889
unique                     889
top       Naidenoff, Mr. Penko
freq                         1
Name: Name, dtype: object`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [39]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Name'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [39]:</div>
                                        <div>
<pre><code className="language-bash">
{`Naidenoff, Mr. Penko                           1
Lester, Mr. James                              1
Lemore, Mrs. (Amelia Milley)                   1
Emanuel, Miss. Virginia Ethel                  1
Levy, Mr. Rene Jacques                         1
                                        ..
Collyer, Mrs. Harvey (Charlotte Annie Tate)    1
Panula, Mr. Ernesti Arvid                      1
Newell, Miss. Marjorie                         1
Jussila, Mr. Eiriik                            1
Vande Velde, Mr. Johannes Joseph               1
Name: Name, Length: 889, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [40]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='Name', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [40]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12da9aeb8>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">Given that every name in our dataset is unique it's tempting to
                                        conclude that we should just drop this
                                        feature from our dataset, however it looks like every name includes a title.
                                        Let's see what the data looks
                                        like if we isolate that substring and create a new feature called 'Title'.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [41]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# create a new 'Title' column in our dataframe using regex to extract the title from the 'Name' column
train['Title'] = train.Name.str.extract('([A-Za-z]+)\.', expand=False).str.lower()

# preview the updated dataframe
train.head()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [41]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th></th>
                                                            <th>PassengerId</th>
                                                            <th>Survived</th>
                                                            <th>Pclass</th>
                                                            <th>Name</th>
                                                            <th>Sex</th>
                                                            <th>SibSp</th>
                                                            <th>Parch</th>
                                                            <th>Ticket</th>
                                                            <th>Embarked</th>
                                                            <th>Deck</th>
                                                            <th>AgeCategories</th>
                                                            <th>FareCategories</th>
                                                            <th>Title</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Braund, Mr. Owen Harris</td>
                                                            <td>male</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>A/5 21171</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                            <td>mr</td>
                                                        </tr>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>2</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
                                                            <td>female</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>PC 17599</td>
                                                            <td>c</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>4</td>
                                                            <td>mrs</td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>Heikkinen, Miss. Laina</td>
                                                            <td>female</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>STON/O2. 3101282</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                            <td>miss</td>
                                                        </tr>
                                                        <tr>
                                                            <td>3</td>
                                                            <td>4</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
                                                            <td>female</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>113803</td>
                                                            <td>s</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>3</td>
                                                            <td>mrs</td>
                                                        </tr>
                                                        <tr>
                                                            <td>4</td>
                                                            <td>5</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>Allen, Mr. William Henry</td>
                                                            <td>male</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>373450</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>adult</td>
                                                            <td>1</td>
                                                            <td>mr</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [42]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Title'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [42]:</div>
                                        <div>
<pre><code className="language-bash">
{`count     889
unique     17
top        mr
freq      517
Name: Title, dtype: object`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [43]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Title'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [43]:</div>
                                        <div>
<pre><code className="language-bash">
{`mr          517
miss        181
mrs         124
master       40
dr            7
rev           6
mlle          2
major         2
col           2
mme           1
sir           1
countess      1
jonkheer      1
capt          1
lady          1
don           1
ms            1
Name: Title, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [44]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='Title', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [44]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12dba8c18>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">This is looking much better, but since 'Mr', 'Miss', 'Mrs' and
                                        'Master' comprise a majority of the values
                                        in this feature we can reduce the others into an 'Other' category.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [45]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# replace values if matched with a value in a list of defined values to be replaced
train['Title'] = train['Title'].replace(['dr', 'rev', 'mlle', 'major', 'col', 'don', 'lady', 'sir', 'capt', 'ms', 'jonkheer', 'mme', 'countess'], 'other')

# count the unique values
train['Title'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [45]:</div>
                                        <div>
<pre><code className="language-bash">
{`mr        517
miss      181
mrs       124
master     40
other      27
Name: Title, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [46]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='Title', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [46]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12fa81710>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">This looks much better. We can now drop the existing 'Name'
                                        feature from our dataset.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [47]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# drop the 'Name' column from the dataframe
train = train.drop(columns=['Name'])

# preview the updated dataframe
train.head()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [47]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th></th>
                                                            <th>PassengerId</th>
                                                            <th>Survived</th>
                                                            <th>Pclass</th>
                                                            <th>Sex</th>
                                                            <th>SibSp</th>
                                                            <th>Parch</th>
                                                            <th>Ticket</th>
                                                            <th>Embarked</th>
                                                            <th>Deck</th>
                                                            <th>AgeCategories</th>
                                                            <th>FareCategories</th>
                                                            <th>Title</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>male</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>A/5 21171</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                            <td>mr</td>
                                                        </tr>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>2</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>female</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>PC 17599</td>
                                                            <td>c</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>4</td>
                                                            <td>mrs</td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>female</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>STON/O2. 3101282</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                            <td>miss</td>
                                                        </tr>
                                                        <tr>
                                                            <td>3</td>
                                                            <td>4</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>female</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>113803</td>
                                                            <td>s</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>3</td>
                                                            <td>mrs</td>
                                                        </tr>
                                                        <tr>
                                                            <td>4</td>
                                                            <td>5</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>male</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>373450</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>adult</td>
                                                            <td>1</td>
                                                            <td>mr</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>SibSp and Parch</h4>
                                    <p className="caption">We still have two more features that we'll need to manipulate.
                                        We'll look at the 'SibSp' and 'Parch'
                                        features next.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [48]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['SibSp'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [48]:</div>
                                        <div>
<pre><code className="language-bash">
{`count    889.000000
mean       0.524184
std        1.103705
min        0.000000
25%        0.000000
50%        0.000000
75%        1.000000
max        8.000000
Name: SibSp, dtype: float64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [49]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['SibSp'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [49]:</div>
                                        <div>
<pre><code className="language-bash">
{`0    606
1    209
2     28
4     18
3     16
8      7
5      5
Name: SibSp, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [50]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='SibSp', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [50]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12faef1d0>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [51]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Parch'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [51]:</div>
                                        <div>
<pre><code className="language-bash">
{`count    889.000000
mean       0.382452
std        0.806761
min        0.000000
25%        0.000000
50%        0.000000
75%        0.000000
max        6.000000
Name: Parch, dtype: float64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [52]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Parch'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [52]:</div>
                                        <div>
<pre><code className="language-bash">
{`0    676
1    118
2     80
5      5
3      5
4      4
6      1
Name: Parch, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [53]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='Parch', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [53]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12fb0c0b8>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">At first glance these features seem like they would work without
                                        adjustment at this time, but I already
                                        know that I'll want to simplify the corresponding form in our web app to ask a
                                        question like "are you
                                        traveling with any children?" as opposed to "how many children are you traveling
                                        with?" (i.e. Reduce this
                                        to a binary true / false question.)</p>
                                    <p className="caption">Let's create two new features that indicate whether or not a
                                        passenger is traveling with 1 or more
                                        siblings / spouse and 1 or more children / parents.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [54]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# reduce the 'SibSp' feature to 1 or 0
def sibsp(row):
if row['SibSp'] == 0:
    val = 0
else:
    val = 1
return val

# reduce the 'Parch' feature to 1 or 0
def parch(row):
if row['Parch'] == 0:
    val = 0
else:
    val = 1
return val`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [55]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# apply a function along an axis of a DataFrame
train['SiblingSpouse'] = train.apply(sibsp, axis=1)
train['ParentChild'] = train.apply(parch, axis=1)
train.head(10)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [55]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th>PassengerId</th>
                                                            <th>Survived</th>
                                                            <th>Pclass</th>
                                                            <th>Sex</th>
                                                            <th>SibSp</th>
                                                            <th>Parch</th>
                                                            <th>Ticket</th>
                                                            <th>Embarked</th>
                                                            <th>Deck</th>
                                                            <th>AgeCategories</th>
                                                            <th>FareCategories</th>
                                                            <th>Title</th>
                                                            <th>SiblingSpouse</th>
                                                            <th>ParentChild</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>male</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>A/5 21171</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                            <td>mr</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>female</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>PC 17599</td>
                                                            <td>c</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>4</td>
                                                            <td>mrs</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>female</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>STON/O2. 3101282</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                            <td>miss</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>4</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>female</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>113803</td>
                                                            <td>s</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>3</td>
                                                            <td>mrs</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>5</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>male</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>373450</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>adult</td>
                                                            <td>1</td>
                                                            <td>mr</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>6</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>male</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>330877</td>
                                                            <td>q</td>
                                                            <td>unavailable</td>
                                                            <td>missing</td>
                                                            <td>1</td>
                                                            <td>mr</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>7</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>male</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>17463</td>
                                                            <td>s</td>
                                                            <td>e</td>
                                                            <td>middle_age</td>
                                                            <td>3</td>
                                                            <td>mr</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>8</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>male</td>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>349909</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>child</td>
                                                            <td>2</td>
                                                            <td>master</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                        </tr>
                                                        <tr>
                                                            <td>9</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>female</td>
                                                            <td>0</td>
                                                            <td>2</td>
                                                            <td>347742</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>1</td>
                                                            <td>mrs</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                        </tr>
                                                        <tr>
                                                            <td>10</td>
                                                            <td>1</td>
                                                            <td>2</td>
                                                            <td>female</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>237736</td>
                                                            <td>c</td>
                                                            <td>unavailable</td>
                                                            <td>teenager</td>
                                                            <td>2</td>
                                                            <td>mrs</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [56]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['SiblingSpouse'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [56]:</div>
                                        <div>
<pre><code className="language-bash">
{`count    889.000000
mean       0.318335
std        0.466093
min        0.000000
25%        0.000000
50%        0.000000
75%        1.000000
max        1.000000
Name: SiblingSpouse, dtype: float64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [57]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['SiblingSpouse'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [57]:</div>
                                        <div>
<pre><code className="language-bash">
{`0    606
1    283
Name: SiblingSpouse, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [58]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='SiblingSpouse', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [58]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12fe4d9e8>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [59]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['ParentChild'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [59]:</div>
                                        <div>
<pre><code className="language-bash">
{`count    889.000000
mean       0.239595
std        0.427077
min        0.000000
25%        0.000000
50%        0.000000
75%        0.000000
max        1.000000
Name: ParentChild, dtype: float64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [60]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['ParentChild'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [60]:</div>
                                        <div>
<pre><code className="language-bash">
{`0    676
1    213
Name: ParentChild, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [61]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='ParentChild', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [61]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12fec9550>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">This looks great and the correlation between traveling with
                                        someone and survival is clear. Let's drop the
                                        existing 'SibSp' and 'Parch' features from our dataset.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [62]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# drop the 'SibSp' and 'Parch' columns from the dataframe
train = train.drop(columns=['SibSp', 'Parch'])

# preview the updated dataframe
train.head()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [62]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th>PassengerId</th>
                                                            <th>Survived</th>
                                                            <th>Pclass</th>
                                                            <th>Sex</th>
                                                            <th>Ticket</th>
                                                            <th>Embarked</th>
                                                            <th>Deck</th>
                                                            <th>AgeCategories</th>
                                                            <th>FareCategories</th>
                                                            <th>Title</th>
                                                            <th>SiblingSpouse</th>
                                                            <th>ParentChild</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>male</td>
                                                            <td>A/5 21171</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                            <td>mr</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>female</td>
                                                            <td>PC 17599</td>
                                                            <td>c</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>4</td>
                                                            <td>mrs</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>female</td>
                                                            <td>STON/O2. 3101282</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                            <td>miss</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>4</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>female</td>
                                                            <td>113803</td>
                                                            <td>s</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>3</td>
                                                            <td>mrs</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>5</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>male</td>
                                                            <td>373450</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>adult</td>
                                                            <td>1</td>
                                                            <td>mr</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Sex and Pclass</h4>
                                    <p className="caption">Now that we've finished working with the features that needed
                                        some extra effort let's move on to the
                                        'Sex' and 'Pclass fetures.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [63]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Sex'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [63]:</div>
                                        <div>
<pre><code className="language-bash">
{`count      889
unique       2
top       male
freq       577
Name: Sex, dtype: object`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [64]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Sex'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [64]:</div>
                                        <div>
<pre><code className="language-bash">
{`male      577
female    312
Name: Sex, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [65]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='Sex', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [65]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12ff7e470>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [66]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Pclass'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [66]:</div>
                                        <div>
<pre><code className="language-bash">
{`count    889.000000
mean       2.311586
std        0.834700
min        1.000000
25%        2.000000
50%        3.000000
75%        3.000000
max        3.000000
Name: Pclass, dtype: float64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [67]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Pclass'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [67]:</div>
                                        <div>
<pre><code className="language-bash">
{`3    491
1    214
2    184
Name: Pclass, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [68]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='Pclass', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [68]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x12fb1b358>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">These features look great as-is. We don't need to do anything
                                        else with them at this time. Let's move on
                                        to our final two features.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Ticket and PassengerId</h4>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [69]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Ticket'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [69]:</div>
                                        <div>
<pre><code className="language-bash">
{`count          889
unique         680
top       CA. 2343
freq             7
Name: Ticket, dtype: object`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [70]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['Ticket'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [70]:</div>
                                        <div>
<pre><code className="language-bash">
{`CA. 2343      7
1601          7
347082        7
3101295       6
CA 2144       6
            ..
374910        1
CA. 2314      1
349912        1
113509        1
C.A. 24579    1
Name: Ticket, Length: 680, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [71]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='Ticket', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [71]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x130131278>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">Looks like a familiar problem we've seen before. Out of 889
                                        records there are 680 unique values. After
                                        reviewing the dataset one more time I'm not seeing any meaningful pattern in the
                                        values used for the
                                        'Ticket' feature. Given these limitations we will drop this column.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [72]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# drop the 'Ticket' column from the dataframe
train = train.drop(columns=['Ticket'])

# preview the updated dataframe
train.head()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [72]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th></th>
                                                            <th>PassengerId</th>
                                                            <th>Survived</th>
                                                            <th>Pclass</th>
                                                            <th>Sex</th>
                                                            <th>Embarked</th>
                                                            <th>Deck</th>
                                                            <th>AgeCategories</th>
                                                            <th>FareCategories</th>
                                                            <th>Title</th>
                                                            <th>SiblingSpouse</th>
                                                            <th>ParentChild</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>male</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                            <td>mr</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>2</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>female</td>
                                                            <td>c</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>4</td>
                                                            <td>mrs</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>female</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                            <td>miss</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>3</td>
                                                            <td>4</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>female</td>
                                                            <td>s</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>3</td>
                                                            <td>mrs</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>4</td>
                                                            <td>5</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>male</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>adult</td>
                                                            <td>1</td>
                                                            <td>mr</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [73]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['PassengerId'].describe()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [73]:</div>
                                        <div>
<pre><code className="language-bash">
{`count    889.000000
mean     446.000000
std      256.998173
min        1.000000
25%      224.000000
50%      446.000000
75%      668.000000
max      891.000000
Name: PassengerId, dtype: float64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [74]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`train['PassengerId'].value_counts()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [74]:</div>
                                        <div>
<pre><code className="language-bash">
{`891    1
293    1
304    1
303    1
302    1
     ..
590    1
589    1
588    1
587    1
1      1
Name: PassengerId, Length: 889, dtype: int64`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [75]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sns.barplot(x='PassengerId', y='Survived', data=train)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [75]:</div>
                                        <div>
<pre><code className="language-bash">
{`<matplotlib.axes._subplots.AxesSubplot at 0x131518588>`}
</code></pre>
                                        </div>
                                    </div>
                                    <div>
                                        <div>
                                            <img alt="" src="data:image/png;base64,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" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">We can drop the 'PassengerId' feature as it is irrelevant to our
                                        target variable 'survival'.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [76]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# drop the 'PassengerId' column from the dataframe
train = train.drop(columns=['PassengerId'])

# preview the updated dataframe
train.head()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [76]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th></th>
                                                            <th>Survived</th>
                                                            <th>Pclass</th>
                                                            <th>Sex</th>
                                                            <th>Embarked</th>
                                                            <th>Deck</th>
                                                            <th>AgeCategories</th>
                                                            <th>FareCategories</th>
                                                            <th>Title</th>
                                                            <th>SiblingSpouse</th>
                                                            <th>ParentChild</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>male</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                            <td>mr</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>female</td>
                                                            <td>c</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>4</td>
                                                            <td>mrs</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>2</td>
                                                            <td>1</td>
                                                            <td>3</td>
                                                            <td>female</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>young_adult</td>
                                                            <td>0</td>
                                                            <td>miss</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>3</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>female</td>
                                                            <td>s</td>
                                                            <td>c</td>
                                                            <td>adult</td>
                                                            <td>3</td>
                                                            <td>mrs</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>4</td>
                                                            <td>0</td>
                                                            <td>3</td>
                                                            <td>male</td>
                                                            <td>s</td>
                                                            <td>unavailable</td>
                                                            <td>adult</td>
                                                            <td>1</td>
                                                            <td>mr</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">After exploring our data and engineering some new features we're
                                        ready to move on and preprocess our
                                        data.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h2>Data preprocessing</h2>
                                    <p className="caption">Before doing anything else, let's pause to summarize our updated
                                        dataset.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [77]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# print a summary of the dataframe
train.info()`}</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`<class 'pandas.core.frame.DataFrame'>
Int64Index: 889 entries, 0 to 890
Data columns (total 10 columns):
Survived          889 non-null int64
Pclass            889 non-null int64
Sex               889 non-null object
Embarked          889 non-null object
Deck              889 non-null object
AgeCategories     889 non-null category
FareCategories    889 non-null category
Title             889 non-null object
SiblingSpouse     889 non-null int64
ParentChild       889 non-null int64
dtypes: category(2), int64(4), object(4)
memory usage: 105.0+ KB`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">The dataset has <strong>889 examples</strong> (2 fewer than
                                        before after we dropped the rows with missing
                                        'Embarked' values) and <strong>9 features</strong> plus the <strong>1 target
                                            variable</strong>,
                                        'Survived'.</p>
                                    <p className="caption">4 of the features are integers<br />
                                        4 of the features are objects<br />
                                        2 of the features are categories</p>
                                    <p className="caption">Before we feed the dataset into our machine learning models we
                                        need to transform categorical values into
                                        numerical values because many models do not work with textual values.</p>
                                    <p className="caption">There are a number of strategies to handle categorical features.
                                        We're going to use the "one hot
                                        encoding" method which will accomplish two things for us:</p>
                                    <p className="caption">1) Transfrom the dataset's textual values into numerical
                                        values<br />
                                        2) Ensure the transformed values are equally important</p>
                                    <p className="caption">In some cases, when textual values are encoded to numerical
                                        values, some values will be greater than the
                                        other values. This can imply that those values are of higher importance than the
                                        others, which can result
                                        in inacurate predictions.</p>
                                    <p className="caption">The one hot encoding technique essentially creates a "dummy"
                                        feature for each distinct value in a
                                        categorical feature. Once the dummy values are created a boolean value (0 or 1)
                                        is populated to indicate
                                        whether the value is true or false for the feature.</p>
                                    <p className="caption">The pandas library has a built-in function called get_dummies()
                                        that does exactly this.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [78]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# rename column headers
train = train.rename(columns={
    'Survived': 'survived',
    'Pclass': 'passenger_class',
    'Sex': 'gender',
    'Embarked': 'embarked',
    'Deck': 'deck',
    'AgeCategories': 'age_categories',
    'FareCategories': 'fare_categories',
    'Title': 'title',
    'SiblingSpouse': 'sibling_or_spouse',
    'ParentChild': 'parent_or_child',
})

# convert categorical variable into dummy variables
train = pd.get_dummies(train.astype(str), columns=['passenger_class', 'gender', 'sibling_or_spouse', 'parent_or_child', 'embarked', 'deck', 'age_categories', 'fare_categories', 'title',])

# cast all values in the dataframe as integers
train = train.astype(int) 

# preview the updated dataframe
train.head()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [78]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th>survived</th>
                                                            <th>passenger_class_1</th>
                                                            <th>passenger_class_2</th>
                                                            <th>passenger_class_3</th>
                                                            <th>gender_female</th>
                                                            <th>gender_male</th>
                                                            <th>sibling_or_spouse_0</th>
                                                            <th>sibling_or_spouse_1</th>
                                                            <th>parent_or_child_0</th>
                                                            <th>parent_or_child_1</th>
                                                            <th>...</th>
                                                            <th>fare_categories_2</th>
                                                            <th>fare_categories_3</th>
                                                            <th>fare_categories_4</th>
                                                            <th>fare_categories_5</th>
                                                            <th>fare_categories_Missing</th>
                                                            <th>title_master</th>
                                                            <th>title_miss</th>
                                                            <th>title_mr</th>
                                                            <th>title_mrs</th>
                                                            <th>title_other</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>...</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>...</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>...</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>...</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                        </tr>
                                                        <tr>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>...</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                            <td>1</td>
                                                            <td>0</td>
                                                            <td>0</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                                <p className="caption">5 rows × 41 columns</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">Let's pause one last time to summarize our dataset to see what it
                                        looks like before moving on.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [79]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# print a summary of the dataframe
train.info()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`<class 'pandas.core.frame.DataFrame'>
Int64Index: 889 entries, 0 to 890
Data columns (total 41 columns):
survived                      889 non-null int64
passenger_class_1             889 non-null int64
passenger_class_2             889 non-null int64
passenger_class_3             889 non-null int64
gender_female                 889 non-null int64
gender_male                   889 non-null int64
sibling_or_spouse_0           889 non-null int64
sibling_or_spouse_1           889 non-null int64
parent_or_child_0             889 non-null int64
parent_or_child_1             889 non-null int64
embarked_c                    889 non-null int64
embarked_q                    889 non-null int64
embarked_s                    889 non-null int64
deck_a                        889 non-null int64
deck_b                        889 non-null int64
deck_c                        889 non-null int64
deck_d                        889 non-null int64
deck_e                        889 non-null int64
deck_f                        889 non-null int64
deck_g                        889 non-null int64
deck_t                        889 non-null int64
deck_unavailable              889 non-null int64
age_categories_adult          889 non-null int64
age_categories_child          889 non-null int64
age_categories_middle_age     889 non-null int64
age_categories_missing        889 non-null int64
age_categories_senior         889 non-null int64
age_categories_teenager       889 non-null int64
age_categories_young_adult    889 non-null int64
fare_categories_0             889 non-null int64
fare_categories_1             889 non-null int64
fare_categories_2             889 non-null int64
fare_categories_3             889 non-null int64
fare_categories_4             889 non-null int64
fare_categories_5             889 non-null int64
fare_categories_Missing       889 non-null int64
title_master                  889 non-null int64
title_miss                    889 non-null int64
title_mr                      889 non-null int64
title_mrs                     889 non-null int64
title_other                   889 non-null int64
dtypes: int64(41)
memory usage: 291.7 KB`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">We still have <strong>889 examples</strong>, all integers, but
                                        after one hot encoding our dataset we now
                                        have <strong>40 features</strong> plus the <strong>1 target variable</strong>
                                        'Survived'.</p>
                                    <p className="caption">Out dataset is now ready to be used in our machine learning
                                        models.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div id="fit-train-predict" className="scrollspy">
                                    <h1>Part II: Fit models and make predictions</h1>
                                    <h4>Split data into train and test</h4>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [80]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`feature_set = [
    'passenger_class_1',
    'passenger_class_2',
    'passenger_class_3',
    'gender_female',
    'gender_male',
    'sibling_or_spouse_0',
    'sibling_or_spouse_1',
    'parent_or_child_0',
    'parent_or_child_1',
    'embarked_c',
    'embarked_q',
    'embarked_s',
    'deck_a',
    'deck_b',
    'deck_c',
    'deck_d',
    'deck_e',
    'deck_f',
    'deck_g',
    'deck_t',
    'deck_unavailable',
    'age_categories_adult',
    'age_categories_child',
    'age_categories_middle_age',
    'age_categories_missing',
    'age_categories_senior',
    'age_categories_teenager',
    'age_categories_young_adult',
    'fare_categories_0',
    'fare_categories_1',
    'fare_categories_2',
    'fare_categories_3',
    'fare_categories_4',
    'fare_categories_5',
    'fare_categories_Missing',
    'title_master',
    'title_miss',
    'title_mr',
    'title_mrs',
    'title_other'
]

X = train[feature_set]
y = train['survived']

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h3>Logistic Regression</h3>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [81]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`lr = LogisticRegression()

lr.fit(train_X, train_y)

lr_predictions = lr.predict(test_X)
lr_predictions`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [81]:</div>
                                        <div>
<pre><code className="language-bash">
{`array([1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
       1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0,
       1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0,
       0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1,
       1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
       0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0,
       1, 0, 1])`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h3>Stochastic Gradient Descent (SGD</h3>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [82]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`sgd = SGDClassifier(max_iter=5, tol=None)

sgd.fit(train_X, train_y)

sgd_predictions = sgd.predict(test_X)
sgd_predictions`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [82]:</div>
                                        <div>
<pre><code className="language-bash">
{`array([1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1,
       1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1,
       1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0,
       0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
       1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
       1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0,
       0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
       1, 0, 1])`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h3>Random Forest</h3>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [83]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`rf = RandomForestClassifier(n_estimators=100)

rf.fit(train_X, train_y)

rf_predictions = rf.predict(test_X)
rf_predictions`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [83]:</div>
                                        <div>
<pre><code className="language-bash">
{`array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
       1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
       1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0,
       0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1,
       1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0,
       0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1,
       1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0,
       1, 0, 1])`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h3>K Nearest Neighbor</h3>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [84]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(train_X, train_y)

knn_predictions = knn.predict(test_X)
knn_predictions`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [84]:</div>
                                        <div>
<pre><code className="language-bash">
{`array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
       1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0,
       1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0,
       1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1,
       1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
       1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0,
       0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
       1, 0, 1])`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h3>Gaussian Naive Bayes</h3>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [85]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`gnb = GaussianNB()

gnb.fit(train_X, train_y)

gnb_predictions = gnb.predict(test_X)
gnb_predictions`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [85]:</div>
                                        <div>
<pre><code className="language-bash">
{`array([1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
       1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0,
       1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0,
       0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1,
       1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
       1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
       0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1,
       1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
       1, 0, 1])`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h3>Perceptron</h3>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [86]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`perceptron = Perceptron(max_iter=10)

perceptron.fit(train_X, train_y)

perceptron_predictions = perceptron.predict(test_X)
perceptron_predictions`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [86]:</div>
                                        <div>
<pre><code className="language-bash">
{`array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1,
       1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0,
       1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1,
       1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0,
       1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0,
       0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1])`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h3>Linear Support Vector Machine</h3>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [87]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`linear_svc = LinearSVC()

linear_svc.fit(train_X, train_y)

linear_svc_predictions = linear_svc.predict(test_X)
linear_svc_predictions`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [87]:</div>
                                        <div>
<pre><code className="language-bash">
{`array([1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
       1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0,
       1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0,
       0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1,
       1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
       0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0,
       1, 0, 1])`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h3>Decision Tree</h3>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [88]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`dt = DecisionTreeClassifier()

dt.fit(train_X, train_y)

dt_predictions = dt.predict(test_X)
dt_predictions`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [88]:</div>
                                        <div>
<pre><code className="language-bash">
{`array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
       1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
       1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0,
       0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1,
       1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0,
       0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1,
       0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0,
       1, 0, 1])`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h2>Score models</h2>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [89]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# Logistic Regression
lr_accuracy = accuracy_score(test_y, lr_predictions)

# Stochastic Gradient Descent (SGD)
sgd_accuracy = accuracy_score(test_y, sgd_predictions)

# Random Forest
rf_accuracy = accuracy_score(test_y, rf_predictions)

# K Nearest Neighbor
knn_accuracy = accuracy_score(test_y, knn_predictions)

# Gaussian Naive Bayes
gnb_accuracy = accuracy_score(test_y, gnb_predictions)

# Perceptron
perceptron_accuracy = accuracy_score(test_y, perceptron_predictions)

# Linear Support Vector Machine
linear_svc_accuracy = accuracy_score(test_y, linear_svc_predictions)

# Decision Tree
dt_accuracy = accuracy_score(test_y, dt_predictions)

# create new dataframe
results = pd.DataFrame({
'Model': ['Logistic Regression',
        'Stochastic Gradient Descent (SGD)',
        'Random Forest', 
        'K Nearest Neighbor',
        'Gaussian Naive Bayes',
        'Perceptron', 
        'Linear Support Vector Machine', 
        'Decision Tree'],
'Score': [lr_accuracy,
        sgd_accuracy,
        rf_accuracy, 
        knn_accuracy,
        gnb_accuracy,
        perceptron_accuracy, 
        linear_svc_accuracy,
        dt_accuracy]
})

# sort results by 'Score'
result_df = results.sort_values(by='Score', ascending=False)

# preview the dataframe
result_df.head(8)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [89]:</div>
                                        <div>
                                            <div className="output-table">
                                                <table className="bordered">
                                                    <thead>
                                                        <tr>
                                                            <th>Model</th>
                                                            <th>Score</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <td>Random Forest</td>
                                                            <td>0.793722</td>
                                                        </tr>
                                                        <tr>
                                                            <td>Logistic Regression</td>
                                                            <td>0.789238</td>
                                                        </tr>
                                                        <tr>
                                                            <td>Linear Support Vector Machine</td>
                                                            <td>0.780269</td>
                                                        </tr>
                                                        <tr>
                                                            <td>Decision Tree</td>
                                                            <td>0.766816</td>
                                                        </tr>
                                                        <tr>
                                                            <td>K Nearest Neighbor</td>
                                                            <td>0.739910</td>
                                                        </tr>
                                                        <tr>
                                                            <td>Gaussian Naive Bayes</td>
                                                            <td>0.704036</td>
                                                        </tr>
                                                        <tr>
                                                            <td>Stochastic Gradient Descent (SGD)</td>
                                                            <td>0.605381</td>
                                                        </tr>
                                                        <tr>
                                                            <td>Perceptron</td>
                                                            <td>0.479821</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">The Random Forest Classifier model scored best. Let's use K-Fold
                                        Cross Validation to randomly split the
                                        training data into <em>K</em> subsets (i.e. folds). This retruns a list of
                                        length <em>K</em> which we can
                                        use to calculate the mean (the central tendency of a given set of numbers) and
                                        standard deviation (measure
                                        of the amount of variation or dispersion of a set of values).</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h2>K-Fold Cross Validation</h2>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [90]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`# evaluate a score by cross-validation
cross_validation = cross_val_score(rf, train_X, train_y, cv=10, scoring='accuracy')

# print the resulting list, calculated mean and calculated standard deviation
print('Cross Vzlidation:', cross_validation)
print('Mean: %', round(cross_validation.mean() * 100, 2))
print('Standard Deviation: %', round(cross_validation.std() * 100, 2))`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`Cross Vzlidation: [0.7761194  0.7761194  0.89552239 0.82089552 0.79104478 0.85074627
0.70149254 0.78787879 0.87878788 0.81538462]
Mean: % 80.94
Standard Deviation: % 5.36`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">Not much of an improvement in accuracy, but we have verified that
                                        the model's perfomance.</p>
                                    <p className="caption">There are numerours other things we could do to try an improve
                                        the model's performance and accuracy, but what we have is a working model that's
                                        been trained, so let's call this
                                        good enough for now and move on.</p>
                                    <p className="caption">Our final task is to serialize the model so it can be used by
                                        our web app to make predictions.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div id="export" className="scrollspy">
                                    <h1>Part III: Export the model</h1>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Save the model (i.e. serialize a Python object)</h4>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [91]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`pickle_out = open('model.pickle', 'wb')
pickle.dump(rf, pickle_out)
pickle_out.close()`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">That's all it takes! Our trained model has been serialized and
                                        can be used by our web app. Let's take a moment to review how that works.</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Import the model (i.e. deserialize a Python object)</h4>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [92]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`pickle_in = open('model.pickle', 'rb')
model = pickle.load(pickle_in)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <h4>Use the model to make predictions</h4>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div className="input">
                                <div>In [93]:</div>
                                <div>
                                    <div>
                                        <div>
<pre><code className="language-python">
{`model.predict(test_X)`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div>
                                    <div className="output">
                                        <div>Out [93]:</div>
                                        <div>
<pre><code className="language-bash">
{`array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
       1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
       1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0,
       0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1,
       1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0,
       0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1,
       1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0,
       1, 0, 1])`}
</code></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div>
                                    <p className="caption">It works!</p>
                                </div>
                            </div>
                        </div>
                        <div>
                            <div>
                                <div id="conclusion" className="scrollspy">
                                    <h1>Conclusion</h1>
                                    <p className="caption">Let's close by reviewing what we've accomplished.</p>
                                    <p className="caption">First we took a raw dataset and analyzed its contents to
                                        understand how its feature values are related to a target variable.<br />
                                        Next we transformed the data so that it could be used to train machine learning
                                        models.<br />
                                        Then we selected the highest performing model and serialized it so that it can
                                        be used to make predictions based on user input.</p>
                                    <p className="caption">I hope this has been a helpful exercise that demonstrates the
                                        basics of machine learning.</p>
                                    <p className="caption">For more information about this project you can review the
                                        repository on my <a
                                            href="https://github.com/NicChappell/titanic-dataset">github</a>.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
        
            </div>
        </div>
    )
}

export default BehindTheScenes
