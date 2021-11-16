#!/usr/bin/env python
# coding: utf-8

# > *Name: Liban Ali*
# >
# > *Email : LA627@live.mdx.ac.uk*
# >
# > *ID : M00520253*
# >
# > **Which Algorithm?**
# >
# > The algorithm I chose for this task was K Nearest Neighbour. KNN is a
# > supervised classification algorithm we can use to classify data points
# > based
# >
# > on the points that are most similar to it. It uses test data to make
# > an “educated guess” on what an unclassified point should be classified
# > as.
# >
# > Although there are benefits of using KNN, there are some disadvantages
# > that come with this algorithm. One of them being that they do not work
# >
# > well with large datasets. This is because with large datasets, the
# > cost of calculating the distance between the new point and each
# > existing point is
# >
# > huge which degrades the performance of the algorithm. Another one is
# > that it doesn’t work well with high dimensions. This is because with
# > the
# >
# > large number of dimensions, it becomes more difficult for the
# > algorithm to calculate the distance in each dimension.
# >
# > **Importing Libraries**
# >
# > I will be building a KNN model with the help of the machine learning
# > package ‘scikit-learn’. There were three critical libraries that I
# > needed to
# 
# import, and they were pandas, NumPy, and KNN. I needed to upload the
# pandas library so that it could give me the ability to analyse and
# 
# > manipulate data. Data analysis is important here because it would
# > allow me to read the data in deep critical thinking and also help me
# > find the
# >
# > relations between the variables. I needed to upload the NumPy library
# > so that it could give me the ability to work with multi-dimensional
# > arrays and
# >
# > matrix data structures. This is important because the dataset that was
# > provided to me contains numeric data and NumPy will give me the
# > ability to
# >
# > perform mathematical and logical operations on this data. Finally,
# > ‘KNeighborsClassifier’ was added because I needed to get access to the
# > KNN
# >
# > algorithm.
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>In [2]:</p>
# <p>Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p><strong>import</strong> matplotlib.pyplot <strong>as</strong> plt<br />
# <strong>import</strong> pandas <strong>as</strong> pd<br />
# <strong>import</strong> numpy <strong>as</strong> np<br />
# <strong>from</strong> sklearn.model_selection <strong>import</strong> train_test_split<br />
# <strong>from</strong> sklearn.preprocessing <strong>import</strong> StandardScaler<br />
# <strong>from</strong> sklearn.metrics <strong>import</strong> confusion_matrix<br />
# <strong>from</strong> sklearn.neighbors <strong>import</strong> KNeighborsClassifier</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# |     |
# |-----|
# |     |
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p><strong>from</strong> sklearn.metrics <strong>import</strong> accuracy_score<br />
# <strong>import</strong> seaborn <strong>as</strong> sb<br />
# <strong>from</strong> sklearn.metrics <strong>import</strong> classification_report<br />
# <strong>from</strong> sklearn.metrics <strong>import</strong> precision_score<br />
# <strong>from</strong> sklearn <strong>import</strong> metrics</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# > **Importing Dataset**
# >
# > In this section, I imported the dataset from my desktop with the help
# > of pandas. The dataset contains 51 columns. 50 of these columns are
# > labelled
# >
# > ‘x1’ to ‘x50’ and one of them is called ‘Y’. The ‘x’ columns contain
# > numeric data, and the ‘y’ column contain the characters ‘a’, ‘b’ and
# > ‘c’.
# 
# <table>
# <thead>
# <tr class="header">
# <th>In [7]:</th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# </tr>
# </thead>
# <tbody>
# <tr class="odd">
# <td>In [8]:</td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# </tr>
# <tr class="even">
# <td>Out[8]:</td>
# <td><strong>x1</strong></td>
# <td><strong>x2</strong></td>
# <td><strong>x3</strong></td>
# <td><strong>x4</strong></td>
# <td><strong>x5</strong></td>
# <td><strong>x6</strong></td>
# <td><strong>x7</strong></td>
# <td><strong>x8</strong></td>
# <td><strong>x9</strong></td>
# <td><strong>x10</strong></td>
# <td><strong>...</strong></td>
# <td><strong>x42</strong></td>
# <td><blockquote>
# <p><strong>x43</strong></p>
# </blockquote></td>
# <td><strong>x</strong></td>
# <td></td>
# </tr>
# <tr class="odd">
# <td></td>
# <td><strong>0</strong></td>
# <td>16.967792</td>
# <td>17.299906</td>
# <td>10.842266</td>
# <td>15.511571</td>
# <td>27.006127</td>
# <td>-25.992496</td>
# <td>5.533636</td>
# <td>18.422843</td>
# <td>-19.352110</td>
# <td><blockquote>
# <p>-0.224973</p>
# </blockquote></td>
# <td>...</td>
# <td>13.980005</td>
# <td><blockquote>
# <p>-9.398601</p>
# </blockquote></td>
# <td><blockquote>
# <p>21.1724</p>
# </blockquote></td>
# </tr>
# <tr class="even">
# <td></td>
# <td><strong>1</strong></td>
# <td>16.780632</td>
# <td>21.942175</td>
# <td>48.655572</td>
# <td>-3.923519</td>
# <td>42.845266</td>
# <td><blockquote>
# <p>1.189123</p>
# </blockquote></td>
# <td>4.291448</td>
# <td><blockquote>
# <p>5.755105</p>
# </blockquote></td>
# <td>-18.177523</td>
# <td>-13.574741</td>
# <td>...</td>
# <td>-0.510062</td>
# <td><blockquote>
# <p>-6.146813</p>
# </blockquote></td>
# <td><blockquote>
# <p>32.6199</p>
# </blockquote></td>
# </tr>
# <tr class="odd">
# <td></td>
# <td><strong>2</strong></td>
# <td>13.982242</td>
# <td>10.523966</td>
# <td>18.327122</td>
# <td>-4.600511</td>
# <td>11.254636</td>
# <td>-18.146897</td>
# <td>6.053160</td>
# <td>17.263444</td>
# <td>-18.060451</td>
# <td><blockquote>
# <p>-0.279081</p>
# </blockquote></td>
# <td>...</td>
# <td>23.282090</td>
# <td>-12.559800</td>
# <td><blockquote>
# <p>-0.9359</p>
# </blockquote></td>
# </tr>
# <tr class="even">
# <td></td>
# <td><strong>3</strong></td>
# <td><blockquote>
# <p>7.605712</p>
# </blockquote></td>
# <td>17.542918</td>
# <td>41.860639</td>
# <td>-1.182127</td>
# <td>39.226712</td>
# <td><blockquote>
# <p>-6.528176</p>
# </blockquote></td>
# <td>4.410899</td>
# <td>23.213193</td>
# <td>-15.128678</td>
# <td><blockquote>
# <p>8.573257</p>
# </blockquote></td>
# <td>...</td>
# <td>-2.296166</td>
# <td><blockquote>
# <p>-7.814061</p>
# </blockquote></td>
# <td><blockquote>
# <p>-0.4543</p>
# </blockquote></td>
# </tr>
# <tr class="odd">
# <td></td>
# <td><strong>4</strong></td>
# <td><blockquote>
# <p>9.600247</p>
# </blockquote></td>
# <td>12.333797</td>
# <td>26.824897</td>
# <td>11.596199</td>
# <td>39.108405</td>
# <td>-17.500930</td>
# <td>7.472530</td>
# <td>18.702467</td>
# <td>-36.212971</td>
# <td>-16.737312</td>
# <td>...</td>
# <td>15.090340</td>
# <td>-10.628251</td>
# <td><blockquote>
# <p>13.8985</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>df <strong>=</strong> pd<strong>.</strong>read_csv('Desktop/synthetic_Classification.csv')</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>df<strong>.</strong>head()</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# > 5 rows × 51 columns
# >
# > **Data Split, Feature Selection and Standardisation**
# >
# > Here, I began standardisation and splitting the data. Before actually
# > training and testing the data, I had to check if there were any
# > null/missing
# >
# > values in the dataset. Evidently, there wasn’t any which is good. It’s
# > important to check for any null values just in case you need to clean
# > the
# >
# > dataset which is why I had to check. I then I had to determine which
# > column was going to be my independent variable and which was going to
# > be
# >
# > my dependent variable. For the independent variable (x), I selected
# > columns 0 to 50 and then assigned it to ‘X’. For the dependent
# > variable (y), I
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>Create PDF in your applications with the Pdfcrowd HTML to PDF API</p>
# </blockquote></td>
# <td>PDFCROWD</td>
# </tr>
# </tbody>
# </table>
# 
# > selected the ‘y’ column and assigned it to ‘Y’. It is important to do
# > this because the algorithm needs to understand which is going to be
# > the target
# >
# > variable (y) and which is going to be the predictors (x). Furthermore,
# > many machine learning algorithms require attributes and labels to be
# > included
# >
# > in separate labels, in this case it would be a ‘x’ and ‘y’.
# >
# > Next step was data standardisation. ‘StandardScaler’ is to carry out
# > the task of data standardisation. A dataset typically involves
# > variables that
# >
# > differ in scale. For example, with the dataset that was given, the
# > values differed by each column. Since these columns vary in scale,
# > they are then
# >
# > standardised in order to have a similar scale while building a machine
# > learning model. Variables that vary in scale do not contribute to the
# > analysis
# >
# > equally and this can end up generating a bias in the data. This is why
# > it is important to scale the data.
# >
# > Now that I have successfully assigned the variables and standardised
# > the data, I went on to split the data into training and testing
# > through using
# >
# > the ‘train_test_split’ library. This allows me to split the dataset
# > returning 4 values, these being the train attributes (X_train), test
# > attributes (X_test),
# >
# > train labels (Y_train) and the test label (Y_test). I will be training
# > 80% of the data and testing the remaining 20%.
# 
# |           |     |
# |-----------|-----|
# | In \[9\]: |     |
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>print(df<strong>.</strong>columns)</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# > Index(\['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
# > 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
# > 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30',
# > 'x31', 'x32', 'x33', 'x34', 'x35', 'x36', 'x37', 'x38', 'x39', 'x40',
# > 'x41', 'x42', 'x43', 'x44', 'x45', 'x46', 'x47', 'x48', 'x49', 'x50',
# > 'y'\], dtype='object')
# 
# <table>
# <thead>
# <tr class="header">
# <th>In [37]:</th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# <th></th>
# </tr>
# </thead>
# <tbody>
# <tr class="odd">
# <td>Out[37]:</td>
# <td><strong>x1</strong></td>
# <td><strong>x2</strong></td>
# <td><strong>x3</strong></td>
# <td><strong>x4</strong></td>
# <td><strong>x5</strong></td>
# <td><strong>x6</strong></td>
# <td><strong>x7</strong></td>
# <td><strong>x8</strong></td>
# <td><strong>x9</strong></td>
# <td><strong>x10</strong></td>
# <td><strong>...</strong></td>
# <td><strong>x42</strong></td>
# <td><strong>x43</strong></td>
# <td><strong>x44</strong></td>
# <td><strong>x45</strong></td>
# <td><strong>x46</strong></td>
# <td><strong>x47</strong></td>
# <td><strong>x48</strong></td>
# <td><strong>x49</strong></td>
# <td><strong>x50</strong></td>
# <td><strong>y</strong></td>
# </tr>
# <tr class="even">
# <td><strong>0</strong></td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>...</td>
# <td>False</td>
# <td>False</td>
# <td><blockquote>
# <p>False</p>
# </blockquote></td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td><blockquote>
# <p>False</p>
# </blockquote></td>
# </tr>
# <tr class="odd">
# <td><strong>1</strong></td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>...</td>
# <td>False</td>
# <td>False</td>
# <td><blockquote>
# <p>False</p>
# </blockquote></td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td><blockquote>
# <p>False</p>
# </blockquote></td>
# </tr>
# <tr class="even">
# <td><strong>2</strong></td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>...</td>
# <td>False</td>
# <td>False</td>
# <td><blockquote>
# <p>False</p>
# </blockquote></td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td><blockquote>
# <p>False</p>
# </blockquote></td>
# </tr>
# <tr class="odd">
# <td><strong>3</strong></td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>...</td>
# <td>False</td>
# <td>False</td>
# <td><blockquote>
# <p>False</p>
# </blockquote></td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td><blockquote>
# <p>False</p>
# </blockquote></td>
# </tr>
# <tr class="even">
# <td><strong>4</strong></td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>...</td>
# <td>False</td>
# <td>False</td>
# <td><blockquote>
# <p>False</p>
# </blockquote></td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td><blockquote>
# <p>False</p>
# </blockquote></td>
# </tr>
# <tr class="odd">
# <td><strong>...</strong></td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# <td>...</td>
# </tr>
# <tr class="even">
# <td><strong>195</strong></td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>...</td>
# <td>False</td>
# <td>False</td>
# <td><blockquote>
# <p>False</p>
# </blockquote></td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td><blockquote>
# <p>False</p>
# </blockquote></td>
# </tr>
# <tr class="odd">
# <td><strong>196</strong></td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>...</td>
# <td>False</td>
# <td>False</td>
# <td><blockquote>
# <p>False</p>
# </blockquote></td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td>False</td>
# <td><blockquote>
# <p>False</p>
# </blockquote></td>
# </tr>
# <tr class="even">
# <td><blockquote>
# <p>Create PDF in your applications with the Pdfcrowd HTML to PDF API</p>
# </blockquote></td>
# <td>PDFCROWD</td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>df<strong>.</strong>isnull()</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# | **197** | **x1** | **x2** | **x3** | **x4** | **x5** | **x6** | **x7** | **x8** | **x9** | **x10** | **...** | **x42** | **x43** | **x44** | **x45** | **x46** | **x47** | **x48** | **x49** | **x50** | **y** |
# |---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|-------|
# |         | False  | False  | False  | False  | False  | False  | False  | False  | False  | False   | ...     | False   | False   | False   | False   | False   | False   | False   | False   | False   | False |
# | **198** | False  | False  | False  | False  | False  | False  | False  | False  | False  | False   | ...     | False   | False   | False   | False   | False   | False   | False   | False   | False   | False |
# | **199** | False  | False  | False  | False  | False  | False  | False  | False  | False  | False   | ...     | False   | False   | False   | False   | False   | False   | False   | False   | False   | False |
# 
# > 200 rows × 51 columns
# 
# | In \[10\]: |     |
# |------------|-----|
# | In \[13\]: |     |
# | In \[12\]: |     |
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>X <strong>=</strong> df<strong>.</strong>iloc[:,0:49]<br />
# Y <strong>=</strong> df<strong>.</strong>iloc[:,50]</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>sc <strong>=</strong> StandardScaler()<br />
# X_train <strong>=</strong> sc<strong>.</strong>fit_transform(X_train)<br />
# X_test <strong>=</strong> sc<strong>.</strong>transform(X_test)</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>X_train, X_test, Y_train, Y_test <strong>=</strong> train_test_split(X, Y, test_size<strong>=</strong>0.2, random_state<strong>=</strong>1)</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# > **KNN - Training, Predicting and Accuracy**
# >
# > In this section I began to train my algorithm. I fit the model with
# > the X_train and Y_train data which, as mentioned before, is 80% of the
# > data. I will
# >
# > use this data to train the KNN model. The ‘fit’ method is used to
# > train the model. I also used the ‘predict’ method to do the testing on
# > the remaining
# >
# > 20% of the data. After training and predicting, I tried to calculate
# > the accuracy of the model. There were many metrics I could have used
# > to help me
# >
# > calculate the accuracy of my model. I chose to use the ‘accuracy
# > score’ and ‘confusion matrix’. The accuracy score metric is the most
# > basic of all
# >
# > metrics. I chose this metric because of its simplicity and how easy it
# > was to understand. However, just like any other metric, it has its
# > limitations.
# >
# > One such limitation is that it does not take into account how the data
# > is distributed which could lead to incorrect conclusions. The other
# > one I chose
# >
# > was confusion matrix which is often used to measure the performance of
# > a classification algorithm. It compares the actual target values with
# > those
# >
# > predicted by the machine learning model. This then gives us a good
# > idea of how well our model is performing and what kind of errors its
# > making.
# >
# > As you can see, I got an accuracy score of 60% which seems to be okay.
# > I then used the transpose method to make a column of ‘Y_test’ and
# >
# > ‘y_pred’. As you can see, the algorithm predicted the majority of the
# > classes correctly.
# >
# > <img src="attachment:2371bcb7aefa42d3ad953b6c35b8fce6/media/image1.png" style="width:4.71111in;height:0.65833in" />
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>Create PDF in your applications with the Pdfcrowd HTML to PDF API</p>
# </blockquote></td>
# <td>PDFCROWD</td>
# </tr>
# </tbody>
# </table>
# 
# > <img src="attachment:2371bcb7aefa42d3ad953b6c35b8fce6/media/image2.png" style="width:3.71806in;height:2.78889in" />
# 
# <table>
# <thead>
# <tr class="header">
# <th>In [24]:</th>
# <th></th>
# <th></th>
# <th></th>
# </tr>
# </thead>
# <tbody>
# <tr class="odd">
# <td>In [28]:</td>
# <td></td>
# <td></td>
# <td></td>
# </tr>
# <tr class="even">
# <td>Out[28]:</td>
# <td><blockquote>
# <p>0.6</p>
# </blockquote></td>
# <td></td>
# <td></td>
# </tr>
# <tr class="odd">
# <td>In [29]:</td>
# <td></td>
# <td></td>
# <td></td>
# </tr>
# <tr class="even">
# <td>Out[29]:</td>
# <td><blockquote>
# <p>array([[ 3, 2, 0],<br />
# [ 3, 12, 2],</p>
# </blockquote></td>
# <td></td>
# <td></td>
# </tr>
# <tr class="odd">
# <td><blockquote>
# <p>[ 0, 9, 9]])</p>
# </blockquote></td>
# <td></td>
# <td></td>
# <td></td>
# </tr>
# <tr class="even">
# <td>In [30]:</td>
# <td></td>
# <td></td>
# <td></td>
# </tr>
# <tr class="odd">
# <td>Out[30]:</td>
# <td><strong>a</strong></td>
# <td><strong>b</strong></td>
# <td><blockquote>
# <p><strong>c</strong></p>
# </blockquote></td>
# </tr>
# <tr class="even">
# <td><strong>a</strong></td>
# <td>3</td>
# <td>2</td>
# <td><blockquote>
# <p>0</p>
# </blockquote></td>
# </tr>
# <tr class="odd">
# <td><blockquote>
# <p>Create PDF in your applications with the Pdfcrowd HTML to PDF API</p>
# </blockquote></td>
# <td>PDFCROWD</td>
# <td></td>
# <td></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>KNN <strong>=</strong> KNeighborsClassifier(n_neighbors <strong>=</strong>7)<br />
# KNN<strong>.</strong>fit(X_train, Y_train)<br />
# Y_pred <strong>=</strong> KNN<strong>.</strong>predict(X_test)</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>acc <strong>=</strong> accuracy_score(Y_test, Y_pred)<br />
# acc</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>cm <strong>=</strong> confusion_matrix(Y_test<strong>.</strong>values, Y_pred)<br />
# cm</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>cm1 <strong>=</strong> pd<strong>.</strong>DataFrame(data <strong>=</strong> cm, index <strong>=</strong>['a','b','c'], columns <strong>=</strong>['a','b','c']) cm1</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <thead>
# <tr class="header">
# <th><strong>a</strong></th>
# <th><strong>b</strong></th>
# <th><blockquote>
# <p><strong>c</strong></p>
# </blockquote></th>
# <th></th>
# </tr>
# </thead>
# <tbody>
# <tr class="odd">
# <td><strong>b</strong></td>
# <td>3</td>
# <td>12</td>
# <td><blockquote>
# <p>2</p>
# </blockquote></td>
# </tr>
# <tr class="even">
# <td><strong>c</strong></td>
# <td>0</td>
# <td>9</td>
# <td><blockquote>
# <p>9</p>
# </blockquote></td>
# </tr>
# <tr class="odd">
# <td>In [31]:</td>
# <td></td>
# <td></td>
# <td></td>
# </tr>
# <tr class="even">
# <td>In [32]:</td>
# <td></td>
# <td></td>
# <td></td>
# </tr>
# <tr class="odd">
# <td>Out[32]:</td>
# <td><strong>Y_test</strong></td>
# <td><blockquote>
# <p><strong>y_pred</strong></p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="even">
# <td><strong>0</strong></td>
# <td>b</td>
# <td><blockquote>
# <p>b</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="odd">
# <td><strong>1</strong></td>
# <td>c</td>
# <td><blockquote>
# <p>c</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="even">
# <td><strong>2</strong></td>
# <td>c</td>
# <td><blockquote>
# <p>c</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="odd">
# <td><strong>3</strong></td>
# <td>b</td>
# <td><blockquote>
# <p>b</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="even">
# <td><strong>4</strong></td>
# <td>b</td>
# <td><blockquote>
# <p>b</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="odd">
# <td><table>
# <tbody>
# <tr class="odd">
# <td><strong>5</strong></td>
# </tr>
# </tbody>
# </table></td>
# <td>c</td>
# <td><blockquote>
# <p>c</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="even">
# <td><strong>6</strong></td>
# <td>c</td>
# <td><blockquote>
# <p>b</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="odd">
# <td><strong>7</strong></td>
# <td>b</td>
# <td><blockquote>
# <p>c</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="even">
# <td><strong>8</strong></td>
# <td>b</td>
# <td><blockquote>
# <p>b</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="odd">
# <td><strong>9</strong></td>
# <td>c</td>
# <td><blockquote>
# <p>c</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="even">
# <td><strong>10</strong></td>
# <td>b</td>
# <td><blockquote>
# <p>a</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="odd">
# <td><strong>11</strong></td>
# <td>b</td>
# <td><blockquote>
# <p>a</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="even">
# <td><strong>12</strong></td>
# <td>b</td>
# <td><blockquote>
# <p>b</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="odd">
# <td><strong>13</strong></td>
# <td>b</td>
# <td><blockquote>
# <p>b</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="even">
# <td><strong>14</strong></td>
# <td>c</td>
# <td><blockquote>
# <p>c</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="odd">
# <td><strong>15</strong></td>
# <td>b</td>
# <td><blockquote>
# <p>a</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="even">
# <td><strong>16</strong></td>
# <td>a</td>
# <td><blockquote>
# <p>a</p>
# </blockquote></td>
# <td></td>
# </tr>
# <tr class="odd">
# <td>PDFCROWD</td>
# <td></td>
# <td></td>
# <td></td>
# </tr>
# <tr class="even">
# <td></td>
# <td></td>
# <td></td>
# <td></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>prediction_output <strong>=</strong> pd<strong>.</strong>DataFrame(data <strong>=</strong>[Y_test<strong>.</strong>values, Y_pred], index <strong>=</strong>['Y_test','y_pred'])</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>prediction_output<strong>.</strong>transpose()</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <thead>
# <tr class="header">
# <th><strong>18</strong></th>
# <th><strong>Y_test</strong></th>
# <th><strong>y_pred</strong></th>
# </tr>
# </thead>
# <tbody>
# <tr class="odd">
# <td></td>
# <td>a</td>
# <td>a</td>
# </tr>
# <tr class="even">
# <td><strong>19</strong></td>
# <td>a</td>
# <td>b</td>
# </tr>
# <tr class="odd">
# <td><strong>20</strong></td>
# <td>b</td>
# <td>b</td>
# </tr>
# <tr class="even">
# <td><table>
# <tbody>
# <tr class="odd">
# <td><strong>21</strong></td>
# </tr>
# </tbody>
# </table></td>
# <td>c</td>
# <td>c</td>
# </tr>
# <tr class="odd">
# <td><strong>22</strong></td>
# <td>c</td>
# <td>c</td>
# </tr>
# <tr class="even">
# <td><strong>23</strong></td>
# <td>c</td>
# <td>b</td>
# </tr>
# <tr class="odd">
# <td><strong>24</strong></td>
# <td>c</td>
# <td>b</td>
# </tr>
# <tr class="even">
# <td><strong>25</strong></td>
# <td>c</td>
# <td>b</td>
# </tr>
# <tr class="odd">
# <td><strong>26</strong></td>
# <td>c</td>
# <td>b</td>
# </tr>
# <tr class="even">
# <td><strong>27</strong></td>
# <td>b</td>
# <td>c</td>
# </tr>
# <tr class="odd">
# <td><strong>28</strong></td>
# <td>c</td>
# <td>c</td>
# </tr>
# <tr class="even">
# <td><strong>29</strong></td>
# <td>c</td>
# <td>c</td>
# </tr>
# <tr class="odd">
# <td><strong>30</strong></td>
# <td>c</td>
# <td>b</td>
# </tr>
# <tr class="even">
# <td><strong>31</strong></td>
# <td>c</td>
# <td>b</td>
# </tr>
# <tr class="odd">
# <td><strong>32</strong></td>
# <td>b</td>
# <td>b</td>
# </tr>
# <tr class="even">
# <td><strong>33</strong></td>
# <td>c</td>
# <td>b</td>
# </tr>
# <tr class="odd">
# <td><strong>34</strong></td>
# <td>b</td>
# <td>b</td>
# </tr>
# <tr class="even">
# <td><strong>35</strong></td>
# <td>a</td>
# <td>a</td>
# </tr>
# <tr class="odd">
# <td><strong>36</strong></td>
# <td>a</td>
# <td>b</td>
# </tr>
# <tr class="even">
# <td><strong>37</strong></td>
# <td>b</td>
# <td>b</td>
# </tr>
# <tr class="odd">
# <td><strong>38</strong></td>
# <td>b</td>
# <td>b</td>
# </tr>
# <tr class="even">
# <td><table>
# <tbody>
# <tr class="odd">
# <td><strong>39</strong></td>
# </tr>
# </tbody>
# </table></td>
# <td>b</td>
# <td>b</td>
# </tr>
# </tbody>
# </table>
# 
# > **KNN - Finding the value of 'k', Tuning and Visualisation
# > (interpretability)**
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>Create PDF in your applications with the Pdfcrowd HTML to PDF API</p>
# </blockquote></td>
# <td>PDFCROWD</td>
# </tr>
# </tbody>
# </table>
# 
# > In this section I will discussing how I found the value of k. ‘K’ in
# > KNN is a parameter that refers to the number of nearest neighbours to
# > include in
# >
# > the majority of the voting process. The most important
# > hyperparameter/tuning parameter for KNN is ‘n_neighbours’. I used this
# > to help me tune my
# >
# > model. I made ‘K’ equivalent to 24. It was important for me to make
# > this an even number due to the number of classes present. If the
# > number of
# >
# > classes is odd, then ‘K’ must be an even number in order to avoid a
# > tie. So, in this case, I had to choose an even number since the number
# > of
# >
# > classes was 3 (a, b and c).
# >
# > I created a local variable called ‘neigh’ and assigned it to the KNN
# > model and then I used the ‘fit’ method to train the model on training
# > data, which
# >
# > was X_train and Y_train. I then used the ‘predict’ function to do the
# > testing on the testing data which is X_test like I did earlier on. It
# > is essential to
# >
# > choose the optimal value for ‘K’, so what I did was fit and test the
# > model for different values for ‘K’, which was from 1 to 24. I did this
# > by using the
# >
# > ‘for’ loop and recorded the KNN testing accuracy in a variable. As you
# > can see, I found out that 0.65 (65%) was the maximum testing accuracy
# >
# > with ‘K’ being equivalent to 18 which is an improvement from the
# > previous accuracy (60%). Once this was complete, I decided to
# > visualise it using
# >
# > a line graph. I chose a line graph because they are designed for
# > comparing changes as values increase or decrease. From the graph, we
# > can see
# >
# > a lot of fluctuation, a rise and fall in accuracy as ‘k’ increases. We
# > can also see that the accuracy was at its highest when the number of
# > ‘K’ was 18
# >
# > as stated earlier. I implemented a graph of this so I could make the
# > predictions more interpretable for the user. The prediction phase is
# > the most
# >
# > important part of any algorithm which is why I decided to make it
# > interpretable.
# 
# | In \[45\]: |     |
# |------------|-----|
# | In \[46\]: |     |
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>K <strong>=</strong>24<br />
# <strong>for</strong> k <strong>in</strong> range (1, K):<br />
# neigh <strong>=</strong> KNeighborsClassifier(n_neighbors <strong>=</strong> k)<strong>.</strong>fit(X_train, Y_train) yhat <strong>=</strong> neigh<strong>.</strong>predict(X_test)<br />
# mean_acc[k<strong>-</strong>1]<strong>=</strong> accuracy_score(Y_test, yhat)</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>print(mean_acc)</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# > \[0.425 0.35 0.5 0.6 0.625 0.6 0.625 0.625 0.525 0.575 0.55 0.55 0.575
# > 0.6 0.575 0.6 0.625 0.525 0.65 0.575 0.6 0.6 0.65 \]
# 
# |            |     |
# |------------|-----|
# | In \[47\]: |     |
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>print("The best accuracy was", mean_acc<strong>.</strong>max(),"with k=", mean_acc<strong>.</strong>argmax())</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# > The best accuracy was 0.65 with k= 18
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>In [48]:</p>
# <p>Create PDF in your applications with the Pdfcrowd HTML to PDF API PDFCROWD</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>plt<strong>.</strong>plot(range(1,K), mean_acc,'blue')<br />
# <em>#plt.legend(('Accuracy'))</em></p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# |     |
# |-----|
# |     |
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>plt<strong>.</strong>ylabel('Accuracy')<br />
# plt<strong>.</strong>xlabel('Number of Neighbors (K)')<br />
# plt<strong>.</strong>tight_layout()<br />
# plt<strong>.</strong>show()</p>
# </blockquote></td>
# </tr>
# </tbody>
# </table>
# 
# > <img src="attachment:2371bcb7aefa42d3ad953b6c35b8fce6/media/image3.png" style="width:4.42917in;height:2.925in" />
# >
# > **Bias and Variance**
# >
# > For KNN the bias will always 0 when K=1. However, when K = 2 or higher
# > the bias will increase, and the variance will decrease simultaneously.
# >
# > From this we can learn that when the value of ‘k’ is higher than 1,
# > the model becomes a lot more complex and leans towards bias which can
# > affect
# >
# > the overall outcome. As you can see, the value of ‘K’ is higher than 1
# > which means that the bias will begin to increase and the variance will
# >
# > decrease simultaneously. It is because of this that the KNN model is
# > underfitting.
# >
# > **Time Complexity for testing, training and prediction**
# >
# > One of the limitations of KNN is that it slow when it comes to the
# > testing time. The complexity comes from having to compare the testing
# > point to
# >
# > every training point. For example, if you have data plotted on a graph
# > and you want to quickly find all the nearest neighbours on a graph, as
# > a
# >
# > human you would easily be able to detect the nearest neighbours.
# > However, KNN will view it differently. It sees a list of instances and
# > attribute
# >
# > values. For KNN to calculate the nearest neighbours, it will take the
# > testing instances and compare it individually to every one of the
# > training
# >
# > instances, hence why it is so slow. KNN doesn’t have an explicit
# > training phase, so the training would be O(d), but for testing it
# > would be O(n*d)*
# >
# > *which means it is slow. The testing involves n*d, where ‘n’ is the
# > number of training instances and ‘d’ is the dimensionality that you’re
# > working on.
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>Create PDF in your applications with the Pdfcrowd HTML to PDF API</p>
# </blockquote></td>
# <td>PDFCROWD</td>
# </tr>
# </tbody>
# </table>
# 
# > ‘K’ is the number of neighbours that you consider voting, also know as
# > the prediction phase. The selection of first K indices from the sorted
# > array is
# >
# > O(K). Altogether, the overall time complexity of KNN is O(n*d + n*K).
# > The solution to a faster time is to either reduce ‘d’ by doing a
# > simple feature
# >
# > selection or by reducing ‘n’, where you try to quickly guess a number
# > of potential nearest neighbours (m) using K-D trees which will then
# > become
# >
# > O(md), meaning it will be less time consuming.
# 
# <table>
# <tbody>
# <tr class="odd">
# <td><blockquote>
# <p>Create PDF in your applications with the Pdfcrowd HTML to PDF API</p>
# </blockquote></td>
# <td>PDFCROWD</td>
# </tr>
# </tbody>
# </table>
