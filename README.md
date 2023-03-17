# Application of Machine Learning Techniques for UX Assessment and Enhancement of e-Learning Platforms

# Abstract 
E-Learning is a varied sector, expected to grow by USD 147.7 billion between 2021-2025, catering to a variety of groups. Students and professionals alike can now learn remotely and at their speed in previously inconceivable ways. The average person spent 6 hours and 49 minutes online in the first quarter of 2022. This trend has been increasing for the last few decades and is likely to continue. As technology advances and website visitors become more difficult to impress, assessing the usability and incorporating the best user experience into your Instructional Design is not only important but critical, for any industry based on it. Given the online nature of the interaction, it is relatively easier to collect data about user engagement and experience, which allows for a data-driven approach to the problem. With the use of ML algorithms, fantastic patterns can be spotted in these data to deliver personalised content, track and assist individual performance, and assess the desirability of the platform and factors affecting them to gain value. With this paper, we are going to explore four classic ML techniques, namely: KNN, Logistic Regression, SVM, and K-means Clustering trying to achieve the above-mentioned goals. The paper intends to do so through a brief literature review of the research on Machine Learning algorithms used in the industry pertinent to the tasks mentioned, and also the application of KNN, Logistic Regression and SVM on Kalboard 360 online-learning data containing 480 records and 16 features.

# Introduction
This paper focuses on a thorough review of the machine learning algorithms that have been used in the e-learning industry over the last ten years to measure usability and improve the user experience.
E-learning, or distance learning, is a formally defined learning framework that can be completed independently through electronic correspondence. The e-learning market has enormous potential. The global e-learning market is expected to grow by
147.7 billion US dollars between 2021 and 2025, at a compound annual growth rate (CAGR) of 16%. From corporate and SMB workers to K-12 and college students, almost everyone is a potential consumer, which drives interest in online learning and the need for a standard E-learning interface configuration [2]. This must include an intelligent picture, exciting components like chatbots, personalised content and feedback, and efficient ways for monitoring student performance.
Since the interaction occurs online, data collection about user engagement and experience is easy, allowing for a data-driven approach to the problem. For this purpose, advanced analytics have been made available to tutors and students using various machine-learning algorithms. Assessing a student can be a time- consuming task for tutors, but AI simplifies the analysis of massive amounts of data and the extraction of relevant information, such as strengths, weaknesses, and performance, providing tutors with incredibly useful insights. In addition, incredible patterns are discovered using machine learning algorithms to provide customized content that is tailored to each student's needs, interests, and level of understanding. This makes the platform more appealing and makes it more likely that people will use it for a long time. This will lead to the long-term success of the platform.
As a result, this industry would benefit greatly from a thorough examination of machine learning algorithms to select the best one for answering the following questions.
1. The first question is about tracking student performance to provide assistance quickly and efficiently.
2. The second question is about assessing students' knowledge levels on a platform that delivers content tailored to each individual's cognitive level.
3. Following that, an examination of the recommendation systems used to deliver this content to various individuals is conducted.
4. Finally, we have the factors influencing an e-learning platform's acceptance and their analysis for developing usability judgement metrics.
This paper discusses how K-Nearest Neighbour (KNN), Support Vector Machine (SVM), Clustering, and Logistic Regression can be used to answer these types of questions. The paper briefly reviews all four approaches and their current applications before applying some of them to the Kalboard 360 dataset to answer some of the above questions and then use the accuracy rate metric to compare the results from different methods and conclude.

# Methodology
# Data Preprocessing for Kalboard 360
For data processing, firstly, look at the correlation between numerical features to get basic intuitions. Secondly, one-hot encoding of the categorical features makes it easier to be analysed. Thirdly, we can select the important features by applying supervised learning methods to build the prediction models and using feature ranking functions provided by some models. One limitation of our approach is that sequential encoding (e.g.: 1,2,3) may distort the algorithms which rely on the calculation of distances such as KNN.
# Techniques for Parent school satisfaction prediction
We will verify three supervised learning methods that are mentioned in our literature reviews to address this problem Here is the methodology descriptions of them:
# KNeighborsClassifier
The KNeighborsClassifier(KNC) is an example of a classifier that makes use of k- nearest-neighbour techniques. The majority of a node's neighbours will determine its classification. These nodes' classes are determined by the class that is most frequent among their k nearest neighbours. Tuning n neighbours, weights, and metrics are typical hyperparameters for KNC. The size of K is equal to N neighbours. Uniform weighting assigns equal importance to each neighbouring point and distance weighting gives more weight to those points that are physically closer to the centre of the network. The term "metric" is used to describe the various systems used to calculate distance, such as the Minkowski, Euclidean, and Manhattan systems.
# LogisticRegression
Logistic Regression is most frequently applied to "binary classification" issues. However, it may also be utilized for multi- class classification using particular techniques such as one-vs.-rest or cross- entropy loss. Typical LR hyperparameters include penalty, C, solver, and max iter. The purpose of the penalty is to lessen the overfitting issue; however, some penalties may not be compatible with all solvers. C controls the severity of the punishment. . olver is the method of optimization. Max_iter limits the number of solver iterations.
7
# Support Vector Classifier
The SVC classifier utilizes the Support Vector Machine (SVM) algorithm. SVM's mechanism is to identify the superplane that best differentiates between two classes. SVM approaches multi-class classification similarly to logistic regression, such as one-versus- rest or one-versus-one. By utilizing kernel functions, SVM can also be expanded to address nonlinear classification problems. To tune the SVC hyperparameters, C, gamma, and kernel must be considered. C is still inversely proportional to the regularisation's strength. Gamma refers to the coefficient of the kernel. Kernel identifies the type of kernel utilized by the method.
# Hypertuning parameters for optimization
With respect to the above methodology, a gridsearch was used to find the best hyperparameters. A 5-fold cross validation was used in this search process.
# KNN
For KNN, the following hyperparameters were tried: n_neighbors (5,7,10,15), weights (uniform, distance), metric (cityblock, cosine, euclidean). It was found that the best hyperparameters were 15 n_neighbours, a uniform weighting for distances, and using the cityblock method for distance measurements.
# Logistic Regression
For Logistic Regression, the following hyperparameters were tried:C (0.9,1,1.1), penalty (uniform, distance), solver (lbfgs, liblinear, newton-cg). It was found that the best hyperparameters were a C value of 1.1, a L1 penalty type, and the liblinear solver.
# SVM
For SVM, the following hyperparameters were tried: C (0.9,1,1.1), gamma (scale, auto), kernel (linear, poly, rbf, sigmoid). It was found that the best hyperparameters were a C value of 0.9, scaled for gamma, and the linear kernel.
# Ensemble model for student categorization
We perform the student categorization by classifying them into three categories, i.e Low, Medium and High using Logistic Regression, Support Vector Machine and KNN algorithms and analyse the efficiency of the algorithm by dividing the dataset into train and test datasets. The hyperparameters are kept same as in the previous problem. Furthermore, we also look at an ensemble of the three models for the same problem. For this, we refer to the Kaggle notebook by Mohd. Ashfaq. We split the data into 20% test and 80% train datasets.

LR Confusion matrix for 20 % Kalboard 360 test data for class prediction

![image](https://user-images.githubusercontent.com/127405318/225853775-eebcb83e-07f5-4857-8ed8-bd10ccc1fbc9.png)

SVC Confusion matrix for 20 % Kalboard 360 test data for class prediction

![image](https://user-images.githubusercontent.com/127405318/225853822-c436af65-5197-4e5c-a2f5-8ded73249f00.png)

KNN Confusion matrix for 20 % Kalboard 360 test data for class prediction

![image](https://user-images.githubusercontent.com/127405318/225853851-48e0f316-5d66-4b19-b9b2-0d83c88d2b4d.png)

Ensemble model Confusion matrix for 20 % Kalboard 360 test data for class prediction

![image](https://user-images.githubusercontent.com/127405318/225853889-27f003c4-3d8e-4550-962a-81c00b76e280.png)

# Results
For the first problem of prediction of parent school satisfaction, we have the mean test scores from the cross validation for the top parameters are 0.677 for KNN, 1.00 for SVM, 1.00 for logistic regression.
For the student categorization problem, we have an accuracy of 71.88% for KNN model, 98.96% for the logistic regression, and 100% for the Support Vector based Classifier. Also, we have an accuracy of 98.96% for an ensemble of the three.
