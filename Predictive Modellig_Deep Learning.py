#!/usr/bin/env python
# coding: utf-8

# ## Predictive Modelling with Deep Learning
# 
# <b>Machine Learning & Computational Intelligence</b>
# <b>Building predictive models to predict the target variable of the dataset</b><br>

# ### Import libraries

# In[1]:


import pandas as pd # Import pandas library for data manipulation and analysis
import numpy as np # Import numpy library for array operations
import matplotlib.pyplot as plt # Import pyplot module for creating visualizations
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

get_ipython().run_line_magic('config', 'Completer.use_jedi=False')


# ### Load the dataset

# In[3]:


# Read the CSV file into pandas DataFrame
data = pd.read_csv('risk_factors.csv')
data # Display the first few rows (5) of the dataset


# ### Data Cleaning

# It can be seen that this dataset uses '?' to represent null values. Therefore, we will replace '?' with 'NaN' to ease the processing.

# In[4]:


# Replace '?' with null
data = data.replace('?', np.nan)


# After replacing with null, we count how many null values there are in each column of the dataset.

# In[5]:


data.isnull().sum()


# From the above step, it can be observed that the parameters `STD: Time since the first diagnosis` and `STD: Time since last diagnoses` had many null values. Replacing these null values would make the classifier useless. Hence, these two features were dropped for each training, validation and test set.

# In[6]:


# Remove columns in training set
data = data.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)


# In[7]:


# Check if columns were removed
data.isnull().sum()


# In[8]:


# Check the types of data for each column
data.info()


# Since there are multiple columns that have the 'object' data type, the values in the columns for each set were converted to numerical values to ease the processing.

# In[9]:


# Convert all features to numeric values

data = data.apply(pd.to_numeric)


# The columns in the dataset have both categorical and numerical attributes. Hence, we separate the categorical and numerical attributes for each set as they both require different processes/formulas in imputing the null values.

# In[10]:


# Separate the categorical and numerical values for each set

data_cat = ['Smokes','Hormonal Contraceptives','IUD','STDs','STDs:condylomatosis',
            'STDs:cervical condylomatosis','STDs:vaginal condylomatosis',
            'STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease',
            'STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV',
               'STDs:Hepatitis B','STDs:HPV','Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Schiller','Citology']
data_num = ['Age','Number of sexual partners','First sexual intercourse','Num of pregnancies',
            'Smokes (years)','Smokes (packs/year)','Hormonal Contraceptives (years)',
            'IUD (years)','STDs (number)','STDs: Number of diagnosis']


# <b>Descriptive statistics</b> was used to replace the missing values. The most typical metrics for this task are mean, median and mode. The median was used to replaced numerical attributes, and categorical attributes were replaced by the mode. Mean value imputation was avoided since they highly influence the extreme values/outliers in the data.

# In[11]:


# Replace null values of numerical attributes with the median
# Replace null values of categorical attributes with the mode

for feature in data[data_num]:
    data[feature].fillna((data[feature].median()), inplace=True)
    
for feature in data[data_cat]:
    data[feature].fillna((data[feature].mode()[0]), inplace=True)


# In[12]:


# Check null values again for dataset
data.isnull().sum()


# ### Splitting Dataset & Data Preprocessing

# <b>Step 1: Feature Scaling/Normalisation</b>
#     
# Feature scaling is a method used to normalize the range of independent variables or features of data. In this process, standard scalar was used to normalise the data. This is because it uses the standard normal distribution. All the means of the attributes are made zero, and the variance is scaled to one. As our dataset has both numerical and categorical variables, we will only normalise the numerical attributes.

# In[13]:


features = data.drop(['Biopsy'], axis=1)
X = features
y = data['Biopsy']


# In[14]:


# Encode categorical variables

for col in data_cat:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])


# In[15]:


# Perform feature scaling on numerical columns

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[data_num] = scaler.fit_transform(X[data_num])


# In[16]:


from sklearn.model_selection import train_test_split

# Split the dataset into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the training set into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[17]:


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# In[18]:


# Insert the standardised values in a dataframe for each set

saved_cols = X_train.columns # Save column names for later use
X_train = pd.DataFrame(X_train, columns = saved_cols)
X_val = pd.DataFrame(X_val, columns = saved_cols)
X_test = pd.DataFrame(X_test, columns = saved_cols)

X_train


# <b>Step 2: Feature Selection using Pearson’s correlation technique</b>
# 
# Pearson’s correlation feature selection technique was utilized to find redundant features. This feature selection technique <b>compares the degree of association</b> among all variables. When there is a high correlation between two independent attributes, one of these attributes can be removed since both features contribute the same to the ML model. By observing the diagonal values, any variable that is directly correlated to itself will show a positive correlation. Therefore, age has a positive correlation, which is one, and so the diagonal should also be visible. The dark colour shows the near-zero correlation. This technique can only be used on numerical attributes.

# In[19]:


# Find correlation
import seaborn as sns

X_train_num = X_train[data_num]
corr = X_train_num.corr()

plt.figure(figsize=(20, 8))
heatmap = sns.heatmap(corr, vmin=0, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')


# If the correlation value is above 0.8, we consider that is highly correlated hence the feature is removed. From the heatmap, we can observe that the feature `STDs (number)` has a value of <b>0.92</b> so this feature was dropped.

# In[20]:


# Get upper triangle of correlation matrix
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

# Find features with correlation greater than 0.80
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

# Drop highly correlated features
X_train.drop(to_drop, axis=1, inplace=True)

X_val.drop(to_drop, axis=1, inplace=True)

X_test.drop(to_drop, axis=1, inplace=True)


# In[21]:


X_train


# <b>Step 2: Feature Selection using the feature importance with Tree Based Classifier</b>
# 
# The importance of each feature is determined by using a Tree-Based Classifier, namely the Extra Trees Classifier. The normalized total reduction in the mathematical criteria used in the decision of the feature of the split is computed. This value is called the Gini Importance of the feature. Based on the previous step, we have already selected the best features for numerical attributes. For this step, we will use this technique to select the best features for categorical attributes.

# In[22]:


from sklearn.ensemble import ExtraTreesClassifier

X_train_cat = X_train[data_cat]

model = ExtraTreesClassifier()
model.fit(X_train_cat,y_train)
print(model.feature_importances_)

# Set a threshold for feature importance
threshold = 0.0
importances = model.feature_importances_

# Create a mask of features to keep
mask = importances <= threshold

# Get the remaining feature names
no_importance = X_train_cat.columns[mask]

# Filter the features based on the mask
X_dropped = X[no_importance]

# A graph of feature importances is plotted for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X_train_cat.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.figure(figsize=(10, 8))
plt.show()


# From the graph above, we can observe that there are four features namely `STDs:cervical condylomatosis`,`STDs:vaginal condylomatosis`, `STDs:molluscum contagiosum` and `STDs:pelvic inflammatory disease` that show no importance hence these features are dropped.

# In[23]:


# Drop features with 0 importance

X_train.drop(X_dropped, axis=1, inplace=True)
X_val.drop(X_dropped, axis=1, inplace=True)
X_test.drop(X_dropped, axis=1, inplace=True)


# In[24]:


X_train


# ### Data modeling
# ______________________________________________________________________________________
# <b>Model 1: Neural Network</b>

# In[25]:


X_train = X_train.values
X_val = X_val.values
X_test = X_test.values
y_train = y_train.values
y_val = y_val.values
y_test = y_test.values


# In[26]:


# Define the loss function using Binary Cross Entropy

loss_fn = tf.keras.losses.BinaryCrossentropy()


# We create a neural network with <b>dropout and L2 regularizations</b>. A dropout rate of 0.1 means that, on each training iteration, each neuron in the specified layer has a 10% chance of being deactivated or dropped out. This is to avoid overfitting of the data.

# In[27]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

l2 = tf.keras.regularizers.l2(l=0.1) 

model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu', dtype='float64', kernel_regularizer='l2'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid', dtype='float64',kernel_regularizer='l2')])

model.compile(optimizer='adam', # Optimizer to use to train the model
              loss=loss_fn, # specify the loss function
              metrics=['accuracy']) # Metric to use to monitor the model training


# In[28]:


# Print the summary of the model

model.summary()


# A <b>callback</b> is an object that can perform actions at various stages of training (e.g. at the start or end of an epoch, before or after a single batch, etc). Here we implement <b>early stopping regularization</b> with the parameter of patience equal to 3 to stop the training when the validation loss is not improving. This is also to avoid overfitting the data. We saved the best model based on the validation loss.

# In[29]:


callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
                tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/',
                monitor='val_loss', save_weights_only=True)]

history = model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=100, callbacks=callbacks)


# We plotted graph to compare the model's accuracy and loss between train set and validation set to observe if our data is overfitting or not. From the graphs, we can see that there is only a slight difference between both sets. Therefore, we can say that our data is not overfitting.

# In[30]:


# Plot graph to coompare accuracy between train set and validation set

from matplotlib import pyplot as plt

acc_train = history.history['accuracy']
acc_vald = history.history['val_accuracy']

plt.plot(acc_train)
plt.plot(acc_vald)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper right')
plt.show()


# In[31]:


from matplotlib import pyplot as plt

loss_train = history.history['loss']
loss_vald = history.history['val_loss']

plt.plot(loss_train)
plt.plot(loss_vald)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper right')
plt.show()


# In[32]:


# Predict the test set using the model

y_pred_train = model.predict(X_train)
y_pred_train = np.round(y_pred_train)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)


# The F1 scores between the training and testing datasets are compared to prevent overfitting. In this case, overfitting will happen if the F1 score of training dataset is 1.0 or its value is too high as compared to the testing dataset. As we can see below, there is only a slight difference between the f1 scores. Therefore, we can once again say that our data is not overfitting when using this NN model.

# In[33]:


# Compare the F1 scores between the training and testing datasets
from sklearn import metrics

test_f1_score = metrics.f1_score(y_test, y_pred)
train_f1_score = metrics.f1_score(y_train, y_pred_train)

print('F1 Score (test):', test_f1_score)
print('F1 Score (train):', train_f1_score)


# In[34]:


# Print scores for the model

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

print('Neural Networks')
print('----------------------')
print('F1 Score: %.3f' % f1_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('Specificity: %.3f' % recall_score(y_test, y_pred, pos_label = 0))


# In[35]:


from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(conf_matrix, cmap='PuRd')

# Customize the colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=12)

# Add text annotations to each cell with darker font color
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        text_color = 'white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black'
        ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color=text_color, size=15)

# Set axis labels and title
plt.xlabel('Predictions', fontsize=14)
plt.ylabel('Actuals', fontsize=14)
plt.title('Confusion Matrix: Neural Network', fontsize=16, y=1.05)

# Adjust spacing between the title and the confusion matrix
plt.subplots_adjust(top=0.85)

plt.show()


# ### Data modeling
# ______________________________________________________________________________________
# <b>Model 2: Hybrid Intelligent System - Genetic Algorithm & Decision Tree</b>

# For our second model, we implemented a <b>Hybrid Intelligent System </b>. By utilizing this hybrid strategy, the classification strength of the <b>Decision Tree</b> model is combined with the search and optimisation powers of the <b>Genetic Algorithm</b>. 

# We first display the accuracy of the Decision Tree node that is not fine tuned or optimized.

# In[36]:


# Importing the required libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create the non-optimized decision tree classifier
clf = DecisionTreeClassifier()

# Train the non-optimized decision tree classifier
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_dt = clf.predict(X_test)

# Calculate and display the accuracy of the non-optimized model
accuracy = accuracy_score(y_test, y_pred_dt)
print("Accuracy of Non-Optmised Decision Tree Model:", accuracy)


# In[37]:


# Print scores for the non-optimized model

print('Decision Tree')
print('----------------------')
print('F1 Score: %.3f' % f1_score(y_test, y_pred_dt))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_dt))
print('Precision: %.3f' % precision_score(y_test, y_pred_dt))
print('Recall: %.3f' % recall_score(y_test, y_pred_dt))
print('Specificity: %.3f' % recall_score(y_test, y_pred_dt, pos_label = 0))


# In[38]:


get_ipython().system('pip install deap')


# In[39]:


#Import the necessary libraries for the genetic algorithm (GA) and decision tree optimization:
import random
import numpy as np
from deap import creator, base, tools, algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Define  the fitness evaluation function for the GA
def evaluate_dt(individual):
    # Decode the chromosome to obtain the decision tree parameters
    max_depth = int(individual[0]) # Set the max_depth parameter of the decision tree model
    min_samples_split = int(1 + (len(X_train) - 1) * individual[1]) # Calculate the min_samples_split parameter based on the length of X_train
    
    # Check if any parameter value is invalid
    if any(param <= 0 for param in [max_depth, min_samples_split]):
        return float('-inf'),  # Return a very low fitness for invalid individuals
    
    # Create a decision tree classifier with the parameters
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    
    # Fit the classifier on the training data
    clf.fit(X_train, y_train)
    
    # Make predictions on the validation data
    y_pred = clf.predict(X_val)
    
    # Calculate the accuracy as the fitness value
    accuracy = accuracy_score(y_val, y_pred)
    
    # Return the accuracy as the fitness value (maximize accuracy)
    return accuracy,


# This function evaluates the fitness of an individual in the GA population. It decodes the chromosome of the individual to obtain the decision tree parameters, creates a decision tree classifier, fits it on the training data, and calculates the accuracy as the fitness value.

# In[40]:


# Set up the GA parameters
# These variables define the population size, the number of generations, 
# and the probabilities for crossover and mutation in the GA
population_size = 80
num_generations = 20
crossover_probability = 0.8
mutation_probability = 0.2

# Create individual and population
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Fitness class called "FitnessMax" inheriting from base with a single weight of 1.0
creator.create("Individual", list, fitness=creator.FitnessMax) # Individual class called "Individual" as a list with a fitness attribute of "FitnessMax"
toolbox = base.Toolbox() # Create a toolbox instance used to hold the functions and tools needed for the genetic algorithm.
toolbox.register("attribute", np.random.uniform, low=1, high=10) # Register an attribute function that generates random values from a uniform distribution between 1 and 10
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=2)  

# Register an individual function that initializes an instance of "Individual" using the "attribute" function,
# with a length of 2 (num of parameters)
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # Register a population function that initializes a population as a list of individuals using the "individual" function

# Define genetic operators of the crossover, mutation, selection, and evaluation functions for the GA.
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=10, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_dt)

# Run the GA
population = toolbox.population(n=population_size)

# Evaluate the initial population
fitnesses = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

# Set up early stopping parameters
stopping_generations = 5  # Number of generations without improvement to stop the optimization
best_fitness = float('-inf')  # Initialize the best fitness value
stopping_counter = 0  # Counter to keep track of the number of generations without improvement

# Begin the evolution
# These lines select parents from the current population, clone them to create offspring, 
# and apply crossover and mutation operations with certain probabilities.
for generation in range(num_generations):
    print("-- Generation", generation, "--")
    
    # Select the next generation's parents
    parents = toolbox.select(population, len(population))
    
    # Clone the selected parents
    offspring = [toolbox.clone(ind) for ind in parents]

    # Apply crossover and mutation
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < crossover_probability:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutation_probability:
            toolbox.mutate(mutant)
            del mutant.fitness.values
            
    # Evaluate the offsprings
    invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_individuals)
    for ind, fit in zip(invalid_individuals, fitnesses):
        ind.fitness.values = fit

    # Replace the current population with the offspring updateing the population with the new generation
    population[:] = offspring

    # Get the best individual
    best_individual = tools.selBest(population, k=1)[0]

    # Compare the fitness of the current best individual with the previously best fitness. 
    if best_individual.fitness.values[0] > best_fitness:
        best_fitness = best_individual.fitness.values[0] # If the fitness has improved, the best fitness is updated
        stopping_counter = 0  #  stopping counter is reset to 0.
    else:
        stopping_counter += 1  # Otherwise, increment the counter

    # Check if early stopping criterion is met
    # This condition checks if the stopping counter has reached or exceeded the predefined number of stopping generations. 
    # If so, print a message indicating that the early stopping criterion has been met and breaks out of the optimization loop.
    if stopping_counter >= stopping_generations:
        print("Early stopping criterion met. Optimization stopped.")
        break


# We will now determine the optimized parameters of Decision Tree model and determine the accuracy achieved by implementimg the optimized parameters.

# In[41]:


# Extract decision tree parameters from the best individual
best_max_depth = int(best_individual[0])
best_min_samples_split = int(1 + (len(X_train) - 1) * best_individual[1])

# Print the results
print("Best (Optimized) Decision Tree parameters:")
print("Max Depth =", best_max_depth)
print("Min Samples Split =", best_min_samples_split)

# Create the optimized decision tree classifier with the best parameters
best_clf = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_split=best_min_samples_split)

# Fit the optimized classifier on the training data
best_clf.fit(X_train, y_train)

# Make predictions on the test data using the optimized classifier
y_pred_opt = best_clf.predict(X_test)
y_pred_opt_train = best_clf.predict(X_train) # Make predictions on the train data using the optimized classifier 
                                             # to check f1 score to determine existance of overfitting)

# Calculate the accuracy of the optimized classifier
accuracy_opt = accuracy_score(y_test, y_pred_opt)

# Print the optimized Decision Tree accuracy
print("Accuracy of Optimized Decision Tree Model:", accuracy_opt)


# A bar graph is plotted to depict the <b>difference in accuracies</b> achieved by Non-Optimized DT model and Optimized DT model.

# In[42]:


import matplotlib.pyplot as plt
import numpy as np

# Plotting the accuracy comparison graph between Non-Optimized and Optimized DT models
labels = ['Non-optimized', 'Optimized']
accuracy_values = [accuracy, accuracy_opt]
colors = plt.cm.get_cmap('RdYlBu', len(labels))(np.arange(len(labels)))

plt.bar(labels, accuracy_values, color=colors)
plt.title('Accuracy Comparison: Decision Tree')
plt.xlabel('Code Version')
plt.ylabel('Accuracy')
plt.ylim(0.85, 1.0)  # Set the y-axis limits between 0.85 and 1.0
plt.yticks([0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 0.98, 1.0])
plt.show()


# Compare the performance metrics and analysis of the Not Optimized Decision Tree model and the Optimized Decision Tree. The Optimized Decision Tree model shows<b> an improvement in its performance</b> compared to a Non Optimized model.

# In[43]:


from sklearn.metrics import f1_score,precision_score,recall_score, confusion_matrix, classification_report

#Print  performance metrics and analysis of the Not Optimized Decision Tree model 

print('\nNot Optimized Decision Tree')
print('----------------------')
print('F1 Score: %.3f' % f1_score(y_test, y_pred_dt))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_dt))
print('Precision: %.3f' % precision_score(y_test, y_pred_dt))
print('Recall: %.3f' % recall_score(y_test, y_pred_dt))
print('Specificity: %.3f' % recall_score(y_test, y_pred_dt, pos_label = 0))

print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

#Print  performance metrics and analysis of the Optimized Decision Tree model 

print('\n\nOptimized Decision Tree')
print('----------------------')
print('F1 Score: %.3f' % f1_score(y_test, y_pred_opt))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_opt))
print('Precision: %.3f' % precision_score(y_test, y_pred_opt))
print('Recall: %.3f' % recall_score(y_test, y_pred_opt))
print('Specificity: %.3f' % recall_score(y_test, y_pred_opt, pos_label = 0))

print(confusion_matrix(y_test, y_pred_opt))
print(classification_report(y_test, y_pred_opt))


# As mentioned, the F1 scores between the training and testing datasets of the optimized model are compared to prevent overfitting. In this case, overfitting will happen if the F1 score of training dataset is 1.0 or its value is too high as compared to the testing dataset. As we can see below, there is only a slight difference between the f1 scores. Therefore, we can conclude that our data is not overfitting when using Decision Tree model optimized with Genetic Algorithm.

# In[44]:


from sklearn import metrics

test_f1_score = metrics.f1_score(y_test, y_pred_opt)
train_f1_score = metrics.f1_score(y_train, y_pred_opt_train)

print('F1 Score (test):', test_f1_score)
print('F1 Score (train):', train_f1_score)


# In[45]:


# Print scores for the optimised model

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

print('Optimized Decision Tree')
print('----------------------')
print('F1 Score: %.3f' % f1_score(y_test, y_pred_opt))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_opt))
print('Precision: %.3f' % precision_score(y_test, y_pred_opt))
print('Recall: %.3f' % recall_score(y_test, y_pred_opt))
print('Specificity: %.3f' % recall_score(y_test, y_pred_opt, pos_label = 0))


# In[46]:


# Print the confusion matrix using Matplotlib 
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred_dt)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(conf_matrix, cmap='BuPu')

# Customize the colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=12)

# Add text annotations to each cell 
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        text_color = 'white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black'
        ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color=text_color, size=15)

# Set axis labels and title
plt.xlabel('Predictions', fontsize=14)
plt.ylabel('Actuals', fontsize=14)
plt.title('Confusion Matrix: Not Optimized Decision Tree', fontsize=16, y=1.05)

# Adjust spacing between the title and the confusion matrix
plt.subplots_adjust(top=0.85)

plt.show()


# In[47]:


# Print the confusion matrix using Matplotlib
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred_opt)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(conf_matrix, cmap='Reds')

# Customize the colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=12)

# Add text annotations to each cell r
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        text_color = 'white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black'
        ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color=text_color, size=15)

# Set axis labels and title
plt.xlabel('Predictions', fontsize=14)
plt.ylabel('Actuals', fontsize=14)
plt.title('Confusion Matrix: Decision Tree + Genetic Algorithm', fontsize=16, y=1.05)

# Adjust spacing between the title and the confusion matrix
plt.subplots_adjust(top=0.85)

plt.show()

