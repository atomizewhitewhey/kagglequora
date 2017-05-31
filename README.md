# kagglequora
Quora question pairs text analytics as part of Kaggle Competition

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions.
Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make 
writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a 
better experience to active seekers and writers, and offer more value to both of these groups in the long term.

The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same meaning. 

There are two main files aside from the datasets provided.

1. kagglequora.py

As the starter code, it is just used to explore a little bit of the datasets provided to get a better understanding of what the data 
entails. It will be eventually replaced with the code that will do feature engineering and extract features from the dataset to allow 
machine learning algorithms to be implemented such that we can classify the questions into 'same' question pairs.

2. QuoraQuestionPairs.ipynb

Using the starter code as a base, this notebook is used to do an exploratory analysis of the datasets provided, to gain a 
better understanding of which features can be extracted from the datasets and used to implement machine learning algorithms.
