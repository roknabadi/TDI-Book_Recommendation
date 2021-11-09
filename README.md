# Book Recommendation System
## By: Amir Momen-Roknabadi¶
### The Data Incubator - June 2021 Part Time Cohort¶
The aim of this project is to create a book recommendation system using deep learning. 
I downloaded the Amazon review data from UCSD Amazon's review data: http://deepyeti.ucsd.edu/jianmo/amazon/index.html. 
The file that I used is available at: http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Books.json.gz. 

One of the issues that I noticed in the current recommender systems is that the final test is performed on the same users as the training set. 
This means that the system will not be able to recommed a book to a user that it has not seen before. 
To overcome this, I divided the database into two parts. These two parts have distinct users, but share the same books. 
I used the first part to train, validate and test a DNN model. This model achieved an accuracy of 73% percent. 
Then I used the second part to perform a transfer learning and recommend books to the users the system has not seen before. This achieved a 70% accuracy.

For convenience, I have divided up the notebook in four parts. The first part focuses on preprocessing and a first pass at training, the second part is on hyperparameter optimization. The third part is on final training using optimized parameters. The fourth part is on transfer learning.










