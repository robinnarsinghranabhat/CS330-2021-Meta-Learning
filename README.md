# CS330-2021-Meta-Learning
Assignment solution to Stanford's CS330-2021 Meta-learning MOOC by Chelsea Finn

# General Training Proceduer for Meta-Learning Methods with Omniglot

- **Omniglot Dataset** : 1600+ Classes, just 20 examples for each class
- **Goal** : 

    Here we train learning algorithms, that can learn with very few examples.

    To achieve this, we introduce the concept of **training happening during the inference process as well** 
    
    We sample `N` classes with `M` examples for each (*Note : max val of `M` is just 20 for omniglot*) . Split M examples into train (`num_support`) and test (`num_query`) for each class. Say, sample 5 classes out of 1600 with 8 examples for each. Use 2 examples for training and rest of 6 for validation.
    
    Let's call this a `batch`.

    What the algorithm learns is, for each `batch`, learn to accurately predict the test set given the limited number of training data.
    It should be stressed that the algorithm doesn't remember the internal representation for those input images like in a normal MNIST-Classifier. 
    
    It learns a way to quickly associate similar images in train to similar images in train set. Hope that makes it less confusion.
