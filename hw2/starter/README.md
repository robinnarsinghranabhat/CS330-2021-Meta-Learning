## Notes on Prototypical Networks based Meta-Learning : 

- Just a K-means Clustering, But instead of calculating directly the **distance** between `set of test-images` and `classwise-mean-of-train-images`,
  use a neural-network that learn's to embed the raw pixels into a Feature-Space, and calculate distance in that space.
- Network is trained, such that, embedding it generates will minimize the distance between `similar-looking images`.


## Notes on Model-Agnostic Meta-learning : 

# NOTE : Make sure you are inside hw2/starter

### What is this network learning ? 
- Learn to adapt it's weights given the limited examples in the training set. 



### Problem 1 : Done in protonet.py

### Problem 2 :

#### 2.1
- Train on 5-way 5 Shot omniglot upto desired checkpoint (For me, it was 1400)
  - python protonet.py --num_way 5 --num_support 5

- Visualize the above training : 
  - `tensorboard --logdir="./logs/protonet/omniglot.way:5.support:5.query:15.lr:0.001.batch_size:16"`


#### 2.2 
- Test (on validate) on previously trained 5-Way 5-shot Model : 
 - `python protonet.py --test --log_dir "./logs/protonet/omniglot.way:5.support:5.query:15.lr:0.001.batch_size:16" --checkpoint_step 1400 --num_support 5  --num_way 5`

### Problem 3 : 
#### 3.1 :
#### 3.2 : Accuracy on both training and query set is increasing. So, model is training well

### Problem 43 :
- Train on 5-way 1-Shot Task (Assuming, you have also trained already at **step 2.1**)
  - `python protonet.py --num_way 5 --num_support 5`

- Compare `5-way 1-Shot Task` vs `5-way 5-shot` task
  -  `5-way 5-shot` Results :
      - Testing on tasks with composition num_way=5, num_support=5, num_query=15
      - Accuracy over 600 test tasks: mean 0.989, 95% confidence interval 0.001
  
  -  `5-way 1-shot` Results :
      - Execute using : `python protonet.py --test --log_dir "./logs/protonet/omniglot.way:5.support:1.query:15.lr:0.001.batch_size:16" --checkpoint_step 1400 --num_support 1  --num_way 5`
      - Testing on tasks with composition num_way=5, num_support=1, num_query=15
      - Accuracy over 600 test tasks: mean 0.964, 95% confidence interval 0.004
