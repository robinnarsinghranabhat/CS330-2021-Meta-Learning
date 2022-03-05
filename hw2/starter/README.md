## Notes on Prototypical Networks based Meta-Learning : 

- Just a K-means Clustering, But instead of calculating directly the **distance** between `set of test-images` and `classwise-mean-of-train-images`,
  use a neural-network that learn's to embed the raw pixels into a Feature-Space, and calculate distance in that space.
- Network is trained, such that, embedding it generates will minimize the distance between `similar-looking images`.


## Notes on Model-Agnostic Meta-learning : 

