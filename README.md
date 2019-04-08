# What if all the libraries disappeared?
How might I reimplement some simple algorithms from scratch?


## Take-Home Challenge
Partitioning algorithms divide data objects into a number of partitions, subject to the following requirements: (1) each partition must contain at least one object, and (2) each object must belong to exactly one partition.

My go-to choice for partitioning around medoids (a.k.a. exemplar-based clustering) is k-medoids. k-medoids is similar to k-means, but the cluster centroids are restricted to objects (as opposed to the mean position of objects assigned to that cluster). k-medoids, however, is not implemented in our ML library du-jour (scikit-learn). So let's implement it ourselves!

Your challenge, if you choose to accept, is to write a Python implementation of the k-medoids algorithm. The objectives are threefold:
1. Demonstrate your ability to write clean code that is robust, modular and well structured.
2. Demonstrate your knowledge of a classic machine learning algorithm, including the math that makes it possible.
3. Demonstrate software engineering best practices including self-documenting code, test-driven development, atomic commits and version control.

Ground rules:
1. Do not copy source code from anywhere. We want to see how you would write this yourself, from scratch
2. Only use the Python standard library and the canonical computational library numpy. Please refrain from using specialized ML libraries such as scikit-learn for this exercise.
3. Timebox your efforts to a maximum of 6 hours (e.g. a typical work day after setting aside time for standup, code reviews, planning/retrospectives and other responsibilities.)


The result of your efforts should be a pull-request to this repository.
The code should run, but don't worry about fancy optimizations, edge cases, bells or whistles. Naive defaults and simplified parametrization are fine.

Have fun!


## run the tests
Checkout the source code and run the tests.
```
python -m unittest discover
```
