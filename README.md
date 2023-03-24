# Jihee You's Computer Science Portfolio
Here are some interesting topics I worked on:

### [Project 1](https://github.com/jiheeyy/jiheeyy.github.io/tree/main/project/1)
Hoffmann research - facial landmark linear model

### Project 2([https://github.com/jiheeyy/charactour_/tree/main/project/3](https://github.com/jiheeyy/CharacTour-Non-Confidential)
CharacTour - film recommendation based on user and character personality matches

### [Project 3](https://github.com/jiheeyy/jiheeyy.github.io/tree/main/project/3)
I implmented 2d convolution, gaussian filtering, smoothing and downsampling, and sobel gradients using only numpy. Below is my implementation of sobel gradients (dy) of a koala picture.

![](https://github.com/jiheeyy/jiheeyy.github.io/blob/main/image/dy.png)

### Project 4
To be added!

### [Project 5](https://github.com/jiheeyy/jiheeyy.github.io/tree/main/project/5)
Referencing the [CycleGAN video](https://youtu.be/4LktBHGCNfw), I wrote a CycleGAN code that transforms human faces into simpsons and vice versa with Pytorch. The training and testing were performed on 1000 simpson faces (https://www.kaggle.com/datasets/kostastokis/simpsons-faces) and 1000 GAN-generated human faces (https://github.com/jcpeterson/omi/tree/main/images). After 100~ epochs of training, the image transformations looked plausible (although simpsons -> humans were still very creepy).

![](https://github.com/jiheeyy/jiheeyy.github.io/blob/main/image/Screen%20Shot%202023-03-24%20at%2010.56.52%20AM.png)
![](https://github.com/jiheeyy/jiheeyy.github.io/blob/main/image/Screen%20Shot%202023-03-24%20at%2010.57.24%20AM.png)

### [Project 6](https://github.com/jiheeyy/jiheeyy.github.io/tree/main/project/6)
Part 1: CDC IL life expectancy data
- Analysis on a linear regression on CDC IL life expectancy data suggested that the percentage of black or African American population was an influential factor in determining positive and negative outliers.
- Components and component plus residual plots displayed that predictors '% households that earn $75000 or more' or '% households without social security income' show nonlinear relationship to life expectancy.

Part 2: Household firearm ownership scores (https://www.rand.org/pubs/tools/TL354.html) and rates of mortality by firearms from the CDC
- There was a potential non-linear relationship the two variables of interest since spline regression outperformed linear regression

Part 3: Food Access Research Atlas data
- Logistic regression on predictors 'MedianFamilyIncome', 'PrcntSNAP', 'PrcntAA', 'PrcntNoVehicle', 'PrcntHispanic' performed better on urban tracts compared to non-urban ones.

### [Project 7](https://github.com/jiheeyy/jiheeyy.github.io/tree/main/project/7)
Using Markov chain Monte Carlo methods, I deciphered a paragraph of substitution cipher text  "m it vbeh yjmbl. qbl lgb tfwlgo  ... ". Calculating the log score of each decryption based on popular bigram count in English (Google Corpus Data), I was able to identify the cipher text as a paragraph from chapter 12 of the novel All Quiet on the Western Front.

With similar methods, I also solved the knapsack problem (trying to pack a set of items, with given values and weights, into a knapsack with a maximum capacity), and I generated five or six-letter strings that sound like English names. This project references code presented in UChicago DATA 21300 (Models in Data Science).

![](https://github.com/jiheeyy/jiheeyy.github.io/blob/main/image/mc%2Bo.png)
![](https://github.com/jiheeyy/jiheeyy.github.io/blob/main/image/mc.png)
