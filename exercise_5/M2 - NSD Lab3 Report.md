# M2 - NSD Lab3 Report

Name: Zhou Quan

Date: 2015-12-01

--

Codes can be seen here: [*[Source Code]*][1]

[1]: https://github.com/JoshPAT/network-structure-and-dynamics/tree/master/exercise_5

## 1 Preliminaries

### 1.1 Properties of the sample network

The characteristics of sample network are shown below:

```bash
# Degree
Numbers of degree 0: 0
Max Degree: 151
Min Degree: 1
Average Degree: 19.2058823529
# Density
Density: 0.08120808350
# Clustering
Approximations of average clustering coefficient: 0.41100000000
```

![flickr-test](figures/flickr-test.png)

### 1.2 Properties of the original networks

The characteristics of original network are shown below:

```bash
# Degree
Numbers of degree 0: 0
Max Degree: 12292
Min Degree: 1
Average Degree: 67.3930681654
# Density
Density: 0.00485663301
# Clustering
Approximations of average clustering coefficient: 0.23200000000
```

![flicker](figures/flickr.png)

### 1.3 Comments

1. The density of both networks are low. The density of sample network is much higher than the density of original network although sample network is part of original network. 
2. The clustering of both networks are high in terms of the density.  Considering the possible triangle is heavily constrained by the density, it is not surprising that sample network is denser than the original network. It is really interesting that clustering of original network is really high owing to its low density compared to clustering of sample network.
3. The degree distribution of original networks exactly follows power law, the degree distribution of sample networkand roughly follows power law. But Inverse Culmulative Degree Distribution of both networks have remarkable visible signs of hetergenous networks. So we can say that they are hetergenous network.
4. Both networks is a strong connected component since there is no node has zero link.

In conclusion, sample network may not be a good example of original network, but both networks have the baisc characteristics of real networks that we can proceed our analysis using these networks.
 
## 2 Evaluation

### 2.1 Evolution

Based on what we have implemented in my program, we have got the evolution shown as follow:

![flicker](figures/flickr-whole.png)

### 2.2 Comparison

The table below takes the random phase until it finds 0.1% of the existing links.

|          |   m' | % tested |% found| e | R |
|:-------| :------- | :----: | :---: | :---: | :---: | :---: | :---: |
| Random strategy (random 1000)   | 9842    | 1.04 | 1.05 |  0.006 | 1.01 |
| V_Random strategy (random 1000) |  17827  | 1.04 | 2.15 |  0.009 |  1.52 |
| Complete strategy (random 1000) |  211136 | 1.04 | 22.57| 0.150 | 25.43 |
| TBF strategy (random 1000)      | 64807    | 0.42| 6.92 | 0.054 | 18.49 |
| Combined strategy (random 1000) |  215944  |1.04 | 23.07| 0.137 |23.33 |

From the fisrt table, we can observe that:
1. Complete strategy have the highest efficiency, since its relative efficiency and normalized efficiency is the highest among all the strategies. 
2. The links that Combined strategy finds are the highest among among all the strategies. Although TBF strategy tests less time, it still has good relative efficiency and normalized efficiency. 
3. Relative efficiency of Random strategy is around 1, which is exactly what we have expected because it ratio of two random phase is 1. 
4. Relative efficiency of V_Random strategy is a bit higer than 1 for the reason that it also looks for clustering and in real network the clustering is high.


The figure and table below takes the random phase until it finds 0.15% of the existing links.

![flicker](figures/flickr-whole15.png)


| |   m' | % tested |% found| e | R |
|:-------| :------- | :----: | :---: | :---: | :---: | :---: | :---: |
| Random strategy (random 1500)   |  9432      | 1.04 | 1.01  |  0.006 |  0.98  |
| V_Random strategy (random 1500) |  18212    | 1.04 | 1.94  |  0.008 |  1.43  |
| Complete strategy (random 1500) |  205329   | 1.04 | 21.95 |  0.136 |  23.12  |
| TBF strategy (random 1500)      |  110055    | 0.39 | 11.76  |  0.083 |  17.959 |
| Combined strategy (random 1500) |  201382   |1.04  | 21.53 |  0.110 | 18.76  |

1. Not only from the graph but also from the table, TBF strategy phase stays longer since there are more pairs of two nodes that have degree bigger than 1 to test before TBF strategy phase stops.
2. Overall, relative efficiency and normalized efficiency of all the strategies are lower than before. The reason for this is that random phase have to test more before it finds 0.15%.

### 2.3 Qualitative assessment

| |   m' | density | avg deg| max deg | cc | tr
|:-------| :------- | :----: | :---: | :---: | :---: | :---: | :---: | :---: |
| Flickr                           |  935281    | 0.004 | 67.3  |  12292 |  0.231  | 0.066 |
| Random strategy (random 1000)    |  9842      | 0.000 | 1.3 |  142    |  0.001  |  0.001 |
| V_Random strategy (random 1000)  |  17827     | 0.000 | 1.94  |  363 |  0.038  | 0.067 |
| Complete strategy (random 1000)  |  211136    | 0.0011 | 15.32|  1023 |  0.547  | 0.021|
| TBF strategy (random 1000)       |  64807     | 0.0003 | 4.99  |  0.083 |  0.022| 0.22|
| Combined strategy (random 1000)  |  215944    | 0.0011  | 15.68 |  12292 | 0.520  |0.020|






