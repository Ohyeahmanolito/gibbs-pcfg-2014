gibbs-pcfg-2014
===============

Creating Unknown Classifiers
-----------------------------
Pre-built unknown word-handling models are provided for the languages and datasets used in the paper. These are located in the `/unk` directory.

The process of training new models for use with our code is automated. To run this code, perform the following from the root directory:

```
./build
runMain UnknownClassiferGenerator training.mrg 10
```

This will load the (tagged) trees from `training.mrg`, extract word-level features and perform the clustering with 10 clusters. It will additionally train a LibLinear-based classifier, which will be output in the root directory.

For small datasets we recommend 5-10 unknown clusters, whereas WSJ-scale datasets can get a benefit from perhaps 20 clusters.
