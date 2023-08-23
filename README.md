# NeuronWithNetworkx
Combine NEURON and Networkx

You can just run Main.py to make the code work. The code will import two classes from CellwithNetworkx.py and DistanceAnalyzer.py.

The first class CellwithNetworkx is for adding features of graph to the single neuron model from Networkx. 
In this way, you can calculate the distance between each section and segment, and determine the order of 
each section from the soma.

The second class DistanceAnalyzer is for getting a better clustering pattern for the type of synapses to the object of CellwithNetworkx 
either with Monte Carlo training or classic clustering method (e.g. spectral_clustering and K-means).

Output figures of each simulation have been stored in the folder ./Results.
