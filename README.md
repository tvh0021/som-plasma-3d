# Segmenting turbulent simulations with ensemble learning

Computer vision and machine learning tools offer an exciting new way for automatically analyzing and categorizing information from complex computer simulations.
Here we introduce a new ensemble machine learning fraemwork that can classify and dissect patterns from images, and categorize them into distinct structure catalogues.

Individual segmentation evaluations are performed with Self-Organizing Maps (SOMs) using the `popsom` python package.   
These are combined together using the Statistically-Combined Ensemble (SCE) algorithm, presented in M. Bussov & J. Nattila 2021 (SPIC), to reduce noise in the classification process and to increase performance of the clustering techniques.


## Example data from turbulence simulations

As an example data, we use a 2-dimensional simulation of magnetically-dominated kinetic turbulence.
The SOM and SCE algorithms are used to dissect the data into accurate ROI boundaries of different geometrical structures present in the flow.

TODO: picture of data; features


## Performing individual classification using SOMs

SOM use 2-layer neural network achitecture to produce two-dimensional, compressed map of the original N-dimensional input space.
The distance matrix of the neurons in the map can be used to extract clusters from the original input data.

To perform a single SOM classification of the example data, run
```python
   python3 som.py --args
```

The SOM algorithm uses `popsom` [package](https://github.com/njali2001/popsom) discussed [here](https://digitalcommons.uri.edu/theses/1244/).

## Combining independent evaluations with SCE

SCE is used to combine independent SOM classifications into a joint classification result.
A cluster representation of an image is called a mask (boolean matrix) and the algorithm works by combining the masks together using set theoretical similarity measures.
The final measure gives an estimate for the stability and robustness for each detected cluster in the total ensemble.

To perform an SCE combination of the previously ran SOM maps, use
```python
   python3 sce.py --args
```


## How to cite?

You can use the following BibTeX template to cite Runko in any scientific discourse:
```
@ARTICLE{mbussov2021,
       author = {{Bussov, M} and {N\"attil\"a}, J.,}
}
```

