## SOM code for reproducing Bussov & Nattila 2021 image segmentation results

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
#import tensorflow as tf
from matplotlib import colors
import popsom.popsom as popsom
import pandas as pd
import colorsys
from numba import njit, prange

import h5py as h5
import sys, os

from utils_plot2d import read_var
from utils_plot2d import Conf
from utils_plot2d import imshow

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics

import argparse

parser = argparse.ArgumentParser(description='popsom code')
parser.add_argument("--features_path", type=str, dest='features_path', default='/mnt/ceph/users/tha10/SOM-tests/hr-d3x640/')
parser.add_argument("--file", type=str, dest='file', default='features_4j1b1e_2800.h5')
parser.add_argument('--xdim', type=int, dest='xdim', default=10, help='Map x size')
parser.add_argument('--ydim', type=int, dest='ydim', default=10, help='Map y size')
parser.add_argument('--alpha', type=float, dest='alpha', default=0.5, help='Learning parameter')
parser.add_argument('--train', type=int, dest='train', default=10000, help='Number of training steps')
parser.add_argument('--batch', type=int, dest='batch', default=None, help='Width of domain in a batch', required=False)
parser.add_argument('--pretrained', type=bool, dest='pretrained', default=False, help='Is the model is pretrained?', required=False)
parser.add_argument('--neurons_path', type=str, dest='neurons_path', default=None, help='Path to file containing neuron values', required=False)
parser.add_argument('--save_neuron_values', type=bool, dest='save_neuron_values', default=False, help='Save neuron values to file?', required=False)

args = parser.parse_args()

def get_smaller_domain(data_array, new_width, start_index_x, start_index_y, start_index_z):
    """Get a smaller domain from a full box simulation to save computational time

    Args:
        data_array (numpy 3d array): cubic box of some data [4d array should also work, provided that the first 3 dimensions are spatial]
        fraction (float): fraction of the original domain to keep, between 0 and 1
        start_index (int): starting index of the bigger domain

    Returns:
        numpy 3d(4d) array: cropped cubic box
    """
    if (start_index_z + new_width > data_array.shape[0]) | (start_index_y + new_width > data_array.shape[1]) | (start_index_x + new_width > data_array.shape[2]):
        print("Cannot crop, smaller domain is outside of current domain", flush=True)
        return 
    else:
        print(f"Cropped domain starts at: [{start_index_z},{start_index_y},{start_index_x}], width = {new_width}", flush=True)
        return data_array[start_index_z:start_index_z+new_width, start_index_y:start_index_y+new_width, start_index_x:start_index_x+new_width]
    
def convert_to_4d(data_array_2d):
        """Convert a n x f array to a a x b x c x f array, where f is values of certain features and a, b, c are grid points

        Args:
            data_array_2d (numpy 2d array)

        Returns:
            numpy 4d array
        """
        nd = int(np.cbrt(data_array_2d.shape[0]))
        data_array_4d = np.zeros((nd,nd,nd,data_array_2d.shape[-1]))

        for f in range(data_array_2d.shape[-1]):
                feature_3d = np.reshape(data_array_2d[:,f], newshape=[nd,nd,nd])
                data_array_4d[:,:,:,f] = feature_3d[:,:,:]
        return data_array_4d

def flatten_to_2d(data_array_4d : np.ndarray):
        """Convert a a x b x c x f array to a n x f, where f is values of certain features and a, b, c are grid points

        Args:
            data_array_4d (numpy 4d array)

        Returns:
            numpy 2d array
        """
        nd = data_array_4d.shape[0]
        data_array_2d = np.zeros((int(nd**3), data_array_4d.shape[-1]))
        for f in range(data_array_4d.shape[-1]):
                data_array_2d[:,f] = data_array_4d[:,:,:,f].flatten()
        return data_array_2d

@njit(parallel=True)
def assign_cluster_id(nx : int, ny : int, nz : int, data_Xneuron : np.ndarray, data_Yneuron : np.ndarray, clusters : np.ndarray) -> np.ndarray:
        """From neuron data and cluster assignments, return the cluster id of the cell

        Args:
            nx (int): length of x-dimension
            ny (int): length of y-dimension
            nz (int): length of z-dimension
            data_Xneuron (np.ndarray): 1d array with x coordinate of the neuron associated with a cell
            data_Yneuron (np.ndarray): 1d array with y coordinate of the neuron associated with a cell
            clusters (np.ndarray): n x n matrix of cluster on neuron map

        Returns:
            np.ndarray: cluster_id
        """
        cluster_id = np.zeros((nz,ny,nx))
        for iz in prange(nz):
                for iy in prange(ny):
                        for ix in prange(nx):
                                j = iz * ny * nx + iy * nx + ix # convert from 3d coordinates to 1d row indices
                                cluster_id[iz,iy,ix] = clusters[int(data_Xneuron[j]), int(data_Yneuron[j])]
        return cluster_id

def batch_training(full_data, batch, feature_list, save_neuron_values=False):
        """Function to perform batch training on a full domain

        Args:
            full_data (numpy ndarray): N x F array, where N is the number of data points and F is the number of features
            batch (int): width of the domain to be trained on
            feature_list (list of str): list of feature names

        Returns:
            class: popsom map
        """
        width_of_new_window = batch
        x_4d = convert_to_4d(full_data)
        history = []

        for split_index1 in range(nz // width_of_new_window):
                start_index_crop_z = split_index1 * width_of_new_window
                for split_index2 in range(ny // width_of_new_window):
                        start_index_crop_y = split_index2 * width_of_new_window
                        for split_index3 in range(nx // width_of_new_window):
                                start_index_crop_x = split_index3 * width_of_new_window
                                
                                x_split_4d = get_smaller_domain(x_4d, width_of_new_window, start_index_crop_x, start_index_crop_y, start_index_crop_z)

                                x_split = flatten_to_2d(x_split_4d)
                                attr=pd.DataFrame(x_split)
                                attr.columns=feature_list

                                print(f'constructing batch SOM for xdim={xdim}, ydim={ydim}, alpha={alpha}, train={train}, index=[{start_index_crop_z},{start_index_crop_y},{start_index_crop_x}]...', flush=True)
                                m=popsom.map(xdim, ydim, alpha, train)

                                labels = np.array(list(range(len(x_split))))
                                if (split_index1 == 0) & (split_index2 == 0) & (split_index3 == 0):
                                        m.fit(attr,labels,restart=False)
                                else: # if first window, then initiate random neuron values, else use neurons from last batch
                                        m.fit(attr,labels,restart=True, neurons=neurons)

                                neurons = m.all_neurons()
                                # print("neurons: ", neurons)
                                # if save_neuron_values == True:
                                        # np.save(f'neurons_{lap}_{xdim}{ydim}_{alpha}_{train}_{split_index1}-{split_index2}-{split_index3}.npy', neurons, allow_pickle=True)
                                
                                # print changes in neuron weights
                                neuron_weights = m.weight_history
                                term = m.final_epoch
                                history.extend(neuron_weights)
                                # np.save(f'evolution_{lap}_{xdim}{ydim}_{alpha}_{term}_{split_index1}-{split_index2}-{split_index3}.npy', neuron_weights, allow_pickle=True)

        # at the end, load the entire domain back to m to assign cluster id
        attr=pd.DataFrame(full_data)
        attr.columns=feature_list
        labels = np.array(list(range(len(x))))
        m.fit_notraining(attr, labels, neurons)

        np.save(f'evolution_{lap}_{xdim}{ydim}_{alpha}_{train}_combined.npy', np.array(history), allow_pickle=True)

        return m


if __name__ == "__main__":

        # set-up plotting
        #plt.fig = plt.figure(1, figsize=(4,3.5), dpi=200)
        #fig = plt.figure(1, figsize=(6,6), dpi=300)

        if (args.pretrained == True) & (args.neurons_path is None):
               sys.exit("Cannot run, no neuron values provided.")

        plt.rc('font',  family='sans-serif')
        #plt.rc('text',  usetex=True)
        plt.rc('xtick', labelsize=5)
        plt.rc('ytick', labelsize=5)
        plt.rc('axes',  labelsize=5)

        conf = Conf()

        #--------------------------------------------------
        xmin = 0.0
        ymin = 0.0
        xmax = 1.0
        ymax = 1.0

        laps = [2800] # all the data laps to process
        lap = laps[0] # data file number

        # CLI arguments
        features_path = args.features_path
        file_name = args.file
        xdim = args.xdim
        ydim = args.ydim
        alpha = args.alpha
        train = args.train
        batch = args.batch
        pretrained = args.pretrained
        neurons_path = args.neurons_path
        save_neuron_values = args.save_neuron_values

        nx,ny,nz = 640, 640, 640

        # f5 = h5.File('/mnt/home/tha10/SOM-tests/hr-d3x640/features_4j1b1e_{}.h5'.format(lap), 'r')
        # f5 = h5.File('/Users/tha/Downloads/Archive/features_4j1b1e_{}.h5'.format(lap), 'r')
        f5 = h5.File(features_path+file_name, 'r')

        x = f5['features'][()]

        y = f5['target'][()]
        feature_list = f5['names'][()]

        feature_list = [n.decode('utf-8') for n in feature_list]
        f5.close()
        print(f"File loaded, parameters: {lap}-{xdim}-{ydim}-{alpha}-{train}-{batch}", flush=True)

        # print(feature_list)
        # print("shape after x:", np.shape(x))

        #--------------------------------------------------
        # analyze
        #1. standardize:
        scaler = StandardScaler()
        # scaler = MinMaxScaler()

        scaler.fit(x)
        x = scaler.transform(x)

        # if the SOM is to be divided into smaller batches, separate those batches window by window
        if (batch is None) & (pretrained == False):
                # POPSOM SOM:
                attr=pd.DataFrame(x)
                attr.columns=feature_list

                print(f'constructing full SOM for xdim={xdim}, ydim={ydim}, alpha={alpha}, train={train}...', flush=True)
                m=popsom.map(xdim, ydim, alpha, train)

                labels = np.array(list(range(len(x))))
                m.fit(attr,labels)
                neurons = m.all_neurons()
                # print("neurons: ", neurons)
                if save_neuron_values == True:
                        np.save(f'neurons_{lap}_{xdim}{ydim}_{alpha}_{train}.npy', neurons, allow_pickle=True)
                # print changes in neuron weights
                neuron_weights = m.weight_history
                term = m.final_epoch
                np.save(f'evolution_{lap}_{xdim}{ydim}_{alpha}_{term}.npy', neuron_weights, allow_pickle=True)
        elif (batch is not None) & (pretrained == False):
                m = batch_training(x, batch, feature_list, save_neuron_values)
        else: # if the run is initialized as a no training run, load these values
                print(f'constructing pre-trained SOM for xdim={xdim}, ydim={ydim}, alpha={alpha}, train={train}...', flush=True)
                m=popsom.map(xdim, ydim, alpha, train)
                attr=pd.DataFrame(x)
                attr.columns=feature_list
                labels = np.array(list(range(len(x))))
                neurons = np.load(neurons_path)
                m.fit_notraining(attr,labels,neurons)

        

        # print(f"convergence at {args.train} steps = {m.convergence()}")

        #Data matrix with neuron positions:
        print("Calculating projection")
        data_matrix=m.projection()
        data_Xneuron=data_matrix[:,0]
        data_Yneuron=data_matrix[:,1]
        print("data matrix: ", flush=True)
        print(data_matrix[:10,:], flush=True)
        print("Printing Xneuron info", flush=True)
        print("Shape of Xneuron: ", data_Xneuron.shape, flush=True)
        print("Printing Yneuron info", flush=True)
        print("Shape of Yneuron: ", data_Yneuron.shape, flush=True)

        #Neuron matrix with centroids:
        umat = m.compute_umat(smoothing=2)
        centrs = m.compute_combined_clusters(umat, False, 0.15) #0.15
        centr_x = centrs['centroid_x']
        centr_y = centrs['centroid_y']

        #create list of centroid _locations
        neuron_x, neuron_y = np.shape(centr_x)

        centr_locs = []
        for i in range(neuron_x):
                for j in range(neuron_y):
                        cx = centr_x[i,j]
                        cy = centr_y[i,j]

                        centr_locs.append((cx,cy))

        unique_ids = list(set(centr_locs))
        # print(unique_ids)
        n_clusters = len(unique_ids)
        print("Number of clusters", flush=True)
        print(n_clusters)

        mapping = {}
        for I, key in enumerate(unique_ids):
                # print(key, I)
                mapping[key] = I

        clusters = np.zeros((neuron_x,neuron_y))
        for i in range(neuron_x):
                for j in range(neuron_y):
                        key = (centr_x[i,j], centr_y[i,j])
                        I = mapping[key]

                        clusters[i,j] = I

        # print(centr_x)
        # print(centr_y)

        print("clusters", flush=True)
        print(clusters, flush=True)
        # print(np.shape(clusters))
        

        #TRANSFER RESULT BACK INTO ORIGINAL DATA PLOT
        # xinds = np.zeros(len(data_Xneuron))
        # print("shape of xinds:", np.shape(xinds))
        print("Assigning clusters", flush=True)
        
        cluster_id = assign_cluster_id(nx, ny, nz, data_Xneuron, data_Yneuron, clusters)
        np.save(f'clusters_{lap}_{xdim}{ydim}_{alpha}_{train}.npy', cluster_id, allow_pickle=True)

        # f5 = h5.File(f'clusters_{lap}_{xdim}{ydim}_{alpha}_{train}.h5', 'w')
        # dsetx = f5.create_dataset("cluster_id",  data=cluster_id)
        # f5.close()
        print("Done writing the cluster ID file")

        
        
        # #PLOTTING:
        #visualize clusters

        def get_N_HexCol(N=n_clusters):
            HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
            hex_out = []
            for rgb in HSV_tuples:
                    rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
                    hex_out.append('#%02x%02x%02x' % tuple(rgb))
            return hex_out        

        # x = np.array(x)
        if False:
                print("visualizing SOM data")
                fig2 = plt.figure(2, figsize=(20,20), dpi=400)
                fig2.clf()

                ic = 0
                nfea = len(feature_list)
                cols=get_N_HexCol(N=n_clusters)
                for jcomp in range(nfea):
                        for icomp in range(nfea):
                                ic += 1
                                print("i {} j {}".format(icomp, jcomp))

                                #skip covariance with itself
                                if icomp == jcomp:
                                        continue

                                #skip upper triangle
                                if icomp > jcomp:
                                        continue

                                ax = fig2.add_subplot(nfea, nfea, ic)

                                for ki in range(n_clusters):
                                        indxs = np.where(xinds == ki)
                                        #print("len of found pixels:", len(indxs), indxs)
                                        xx = x[indxs, icomp]
                                        yy = x[indxs, jcomp]

                                        xxt = xx[::500]
                                        yyt = yy[::500]

                                        ax.scatter(
                                                xxt,
                                                yyt,
                                                c=cols[ki],
                                                marker='.',
                                                s=1.,
                                                edgecolors="none",
                                                rasterized=True,
                                                )

                                if False:
                                        #visualize most dissipative points
                                        xx = x[:,icomp]
                                        yy = x[:,jcomp]
                                        zz = y[:]

                                        xxd = xx[np.where(np.abs(zz) > 0.020)]
                                        yyd = yy[np.where(np.abs(zz) > 0.020)]
                                        zzd = zz[np.where(np.abs(zz) > 0.020)]

                                        xxd = xxd[::100]
                                        yyd = yyd[::100]
                                        zzd = zzd[::100]

                                        print("found {} points above threshold".format(len(xxd)))

                                        ax.scatter(xxd,yyd,c=zzd,
                                                                cmap='inferno',
                                                                vmin=-0.015,
                                                                vmax= 0.015,
                                                                marker='.',
                                                                s=0.05,
                                                                alpha=0.1,
                                                                )

                                if jcomp == nfea-1:
                                        ax.set_xlabel('{}'.format(feature_list[icomp]))
                                else:
                                        ax.set_xticks([])

                                if icomp == 0:
                                        ax.set_ylabel('{}'.format(feature_list[jcomp]))
                                else:
                                        ax.set_yticks([])

                fig2.savefig('som_clusters.pdf')
