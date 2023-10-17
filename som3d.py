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
parser.add_argument('--xdim', type=int, dest='xdim', default=10, help='Map x size')
parser.add_argument('--ydim', type=int, dest='ydim', default=10, help='Map y size')
parser.add_argument('--alpha', type=float, dest='alpha', default=0.5, help='Learning parameter')
parser.add_argument('--train', type=int, dest='train', default=10000, help='Number of training steps')
parser.add_argument('--batch', type=int, dest='batch', default=None, help='width of domain in a batch', required=False)

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
        print("Cannot crop, smaller domain is outside of current domain")
        return 
    else:
        print(f"Cropped domain starts at: [{start_index_z},{start_index_y},{start_index_x}], width = {new_width}")
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

def flatten_to_2d(data_array_4d):
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


if __name__ == "__main__":

        # set-up plotting
        #plt.fig = plt.figure(1, figsize=(4,3.5), dpi=200)
        #fig = plt.figure(1, figsize=(6,6), dpi=300)

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

        nx,ny,nz = 640, 640, 640

        # f5 = h5.File('/mnt/home/tha10/SOM-tests/data_features_3dfull_{}.h5'.format(lap), 'r')
        f5 = h5.File('/mnt/home/tha10/SOM-tests/hr-d3x640/features_4j1b1e_{}.h5'.format(lap), 'r')
        x = f5['features'][()]

        y = f5['target'][()]
        feature_list = f5['names'][()]

        feature_list = [n.decode('utf-8') for n in feature_list]
        f5.close()
        print(f"File loaded, parameters: {lap}-{args.xdim}-{args.ydim}-{args.alpha}-{args.train}-{args.batch}")

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
        if args.batch == None:
                # POPSOM SOM:
                attr=pd.DataFrame(x)
                attr.columns=feature_list

                print(f'constructing full SOM for xdim={args.xdim}, ydim={args.ydim}, alpha={args.alpha}, train={args.train}...')
                m=popsom.map(args.xdim, args.ydim, args.alpha, args.train)

                labels = [str(xxx) for xxx in range(len(x))]
                m.fit(attr,labels)
                neurons = m.all_neurons()
                # print("neurons: ", neurons)
                np.save(f'neurons_{lap}_{args.xdim}{args.ydim}_{args.alpha}_{args.train}.npy', neurons, allow_pickle=True)
        else:
                width_of_new_window = args.batch
                x_4d = convert_to_4d(x)

                for split_index1 in range(nz // width_of_new_window):
                        for split_index2 in range(ny // width_of_new_window):
                               for split_index3 in range(nx // width_of_new_window):
                                        start_index_crop_x = split_index1 * width_of_new_window
                                        start_index_crop_y = split_index2 * width_of_new_window
                                        start_index_crop_z = split_index3 * width_of_new_window
                                        x_split_4d = get_smaller_domain(x_4d, width_of_new_window, start_index_crop_x, start_index_crop_y, start_index_crop_z)

                # for split_index1 in range(nx // args.batch):
                #         for split_index2 in range(ny // args.batch):
                #                for split_index3 in range(nz // args.batch):
                #                         matrix_indices = np.array([]) # list of indices that are inside the 3d domain

                #                         for x1 in range(split_index1*args.batch, (split_index1+1)*args.batch):
                #                                 for x2 in range(split_index2*args.batch, (split_index2+1)*args.batch):
                #                                         for x3 in range(split_index3*args.batch, (split_index3+1)*args.batch):
                #                                                 matrix_indices = np.append(matrix_indices, np.array([x1,x2,x3]), axis=1)

                #                         x_split = np.ravel_multi_index(matrix_indices, (nx,ny,nz))
                                        x_split = flatten_to_2d(x_split_4d)
                                        attr=pd.DataFrame(x_split)
                                        attr.columns=feature_list

                                        print(f'constructing batch SOM for xdim={args.xdim}, ydim={args.ydim}, alpha={args.alpha}, train={args.train}, index=[{start_index_crop_z},{start_index_crop_y},{start_index_crop_x}]...')
                                        m=popsom.map(args.xdim, args.ydim, args.alpha, args.train)

                                        labels = [str(xxx) for xxx in range(len(x_split))]
                                        if (split_index1 == 0) & (split_index2 == 0) & (split_index3 == 0):
                                                m.fit(attr,labels,restart=False)
                                        else:
                                                m.fit(attr,labels,restart=True, neurons=neurons)

                                        neurons = m.all_neurons()
                                        # print("neurons: ", neurons)
                                        np.save(f'neurons_{lap}_{args.xdim}{args.ydim}_{args.alpha}_{args.train}_{split_index1}-{split_index2}-{split_index3}.npy', neurons, allow_pickle=True)



        

        # print(f"convergence at {args.train} steps = {m.convergence()}")

        #Data matrix with neuron positions:
        data_matrix=m.projection()
        data_Xneuron=data_matrix['x']
        data_Yneuron=data_matrix['y']
        # print("Printing Xneuron info")
        # print(data_Xneuron)
        # print("Printing Xneuron info position 5")
        # print(data_Xneuron[4])
        # print("Printing Yneuron info")
        # print(data_Yneuron)

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
        # print("Number of clusters")
        # print(n_clusters)

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

        # print("clusters")
        # print(clusters)
        # print(np.shape(clusters))

        def get_N_HexCol(N=n_clusters):
            HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
            hex_out = []
            for rgb in HSV_tuples:
                    rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
                    hex_out.append('#%02x%02x%02x' % tuple(rgb))
            return hex_out

        #TRANSFER RESULT BACK INTO ORIGINAL DATA PLOT
        if True:
                cluster_id = np.zeros((nx,ny,nz))

                xinds = np.zeros(len(data_Xneuron))
                # print("shape of xinds:", np.shape(xinds))
                j = 0
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            cluster_id[ix,iy,iz] = clusters[data_Xneuron[j], data_Yneuron[j]]
                            xinds[j] = clusters[data_Xneuron[j], data_Yneuron[j]]
                            j += 1

                f5 = h5.File(f'clusters_{lap}_{args.xdim}{args.ydim}_{args.alpha}_{args.train}.h5', 'w')
                dsetx = f5.create_dataset("cluster_id",  data=cluster_id)
                f5.close()
                # print("Done writing the cluster ID file")

        # #PLOTTING:
        #visualize clusters

        x = np.array(x)
        cluster_id = np.array(cluster_id)
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
