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

args = parser.parse_args()




def neighbors(arr,x,y,n=3):
        ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
        arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
        return arr[:n,:n]

if __name__ == "__main__":

        # set-up plotting
        #plt.fig = plt.figure(1, figsize=(4,3.5), dpi=200)
        #fig = plt.figure(1, figsize=(6,6), dpi=300)

        plt.rc('font',  family='sans-serif')
        #plt.rc('text',  usetex=True)
        plt.rc('xtick', labelsize=5)
        plt.rc('ytick', labelsize=5)
        plt.rc('axes',  labelsize=5)

        scaler = StandardScaler()
        conf = Conf()

        #build feature matrix
        # feature_list = [
        #                 'rho',
        #                 'bx',
        #                 'by',
        #                 'bz',
        #                 'ex',
        #                 'ey',
        #                 'ez',
        #                 'jx',
        #                 'jy',
        #                 'jz',
        #                 ]

        #--------------------------------------------------
        xmin = 0.0
        ymin = 0.0
        xmax = 1.0
        ymax = 1.0

        laps = [5000] # all the data laps to process
        lap = laps[0] # data file number

        nx,ny,nz = 128,128,128

        if True:
                # f5 = h5.File('/mnt/home/tha10/SOM-tests/data_features_3dfull_{}.h5'.format(lap), 'r')
                f5 = h5.File('/mnt/home/tha10/SOM-tests/features_4j1b1e_{}.h5'.format(lap), 'r')
                x = f5['features'][()]pwd

                y = f5['target'][()]
                feature_list = f5['names'][()]

                feature_list = [n.decode('utf-8') for n in feature_list]
                f5.close()

        # print(feature_list)
        # print("shape after x:", np.shape(x))

        #--------------------------------------------------
        # analyze
        #1. standardize:
        # scaler = StandardScaler()
        scaler = MinMaxScaler()

        scaler.fit(x)
        x = scaler.transform(x)


        ##### Using the SOM:

        #       POPSOM SOM:
        attr=pd.DataFrame(x)
        attr.columns=feature_list
        #parser.parse_args
        # print("setting dimensions", parser.parse_args())

        print(f'constructing SOM for xdim={args.xdim}, ydim={args.ydim}, alpha={args.alpha}, train={args.train}...')
        m=popsom.map(args.xdim, args.ydim, args.alpha, args.train)

        labels = [str(xxx) for xxx in range(len(x))]
        m.fit(attr,labels)
        # m.starburst()

        # m.significance()

        print(f"convergence at {args.train} steps = {m.convergence()}")

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
