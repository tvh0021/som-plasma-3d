import numpy as np
import h5py as h5
import collections
from collections import defaultdict
import os
import glob
import argparse

# visualization
import matplotlib.pyplot as plt


# visualize matrix
def imshow(ax,
           grid, xmin, xmax, ymin, ymax,
           cmap='plasma',
           vmin = 0.0,
           vmax = 1.0,
           clip = -1.0,
           cap = None,
           aspect = 'auto',
           plot_log = False
          ):

    ax.clear()
    ax.minorticks_on()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    #ax.set_xlim(-3.0, 3.0)
    #ax.set_ylim(-3.0, 3.0)

    extent = [ xmin, xmax, ymin, ymax ]

    if clip == None:
        mgrid = grid
    elif type(clip) == tuple:
        cmin, cmax = clip
        print(cmin, cmax)
        mgrid = np.ma.masked_where( np.logical_and(cmin <= grid, grid <= cmax), grid)
    else:
        mgrid = np.ma.masked_where(grid <= clip, grid)

    if cap != None:
        mgrid = np.clip(mgrid, cap )

    if plot_log:
        mgrid = np.log10(mgrid)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)

    # mgrid = mgrid.T
    im = ax.pcolormesh(mgrid,
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax,
                       shading='auto')
    return im


# dictionary that stores mappings of runs and indices to count their score
    #if element is not here it automatically appends empty dict
    #mapping = collections.defaultdict(dict)
    #mapping = defaultdict(lambda : defaultdict(dict))

class InfNestedDict(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

# read array from clusterID.npy
def load_som_npy(path):
    return np.load(path, 'r')


# create a normal dens enumpy array by first copying input array,
# then masking all elements not equal to cid away
# then setting those elements to 1, others to 0
def create_mask(img, cid):
    return np.where(img == cid, 1, 0)

def nested_loop(all_data, number_of_clusters):
    runs = all_data
        
    mapping = InfNestedDict()
    multimap_mapping = InfNestedDict()
    
    # loop over data files reading image by image
    for i in range(len(runs)-1):
        run = runs[i]
        
        print('-----------------------')
        print("Run : ", run, flush=True)

        clusters = load_som_npy(run)

        # nx x ny x nz size maps
        nz,ny,nx = np.shape(clusters)

        # unique ids
        nids = number_of_clusters[i]
        ids = np.arange(nids)
        print('ids:', ids)
        
        for cid in range(nids):
            print('   -----------------------')
            print('   cid : ',cid, flush=True)
            # create masked array where only id == cid are visible
            mask = create_mask(clusters, cid)

            total_mask = np.zeros((ny,nx,nz), dtype=float)
            
            total_SQ_scalar = 0.
            total_S_scalar = 0.
            total_U_scalar = 0.
            
            mask_area_cid = np.sum(mask)/(nx*ny*nz)
            
            for j in range(i+1, len(runs)):
                runC = runs[j]

                print('     -----------------------')
                print('     ',runC, flush=True)

                clustersC = load_som_npy(runC)
                nidsC = number_of_clusters[j]
                idsC = np.arange(nidsC)
                print('    idsC:', idsC)

                for cidC in range(len(idsC)):
                    maskC = create_mask(clustersC, cidC)
                    maskC_area_cidC = np.sum(maskC) / (nx * ny * nz)

                    #--------------------------------------------------
                    # product of two masked arrays; corresponds to intersection
                    I = mask * maskC

                    #count True values of merged array divided by total number of values
                    I_area = np.sum(I) / (nx * ny * nz)

                    #--------------------------------------------------
                    # sum of two masked arrays; corresponds to union
                    U = np.ceil((mask + maskC) * 0.5)
                    U_area = np.sum(U) / (nx * ny * nz)
                    
                    #--------------------------------------------------
                    # Intersection signal strength of two masked arrays, S
                    S= np.sum(I) / np.sum(U)
                    # S_matrix = S * I # only needed for plotting
            
                    #--------------------------------------------------
                    # Union quality of two masked arrays, Q
                    if np.sum(mask) == 0.0  or np.sum(maskC) == 0.0:
                        continue
                    Q = np.sum(U) / (np.sum(mask) + np.sum(maskC)) - np.sum(I) / (np.sum(mask) + np.sum(maskC))
                    
                    if Q == 0.0:
                        continue #break here because this causes NaNs that accumulate.
                    
                    # Q_matrix = Q * U # only needed for plotting
                    
                    #--------------------------------------------------
                    # final measure for this comparison is (S/Q) x Union
                    SQ = S / Q
                    SQ_matrix = SQ * mask
                    
                    # append these measures to the mapping dictionary
                    mapping[run][cid][runC][cidC] = (total_S_scalar, total_U_scalar, SQ, S, Q, U_area, I_area, mask_area_cid, maskC_area_cidC)
                    
                    total_mask += SQ_matrix # pixelwise stacking of 2 masks
                    total_SQ_scalar += SQ
                    total_S_scalar += S
                    total_S_scalar += U
                    print('    S/Q', SQ)
                    # print('    S', S)
            
            # save total mask to file
            print("Saving total mask to file", flush=True)
            np.save(subfolder + '/mask3d-{}-id{}.npy'.format(run, cid), total_mask)
            
        multimap_mapping[run][cid] = (total_SQ_scalar, total_mask)
        
    return mapping, multimap_mapping
                    

parser = argparse.ArgumentParser(description='SCE code')
parser.add_argument('--folder', type=str, dest='folder', help='Folder name')
parser.add_argument('--slice', type=int, dest='slice', default=580, help='Slice number')
parser.add_argument('--subfolder', type=str, dest='subfolder', default='SCE', help='Subfolder name')

args = parser.parse_args()

if __name__ == "__main__":
    
    print("Starting SCE3d", flush=True)
    folder = args.folder
    os.chdir(folder)
    cluster_files = glob.glob('clusters*.npy')
    slice_number = args.slice

    #--------------------------------------------------
    # plotting env

    plt.rc('font',  family='sans-serif')
    #plt.rc('text',  usetex=True)
    plt.rc('xtick', labelsize=5)
    plt.rc('ytick', labelsize=5)
    plt.rc('axes',  labelsize=5)


    fig = plt.figure(1, figsize=(6,6), dpi=300)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    gs = plt.GridSpec(1, 1)
    gs.update(hspace = 0.05)
    gs.update(wspace = 0.05)

    axs = []
    axs.append( plt.subplot(gs[0,0]) )

    #--------------------------------------------------
    # data
    subfolder = args.subfolder
    print(cluster_files)
    
    #--------------------------------------------------
    # identify all clusters in all files
    
    print("Finding clusters in files", flush=True)
    nids_array = np.empty(len(cluster_files), dtype=int)
    for run in range(len(cluster_files)):
        clusters = load_som_npy(cluster_files[run])
        ids = np.unique(clusters)
        nids_array[run] = len(ids)
        # print (nids_array[run])
    
    print('nids_array:', nids_array, flush=True)

    #--------------------------------------------------
    # loop over data files reading image by image and do pairwise comparisons
    # all wrapped inside the nested_loop function, which uses JAX for fast computation
    mapping, multimap_mapping = nested_loop(cluster_files, nids_array)
    
   
    #--------------------------------------------------
    # print multimap stacked values
    if True: #TRUE
        print('\n')
        print('multimap mappings:-----------------------')

        with open(subfolder + '/multimap_mappings.txt', 'w') as f:
            for map1 in mapping:
                f.write('{}\n'.format(map1))
                # f.flush()
                for id1 in mapping[map1]:
                    #SQ, SQ_matrix = multimap_mapping[map1][id1]
                    total_SQ_scalar, SQ_matrix = multimap_mapping[map1][id1]
                    # print('   ', id1, total_SQ_scalar )
                    f.write('{} {}\n'.format(id1, total_SQ_scalar))
                    # f.flush()
        print("Done writing multimap mapping to file")
