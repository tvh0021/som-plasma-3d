import numpy as np
import h5py as h5
# import collections
# from collections import defaultdict
import os
import glob
import argparse

# visualization
import matplotlib.pyplot as plt

# use jax for fast computation
import jax.numpy as jnp
from jax import jit

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
@jit
def create_mask(img, cid):
    return jnp.where(img == cid, 1, 0)

# @jit
def compute_SQ(mask, maskC):
    #--------------------------------------------------
    # product of two masked arrays; corresponds to intersection
    I = jnp.multiply(mask, maskC)

    #--------------------------------------------------
    # sum of two masked arrays; corresponds to union
    U = jnp.ceil((mask + maskC) * 0.5)
    # U_area = jnp.sum(U) / (nx * ny * nz)
    
    #--------------------------------------------------
    # Intersection signal strength of two masked arrays, S
    S = jnp.sum(I) / jnp.sum(U)

    #--------------------------------------------------
    # Union quality of two masked arrays, Q
    if jnp.max(mask) == 0  or jnp.max(maskC) == 0:
        return 0., np.zeros(mask.shape), 0., 0.
    
    Q = jnp.sum(U) / (jnp.sum(mask) + jnp.sum(maskC)) - jnp.sum(I) / (jnp.sum(mask) + jnp.sum(maskC))
    if Q == 0.0:
        return 0., np.zeros(mask.shape), 0., 0. #break here because this causes NaNs that accumulate.
    
    #--------------------------------------------------
    # final measure for this comparison is (S/Q) x Union
    SQ = S / Q
    SQ_matrix = SQ * mask

    return SQ, np.array(SQ_matrix), S, U
    

def nested_loop(all_data, number_of_clusters):
    runs = all_data

    multimap_mapping = InfNestedDict()

    # define a dict beforehand to avoid memory leaks
    print("Predefining multimap_mapping dict", flush=True)
    for i in range(len(runs)):
        run = runs[i]
        for cid in range(number_of_clusters[i]):
            multimap_mapping[run][cid] = 0.
    
    # tracemalloc.start()
    
    # loop over data files reading image by image
    for i in range(len(runs)):
        run = runs[i]
        
        clusters = load_som_npy(run)
        print('-----------------------')
        print("Run : ", run, flush=True)

        # nx x ny x nz size maps
        nz,ny,nx = clusters.shape

        # unique ids
        nids = number_of_clusters[i] #number of cluster ids in this run
        ids = np.arange(nids)
        print('ids:', ids)
        
        
        for cid in range(nids):
            print('  -----------------------')
            print('  cid:', cid, flush=True)

            # create masked array where only id == cid are visible
            mask = create_mask(clusters, cid)

            total_mask = np.zeros((ny,nx,nz), dtype=float)
            
            total_SQ_scalar = 0.
            total_S_scalar = 0.
            # total_U_scalar = 0.
            
            # mask_area_cid = jnp.sum(mask)/(nx*ny*nz)
            
            for j in range(len(runs)):
                runC = runs[j]
                
                if j == i: # don't compare to itself
                    continue
                
                clustersC = load_som_npy(runC)

                print('    -----------------------')
                print('   ',runC, flush=True)

                nidsC = number_of_clusters[j] #number of cluster ids in this run
                idsC = np.arange(nidsC)
                print('    idsC:', idsC)

                for cidC in range(nidsC):
                    maskC = create_mask(clustersC, cidC)
                    
                    SQ, SQ_matrix, S, U = compute_SQ(mask, maskC)
                    
                    # append these measures to the mapping dictionary
                    # mapping[run][cid][runC][cidC] = (total_S_scalar, total_U_scalar, SQ, S, Q, U_area, I_area, mask_area_cid, maskC_area_cidC)
                    
                    # pixelwise stacking of 2 masks
                    total_mask += SQ_matrix # for numpy array
                    
                    total_SQ_scalar += SQ
                    total_S_scalar += S
                    total_S_scalar += U
                    print('    S/Q', SQ)
                    # print('    S', S)
            
            # save total mask to file
            print("Saving total mask to file", flush=True)
            np.save(subfolder + '/mask3d-{}-id{}.npy'.format(run, cid), total_mask)

            multimap_mapping[run][cid] = total_SQ_scalar
            
            # print("Total SQ scalar for this cluster:", flush=True)
            # print(total_SQ_scalar, flush=True)
            # print("Total mask shape : ", flush=True)
            # print(total_mask.shape, flush=True)
            # print("Total mask sum : ", flush=True)
            # print(np.sum(total_mask), flush=True)
            # print("Total mask max : ", flush=True)
            # print(np.max(total_mask), flush=True)
            # print("Total mask min : ", flush=True)
            # print(np.min(total_mask), flush=True)
            
        
    return multimap_mapping
                    

parser = argparse.ArgumentParser(description='SCE code')
parser.add_argument('--folder', type=str, dest='folder', help='Folder name')
# parser.add_argument('--slice', type=int, dest='slice', default=580, help='Slice number')
parser.add_argument('--subfolder', type=str, dest='subfolder', default='SCE', help='Subfolder name')

args = parser.parse_args()

if __name__ == "__main__":
    
    print("Starting SCE3d", flush=True)
    folder = args.folder
    os.chdir(folder)
    cluster_files = glob.glob('clusters*.npy')
    # slice_number = args.slice

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
    # calculate unique number of clusters per run
    
    nids_array = np.empty(len(cluster_files), dtype=int)
    for run in range(len(cluster_files)):
        clusters = load_som_npy(cluster_files[run])
        ids = jnp.unique(clusters)
        nids_array[run] = len(ids)
        # print (nids_array[run])
    
    print('nids_array:', nids_array, flush=True)

    #--------------------------------------------------
    # loop over data files reading image by image and do pairwise comparisons
    # all wrapped inside the nested_loop function, which uses JAX for fast computation
    multimap_mapping = nested_loop(cluster_files, nids_array)
    print("Done with nested loop")
    print("Size of multimap_mapping:", len(multimap_mapping))
    print("Keys of multimap_mapping:", multimap_mapping.keys())
    print("Size of multimap_mapping[0]:", len(multimap_mapping[cluster_files[0]]))
    print("Keys of multimap_mapping[0]:", multimap_mapping[cluster_files[0]].keys())
   
    #--------------------------------------------------
    # print multimap stacked values
    if True: #TRUE
        print('\n')
        print('multimap mappings:-----------------------')

        with open(subfolder + '/multimap_mappings.txt', 'w') as f:
            for map1 in multimap_mapping:
                f.write('{}\n'.format(map1))
                # f.flush()
                for id1 in multimap_mapping[map1]:
                    total_SQ_scalar = multimap_mapping[map1][id1]
                    f.write('{} {}\n'.format(id1, total_SQ_scalar))
                
        print("Done writing multimap mapping to file")
