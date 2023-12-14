import numpy as np
import h5py as h5
import collections
from collections import defaultdict
import sys, os
import glob
import argparse
import tracemalloc

# visualization
import matplotlib.pyplot as plt

# image analysis
from skimage import measure

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

# read array from clusterID.npy
def load_som_npy(path):
    return np.load(path, 'r')


# create a normal dens enumpy array by first copying input array,
# then masking all elements not equal to cid away
# then setting those elements to 1, others to 0
def create_mask(img, cid):
    img_masked = np.copy(img)

    #create masked array
    mask_arr = np.ma.masked_where( img == cid, img )

    # set values mask to 1, 0 elsewhere
    img_masked[mask_arr.mask]  = 1.0
    img_masked[~mask_arr.mask] = 0.0

    return img_masked


parser = argparse.ArgumentParser(description='SCE code')
parser.add_argument('--folder', type=str, dest='folder', help='Folder name')
parser.add_argument('--slice', type=int, dest='slice', default=580, help='Slice number')
parser.add_argument('--subfolder', type=str, dest='subfolder', default='SCE', help='Subfolder name')

args = parser.parse_args()

if __name__ == "__main__":

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

    runid = 0
    if runid == 0:
        subfolder = args.subfolder
        runs = cluster_files
    else:
        print('run not implemented yet')


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

    mapping = InfNestedDict()

    multimap_mapping = InfNestedDict()
    
    

    # loop over data files reading image by image
    for i in range(len(runs)):
        run = runs[i]
        print('-----------------------')
        print(run)

        tracemalloc.start()
        
        clusters = load_som_npy(run)

        # nx x ny size maps and ni subimages
        nz,ny,nx = np.shape(clusters)

        #image size
        xmin, xmax = 0, nx
        ymin, ymax = 0, ny

        # unique ids
        ids = np.unique(clusters[slice_number,:,:])
        print('ids:', ids)
        nids = len(ids) #number of cluster ids in this run

        # visualize first image as an example
        if False:
            imshow(axs[0],
                    clusters[slice_number,:,:],
                    xmin,xmax,ymin,ymax,
                    vmin = 0.0,
                    vmax = nids,
                    cmap='Spectral',
                    )
            fig.savefig(subfolder + '/stack_{}.png'.format(run))


        #pick one/first image as a reference for plotting
        img = clusters[slice_number,:,:]

        #--------------------------------------------------
        if True: #TRUE
            # map 2 map comparison
            #
            # build a raw mapping of ids by naively maximizing overlapping area
            # this is appended to `mappings` dictionary

            # loop over run1 ids
            for cid in range(nids):

                # create masked array where only id == cid are visible
                mask = create_mask(img, cid)


                total_mask = np.zeros(np.shape(mask)) #NOTE: total mask for base map (run) with cluster cid over all other maps (runC) and their clusters
                total_SQ_scalar = 0.0
                total_S_scalar=0.0
                total_U_scalar=0.0

                #mask_area_cid = np.ma.count_masked(mask)/(nx*ny)
                mask_area_cid = np.sum(mask)/(nx*ny)

                #loop over all other runs again to make run1 vs run2 comparison
                for j in range(i+1, len(runs)):
                    runC = runs[j]

                    print('    -----------------------')
                    print('    ',runC)

                    clustersC = load_som_npy(runC)
                    idsC = np.unique(clustersC[slice_number,:,:])
                    imgC = clustersC[slice_number,:,:]


                    #loop over all ids in run2
                    for cidC in range(len(idsC)):
                        maskC = create_mask(imgC, cidC)
                        maskC_area_cidC = np.sum(maskC)/(nx*ny)

                        #--------------------------------------------------
                        # product of two masked arrays; corresponds to intersection
                        I = mask * maskC

                        #count True values of merged array divided by total number of values
                        I_area = np.sum(I)/(nx*ny)

                        #print('{}: {} vs {}: {} ='.format(
                        #    run, cid,
                        #    runC, cidC,
                        #    intersect_area))

                        if False:
                            #print('plotting intersect...')
                            imshow(axs[0],
                                    I,
                                    xmin,xmax,ymin,ymax,
                                    vmin = 0.0,
                                    vmax = 1.0,
                                    cmap='binary',
                                    )
                            fig.savefig(subfolder+'/intersect_map1-{}_map2-{}_id1-{}_id2-{}.png'.format(run, runC, cid, cidC))


                        #--------------------------------------------------
                        # sum of two masked arrays; corresponds to union
                        U = np.ceil((mask + maskC)*0.5) #ceil to make this 0s and 1s

                        #count True values of merged array divided by total number of values
                        U_area = np.sum(U)/(nx*ny)

                        if False:
                            #print('plotting union...')
                            imshow(axs[0],
                                    U,
                                    xmin,xmax,ymin,ymax,
                                    vmin = 0.0,
                                    vmax = 1.0,
                                    cmap='binary',
                                    )
                            fig.savefig(subfolder + '/union_map1-{}_map2-{}_id1-{}_id2-{}.png'.format(run, runC, cid, cidC))

                        #--------------------------------------------------

                        # Intersection signal strength of two masked arrays, S
                        S=np.sum(I)/np.sum(U)
                        #S = I_area/U_area
                        S_matrix = S * I


                        if False:
                            print('plotting intersect area...', S)
                            imshow(axs[0],
                                    S_matrix,
                                    xmin,xmax,ymin,ymax,
                                    vmin = 0.0,
                                    vmax = S,
                                    cmap='seismic',
                                    )
                            fig.savefig(subfolder + '/signalstrength_map1-{}_map2-{}_id1-{}_id2-{}.png'.format(run, runC, cid, cidC))

                        #--------------------------------------------------
                        # Union quality of two masked arrays, Q
                        if np.sum(mask) == 0.0  or np.sum(maskC) == 0.0:
                            continue

                        #Q = U_area/(np.sum(mask)+np.sum(maskC))-I_area/(np.sum(mask)+np.sum(maskC))
                        Q = np.sum(U)/(np.sum(mask)+np.sum(maskC))-np.sum(I)/(np.sum(mask)+np.sum(maskC))
                        if Q == 0.0:
                            continue #break here because this causes NaNs that accumulate. Why we get division by zero?

                        Q_matrix = Q * U

                        if False:
                            print('plotting quality...', Q)
                            imshow(axs[0],
                                    Q_matrix,
                                    xmin,xmax,ymin,ymax,
                                    vmin = 0.0,
                                    vmax = Q,
                                    cmap='YlGn',
                                    )
                            fig.savefig(subfolder+ '/quality_map1-{}_map2-{}_id1-{}_id2-{}.png'.format(run, runC, cid, cidC))



                        #--------------------------------------------------
                        # final measure for this comparison is (S/Q) x Union
                        SQ = (S/Q)

                        #normalize SQ with total map size to get smaller numbers (makes numerics easier)
                        #SQ /= nx*ny


                        #SQ_matrix = SQ*I #NOTE: this is actually (S/Q)xI (not union)
                        #SQ_matrix = (SQ*U)
                        SQ_matrix=SQ*mask

                        if False:
                            print('plotting SQU...', SQ)
                            imshow(axs[0],
                                    SQ_matrix,
                                    xmin,xmax,ymin,ymax,
                                    vmin = 0.0,
                                    vmax = SQ,
                                    cmap='plasma',
                                    plot_log = True,
                                    )
                            fig.savefig(subfolder + '/SQU_map1-{}_map2-{}_id1-{}_id2-{}.png'.format(run, runC, cid, cidC))


                        # append these measures to the mapping dictionary
                        mapping[run][cid][runC][cidC] = (total_S_scalar, total_U_scalar, SQ, S, Q, U_area, I_area, mask_area_cid, maskC_area_cidC)

                        total_mask[:,:] += SQ_matrix[:,:] #pixelwise stacking of 2 masks
                        total_SQ_scalar += SQ
                        total_S_scalar+=S
                        total_S_scalar+=U
                        print('    S/Q', SQ)
                        print('    S', S)

                    #end of loop over runC cids
                #end of loop over runCs


                #--------------------------------------------------
                # total measure of this cluster id in this map is sum( S/Q )
                #total_SQ = np.sum(total_mask)/(nx*ny)

                #skip self to self comparison
                if total_SQ_scalar == 0.0:
                    continue

                total_SQ_from_matrix = np.sum(total_mask)/(nx*ny)

                multimap_mapping[run][cid] = (total_SQ_scalar, total_mask)
                
                # save total mask to file
                print("Saving total mask to file")
                np.save(subfolder + '/mask-{}-id{}.npy'.format(run, cid), total_mask)

                if True: #TRUE
                    # print('plotting total SQU:', total_SQ_scalar, 'vs sum', total_SQ_from_matrix,' min:', np.min(total_mask), ' max:', np.max(total_mask) )
                    imshow(axs[0],
                           total_mask,
                           xmin,xmax,ymin,ymax,
                           vmin = 0.0,     #np.min(total_mask),
                           vmax = np.max(total_mask),  #10, np.max(total_mask), #NOTE: 1e7 is about maximum value we seem to get
                           cmap='Reds',
                           )
                    fig.savefig(subfolder + '/SQ_map1-{}_id1-{}.png'.format(run, cid))

                    #log version
                    # imshow(axs[0],
                    #        total_mask,
                    #        xmin,xmax,ymin,ymax,
                    #        vmin = -2,    #0.1 np.min(total_mask),
                    #        vmax = 0.,   #10, np.max(total_mask), #NOTE: 1e7 is about maximum value we seem to get
                    #        cmap='Blues',
                    #        plot_log = True,
                    #        )
                    # fig.savefig(subfolder + '/SQ_map1-{}_id1-{}_log.png'.format(run, cid))

                    print('\n')
        
        print("Memory usage: ",tracemalloc.get_traced_memory(), flush=True)
                  
        
    print("Total memory usage: ",tracemalloc.get_traced_memory(), flush=True)
    tracemalloc.stop()

    #--------------------------------------------#
    # end of loop over runs

    # print all map2map comparison values
    if False:
        print('mappings:-----------------------')
        #print(mapping)

        for map1 in mapping:
            print(map1)
            for id1 in mapping[map1]:
                print(' ', id1)
                for map2 in mapping[map1][id1]:
                    print('  ', map2)

                    #best mapping of map1 id1 <-> map2 id2 is the one with the largest val
                    # this is one measure of how good the correspondence is

                    for id2 in mapping[map1][id1][map2]:
                        val = mapping[map1][id1][map2][id2]

                        # values we print here are:
                        #(intersect_area, union_area, area(map1, id1), area(map2,id2))
                        print('   ', id2, val)



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
