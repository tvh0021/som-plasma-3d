import argparse
import os
import matplotlib.pyplot as plt
import shutil
import numpy as np
import h5py as h5
from scipy.signal import savgol_filter

parser = argparse.ArgumentParser(description="Use multimap mapping to analyze and segment groups of features")
parser.add_argument('--file_path', type=str, dest='file_path', help='Multimap mapping file path')
parser.add_argument("--copy_clusters", dest='copy_clusters', action='store_true', help="Copy the clusters to a new folder")
parser.add_argument("--save_combined_map", dest='save_combined_map', action='store_true', help="Save the combined map of all clusters")
parser.add_argument("--threshold", type=float, dest='threshold', default=-0.015, help="Threshold for the derivative of the gsum values")
parser.add_argument("--reference_file", type=str, dest='reference_file', default='/mnt/home/tha10/ceph/SOM-tests/pipeline-test/features_2j1b1e0r_2800.h5', help="Reference file to compare the clusters to", required=False)
parser.add_argument('--slice', type=int, dest='slice', default=580, help='Slice number, make sure this matches the slice number in the sce_slice.py call')
args = parser.parse_args()

def makeFilename (n : int) -> str:
    if n < 10:
        file_n = '000' + str(n)
    elif (n >= 10) & (n < 100):
        file_n = '00' + str(n)
    else:
        file_n = '0' + str(n)

    return f"{file_n}.png"

if __name__ == '__main__':
    
    mapping = dict()
    
    # read in the multimap mapping file and store in a dict that includes the file as key name, and the cluster_id and gsum as values
    with open(args.file_path + "/multimap_mappings.txt", 'r') as f:
        for line in f:
            line = line.strip("\n")
            if not line[0].isnumeric():
                key_name = line.strip("clusters_").strip(".npy")
                mapping[key_name] = []
            else:
                mapping[key_name].append(line.split(" "))
                
    # print("Keys", mapping.keys())
    # print("Final map", mapping)
    
    # convert the dict to a list to sort more easily
    map_list = []
    for key in mapping.keys():
        map_list.extend([[float(i[1]), int(i[0]), key] for i in mapping[key]])
    # print("Map list length", len(map_list))
            
    # sort the list based on gsum value
    map_list.sort(key=lambda map_list: map_list[0], reverse=True)
    print("Sorted map", map_list[0])
    print("Length of sorted map", len(map_list))
    
    # now iterate through the list and copy the files to the appropriate cluster folder
    if args.copy_clusters:
        ranked_clusters_dir = os.path.join(args.file_path, "ranked-clusters")
        if not os.path.exists(ranked_clusters_dir):
            os.makedirs(ranked_clusters_dir)
        
        for i in range(len(map_list)):
            origin_file_name = '{}/mask3d-clusters_{}.npy-id{}.png'.format(args.file_path, map_list[i][2], map_list[i][1])
            destination_file_name = '{}/ranked-clusters/{}'.format(args.file_path, makeFilename(i))
            shutil.copyfile(origin_file_name, destination_file_name)
            
        print("Done copying files")
        
    # apply a Savitzky-Golay filter to smooth the gsum values
    smooth_fraction = 10
    order = 4
    smoothed_map = np.array([map_list[i][0] for i in range(len(map_list))])
    print("Applying Savitzky-Golay filter")
    smoothed_map = savgol_filter(smoothed_map, len(map_list)//smooth_fraction, order, deriv=0)
    
    # compute the derivative of the gsum values to find the drop
    gsum_deriv = savgol_filter(smoothed_map, len(map_list)//smooth_fraction, order, deriv=1) / smoothed_map
    
    # iterate through the derivative and find the local minima
    threshold = args.threshold
    
    # gsum_deriv = np.load(f'/mnt/home/tha10/ceph/SOM-tests/hr-d3x640/{folder}/SCE/gsum_deriv_smoothed_10_4.npy')
    cluster_order = np.array(list(range(1,len(gsum_deriv)+1)))

    if not args.save_combined_map:
        plt.figure(dpi=300)
        plt.plot(cluster_order, gsum_deriv, marker='o', c='k', markersize=2, linewidth=1)
        # plt.yscale('log')
        plt.ylim(-0.06, 0.0)
        plt.title(f"Sorted gsum derivatives")
        plt.xlabel("Ranked clusters")
        plt.ylabel("Gsum derivative")
        plt.grid()
        plt.hlines(args.threshold, 0, len(cluster_order))
        plt.savefig(f"{args.file_path}/gsum_deriv.png")
        print("Saved gsum derivative plot")
    
    if args.save_combined_map: # change the threshold to the appropriate value before running this
        threshold_crossed = True if gsum_deriv[0] < threshold else False
        
        peak_locations = []
        for i in range(1,len(gsum_deriv)-1):
            if (gsum_deriv[i] < threshold) & (gsum_deriv[i] < gsum_deriv[i-1]) & (gsum_deriv[i] < gsum_deriv[i+1]) & (threshold_crossed == True):
                # print("Local minima found at index ", i)
                peak_locations.append(i)
                threshold_crossed = False
            
            if (gsum_deriv[i] > threshold) & (threshold_crossed == False):
                threshold_crossed = True
        
        print("Peak locations", peak_locations)
        
        # from the local minima, find the ranges of the clusters
        cluster_ranges = []
        for i in range(len(peak_locations)-1):
            if i == 0:
                cluster_ranges.append([0, peak_locations[i]])
            
            cluster_ranges.append([peak_locations[i], peak_locations[i+1]])
            
            if i == len(peak_locations)-2:
                cluster_ranges.append([peak_locations[i+1], len(map_list)])
        
        print("Cluster ranges", cluster_ranges)
        print("Number of clusters", len(cluster_ranges))
        
        
        # map the clusters back into output clusters
        remapped_clusters = dict()
        
        for i in range(len(cluster_ranges)):
            start_pointer = cluster_ranges[i][0]
            end_pointer = cluster_ranges[i][1]
            
            key_name = str(i)
            remapped_clusters[key_name] = []
            
            # print("Start pointer", start_pointer)
            # print("End pointer", end_pointer)
            
            for j in range(start_pointer, end_pointer):
                # print (map_list[j])
                # print (i)
                remapped_clusters[key_name].append(map_list[j])
        
        print("Length of remapped clusters : ", [len(remapped_clusters[k]) for k in remapped_clusters.keys()])
        print ("First cluster : ", remapped_clusters['0'])
        
        # add values of the binary map of each cluster to obtain a new map
        # read in the binary map
        all_binary_maps = np.empty((len(remapped_clusters), 640, 640), dtype=float)
        for cluster in remapped_clusters.keys():
            print("Currently analyzing cluster : ", cluster)
            print("Number of instances in cluster : ", len(remapped_clusters[cluster]))
            for instance in remapped_clusters[cluster]:
                # print("Currently analyzing binary map : ", instance)
                
                signal_strength_map = np.load(args.file_path + "/mask-clusters_{}.npy-id{}.npy".format(instance[2],instance[1])) # load the binary map
                # print(instance[0])
                all_binary_maps[int(cluster)][:,:,:] += signal_strength_map # binary map should have dimensions 640x640
                
        # save the new binary map
        np.save(args.file_path + "/all_binary_maps.npy", all_binary_maps)
        print("Saved new binary map")
        
        # plot the new binary map with j_par as reference
        # load h5 file as comparison
        f_in = h5.File(args.reference_file,"r")
        
        dataset = f_in['features'][()]
        feature_names = f_in['names'][()]
        
        all_data = np.array(dataset)
        # print("Shape of all data", all_data.shape)
        nx = int(np.cbrt(all_data.shape[0]))
        ny = nx
        nz = nx
        j_par = np.reshape(all_data[:,feature_names == b'j_par'], newshape=[nx,ny,nz])
        slice_number = args.slice
        
        number_of_clusters = all_binary_maps.shape[0]
        print("Identified {} clusters.".format(number_of_clusters))

        ncols = 3
        nrows = int(np.ceil((number_of_clusters+1) / ncols))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,10 / ncols * nrows), sharex=True, sharey=True, dpi=300)
        # fig, axs = plt.subplots(nrows=number_of_clusters+1, ncols=1, figsize=(4,number_of_clusters*4), dpi=200)

        ref = axs[0,0].pcolormesh(j_par[slice_number,:,:], cmap='RdBu', vmin=-1.5, vmax=1.5)
        axs[0,0].set_aspect('equal')
        # fig.colorbar(ref, ax=axs[0], shrink=0.9)
        axs[0,0].set_title('j_par')

        for i, file in enumerate(all_binary_maps):
            a, b = divmod(i+1, ncols)
            axs[a, b].pcolormesh(all_binary_maps[i,:,:], cmap='Reds', vmin=0)
            axs[a, b].set_aspect('equal')
            axs[a, b].set_title(f"Cluster {i}")
            
        plt.savefig(f"{args.file_path}/combined_binary_map.png")
        print("Saved combined binary map")
            
        
        