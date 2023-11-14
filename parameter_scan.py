import os
import threading

import argparse

parser = argparse.ArgumentParser(description='popsom code')
parser.add_argument("--script_location", type=str, dest='script_location', default='~/git_repos/som-plasma-3d/som3d.py')
parser.add_argument("--features_path", type=str, dest='features_path', default='/mnt/ceph/users/tha10/SOM-tests/hr-d3x640/')
parser.add_argument("--file", type=str, dest='file', default='features_4j1b1e_2800.h5')
parser.add_argument('--save_neuron_values', type=bool, dest='save_neuron_values', default=False, help='Save neuron values to file?', required=False)
args = parser.parse_args()

path = args.script_location
xdim = list(range(10, 21, 5))
alpha = [0.05, 0.1, 0.5]
train = [10000, 50000, 100000]
batch = [160, 320]
save_neuron_values = False
features_path = args.features_path
file = args.file

def run_file(path, xdim, alpha, train, batch, save_neuron_values, features_path, file):
    dim = xdim
    os.system("python3 " + path + " --xdim " + str(dim) + " --ydim " + str(dim) + " --alpha " + str(alpha) + " --train " + str(train) + " --batch " + str(batch) + " --save_neuron_values " + str(save_neuron_values) + " --features_path " + features_path + " --file " + file)
    print("Finished dim: " + str(dim) + ", alpha: " + str(alpha) + ", train: " + str(train) + ", batch: " + str(batch) , flush=True)

def print_excecution(path, xdim, alpha, train, batch, save_neuron_values, features_path, file):
    dim = xdim
    print("python3 " + path + " --xdim " + str(dim) + " --ydim " + str(dim) + " --alpha " + str(alpha) + " --train " + str(train) + " --batch " + str(batch) + " --save_neuron_values " + str(save_neuron_values) + " --features_path '" + features_path + "' --file '" + file + "' &")

if __name__ == '__main__':
    for dim in xdim:
        for a in alpha:
            for steps in train:
                for window in batch:
                    # print("Starting dim: " + str(dim) + ", alpha: " + str(a) + ", train: " + str(steps) + ", batch: " + str(window) , flush=True)
                    # process = threading.Thread(target=run_file, args=(path, dim, a, steps, window, save_neuron_values, features_path, file))
                    # process.start()

                    print_excecution(path, dim, a, steps, window, save_neuron_values, features_path, file)

print("wait")
