import os
import threading

import argparse

parser = argparse.ArgumentParser(description='popsom code')
parser.add_argument("--script_location", type=str, dest='script_location', default='~/git_repos/som-plasma-3d/som3d.py')
parser.add_argument("--features_path", type=str, dest='features_path', default='/mnt/ceph/users/tha10/SOM-tests/hr-d3x640/')
parser.add_argument("--file", type=str, dest='file', default='features_4j1b1e_2800.h5')
parser.add_argument('--save_neuron_values', dest='save_neuron_values', action='store_true', help="Save the neuron values to a file")
args = parser.parse_args()

path = args.script_location
xdim = list(range(20, 25, 2))
alpha = [0.1]
train = [2000000, 4000000, 6000000, 8000000, 10000000]
batch = [0]
save_neuron_values = False
features_path = args.features_path
file = args.file

neuron_flag = ""
if save_neuron_values:
    neuron_flag = "--save_neuron_values"

def run_file(path, xdim, alpha, train, batch, save_neuron_values, features_path, file):
    dim = xdim
    batch_print = " --batch " + str(batch) if batch != 0 else ""
    os.system("python3 " + path + " --xdim " + str(dim) + " --ydim " + str(dim) + " --alpha " + str(alpha) + " --train " + str(train) + batch_print + neuron_flag + " --features_path " + features_path + " --file " + file)
    print("Finished dim: " + str(dim) + ", alpha: " + str(alpha) + ", train: " + str(train) + ", batch: " + str(batch) , flush=True)

def print_excecution(path, xdim, alpha, train, batch, save_neuron_values, features_path, file):
    dim = xdim
    batch_print = " --batch " + str(batch) if batch != 0 else ""
    print("python3 " + path + " --xdim " + str(dim) + " --ydim " + str(dim) + " --alpha " + str(alpha) + " --train " + str(train) + batch_print + neuron_flag + " --features_path '" + features_path + "' --file '" + file + "' &")

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
