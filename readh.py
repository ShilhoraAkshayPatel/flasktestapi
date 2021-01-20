import h5py
filename = "res-9-model.h5"


f = h5py.File('res-9-model.h5', 'r')
keys = list(f.keys())
print(keys)
dset = f['model_weights']
print(dset.name)
layers = list(dset.items())
print(layers)

weights1 = f['model_weights']["/model_weights/dense_6"]['dense_6']['kernel:0']
print(weights1[:])
