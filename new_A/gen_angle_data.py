import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel , delayed
from numpy.lib.format import open_memmap
from ThirdOrderRep import getThridOrderRep

# bone

def gen_angle_data_one_num_worker(path,train_path,val_path,test_path):
    
    
    new_train_x = open_memmap('new_train_x.npy',dtype='float32',mode='w+',shape=(16724, 9, 300, 17, 2))
    new_val_x = open_memmap('new_val_x.npy',dtype='float32',mode='w+',shape=(2000, 9, 300, 17, 2))
    new_test_x= open_memmap('new_test_x.npy',dtype='float32',mode='w+',shape=(4307, 9, 300, 17, 2))

    train_x = np.load(train_path,mmap_mode='r')
    val_x = np.load(val_path,mmap_mode='r')
    test_x  = np.load(test_path,mmap_mode='r')
    print('train_x.shape')


    N_train,_, T_train, _ , _= train_x.shape
    N_val,_, T_val, _,_= val_x.shape
    N_test,_, T_test, _,_= test_x.shape

    train_x = train_x.reshape((N_train, T_train, 2, 17, 3)).transpose(0, 4, 1, 3, 2)
    val_x = val_x.reshape((N_val, T_val, 2, 17, 3)).transpose(0, 4, 1, 3, 2)
    test_x = test_x.reshape((N_test, T_test, 2, 17, 3)).transpose(0, 4, 1, 3, 2)

    Parallel(n_jobs=8)(delayed(lambda i: new_train_x.__setitem__(i,getThridOrderRep(train_x[i])))(i) for i in tqdm(range(N_train)))
    Parallel(n_jobs=8)(delayed(lambda i: new_val_x.__setitem__(i,getThridOrderRep(val_x[i])))(i) for i in tqdm(range(N_val)))
    Parallel(n_jobs=8)(delayed(lambda i: new_test_x.__setitem__(i,getThridOrderRep(test_x[i])))(i) for i in tqdm(range(N_test)))

    # new_train_x = new_train_x.transpose(0, 2, 4, 3, 1).reshape(N_train, T_train, -1)
    # new_test_x = new_test_x.transpose(0, 2, 4, 3, 1).reshape(N_test, T_test, -1)

    np.save(f'{path[:-4]}_angle_train.npy', new_train_x)
    np.save(f'{path[:-4]}_angle_val.npy', new_val_x)
    np.save(f'{path[:-4]}_angle_test.npy', new_test_x)
if __name__ == '__main__':
    gen_angle_data_one_num_worker('data','data/train_joint.npy','data/val_joint.npy','data/test_joint.npy')
    