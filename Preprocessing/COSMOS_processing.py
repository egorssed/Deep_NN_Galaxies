import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../Modules')
import COSMOS_preprocessing_utils as utils

def main(start,stop):

    images=np.zeros((0,64,64))
    labels=pd.DataFrame()

    for index in tqdm(range(start,np.minimum(stop,utils.cat.nobjects))):

        try:
            Success,image,parameters=utils.create_Galaxy(index,0.05)
        except:
            continue

        if np.logical_not(Success):
            continue

        #Push results to storages
        labels=labels.append(parameters,ignore_index=True)
        images=np.append(images,[image],axis=0)

    labels.to_csv('../Data/New dataset/Labels_Filtered_new_{start}_{stop}.csv'.format(start=start,stop=stop))
    np.save('../Data//New dataset/Images_Filtered_new_{start}_{stop}.npy'.format(start=start,stop=stop),images)

if __name__ == '__main__':
    arguments=[int(arg) for arg in sys.argv[1:]]
    if len(arguments)==2:
        start=arguments[0]
        stop=arguments[1]
        main(start,stop)

