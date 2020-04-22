import pandas as pd 
import numpy as np 
import os 
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

def readData(PATH, train_file, test_file):
    """Read files and return padaframes
    
    Parameters
    ----------
    PATH : os path
    train_files : training file
    test_files : test file
    
    Return
    ------
    train and test dataframes.
    """
    
    # if len(train_files == 0):
    #     print ("Error no training files")
    #     exit()
    # if len(test_files == 0):
    #     print ("Error no training files")
    
    train_path = os.path.join(PATH, train_file)
    test_path = os.path.join(PATH, test_file)
    
    train = pd.read_csv(train_path, dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv(test_path, dtype={'time': np.float32, 'signal': np.float32})
    return train, test 

def generatebatch(df : pd.DataFrame,
                   batch_size : int) -> pd.DataFrame :

    df['group'] = df.groupby(df.index//batch_size,sort=False)['signal'].agg(['ngroup']).values # check agg
    df['group'] = df['group'].astype(np.uint16)
    return df 

