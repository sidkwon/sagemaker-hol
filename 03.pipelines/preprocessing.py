
import glob
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

if __name__=='__main__':
    
    # ----------------------------------------------------------------------
    # TO DO : 입력파일을 로드 경로를 지정합니다..
    input_files = glob.glob('{}/*.npy'.format('/opt/ml/processing/input'))
    
#     input_files = <TO DO>
    
    # ----------------------------------------------------------------------
       
    print('\nINPUT FILE LIST: \n{}\n'.format(input_files))
    scaler = StandardScaler()
    for file in input_files:
        raw = np.load(file)
        transformed = scaler.fit_transform(raw)
        
        if 'train' in file:
    
            # ----------------------------------------------------------------------     
            # TO DO : 출력파일이 저장될 경로를 지정합니다.
            output_path = os.path.join('/opt/ml/processing/train', 'x_train.npy')
            
#             output_path = <TO DO>
            
            # ----------------------------------------------------------------------       
            
            np.save(output_path, transformed)
            print('SAVED TRANSFORMED TRAINING DATA FILE\n')
        else:
            
            # ----------------------------------------------------------------------       
            # TO DO : 출력파일이 저장될 경로를 지정합니다.
            output_path = os.path.join('/opt/ml/processing/test', 'x_test.npy')
            
#             output_path = <TO DO>
            
            # ----------------------------------------------------------------------
            
            np.save(output_path, transformed)
            print('SAVED TRANSFORMED TEST DATA FILE\n')
