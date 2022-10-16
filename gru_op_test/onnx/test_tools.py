import numpy as np

def save_data_to_local(data, file_path, dtype):
    if dtype == 'int32':
        np.savetxt(file_path, data, fmt='%d')
    elif dtype == 'float32':
        np.savetxt(file_path, data, fmt='%.32f')
    
    
def read_data_from_txt(file_path, dtype):
    if dtype == 'int32':
        return np.loadtxt(file_path, dtype=np.int32)
    elif dtype == 'float32':
        return np.loadtxt(file_path, dtype=np.float32)

def save_and_read_data(data, file_path, dtype):
    save_data_to_local(data, file_path, dtype)
    return read_data_from_txt(file_path, dtype)
    
if __name__ == '__main__':
    data = np.random.randn(15)
    file_path = './test_data/test_data.txt'
    save_data_to_local(data, file_path)
    
    np_array = read_data_from_txt(file_path)
    print(read_data_from_txt(file_path))
    print("finish...")
    