import torch
from torch.utils.data import Dataset
import torchvision

from petrel_client.client import Client
client = Client()
from datetime import datetime, timedelta
import calendar
import io
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

def get_profile_data_multiprocess(temp_args):
    client, file_path, mean, std = temp_args
    res = client.get(file_path)
    buffer = io.BytesIO(res)
    return (np.load(buffer) - mean) / std
    
height_levels = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
class era5_npy_f32(Dataset):
    def __init__(self, year_start,year_end, lead_time):
        super().__init__()
        
        self.years = list(range(year_start,year_end+1))
        self.monthes = list(range(1,13))
        self.lead_time = lead_time

        self.singleVar = ['u10', 'v10', 't2m', 'msl']
        self.multiVar = ['z', 'q', 'u', 'v', 't']
        self.p_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        
        files_path = self.get_filelist()
        with open('files_path.json', 'w') as f:
            json.dump(files_path, f, indent=4) 
        with open('files_path.txt', 'w') as f:
            for item in files_path:
                f.write(f"input1: {item['input1']}, input2: {item['input2']}, target: {item['target']}\n")
        
        x = list(enumerate(files_path))
        random.shuffle(x)
        indices, input_names = zip(*x)
        
        self.file_names = input_names
        self.mean_std = self._get_meanstd()
        #self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        
    def get_filelist(self):
        files_path = []
        for year in self.years:
            for month in self.monthes:
                _, days = calendar.monthrange(year,month)
                for day in range(1,days+1):
                    for hour in range(0,24,6):
                        date = datetime(year, month, day, hour)
                        input1_timestamp = date.strftime("/%Y/%Y-%m-%d/%H:%M:%S")
                        input2_date = date + timedelta(hours=6)
                        input2_timestamp = input2_date.strftime("/%Y/%Y-%m-%d/%H:%M:%S")
                        
                        #input1_timestamp and input2_timestamp are input for FengWu to predict the lead-time background field
                        #the input for fengwu size is 69x721x1440. 69 is the number of varibles 721*1440 is the resolution
                        #FengWu will predict the next 6h weather state after INPUT2_TIMESTAMP auto-regressively
                        
                        leadtime_date = input2_date + timedelta(hours=self.lead_time)
                        leadtime_timestamp = leadtime_date.strftime("/%Y/%Y-%m-%d/%H:%M:%S")

                        #Thus, a batch data should include three 69*721*1440 arrays, which are (input1 input2) to fengwu and the truth
                        #The 69 varibels are: single_varibles ['u10', 'v10', 't2m', 'msl'] + multi_level_varibles: ['z', 'q', 'u', 'v', 't']*[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

                        input1_list = []
                        input2_list = []
                        leadtime_list = []
                        #here we generate the filelist for the single varibles
                        bucket = "s3://era5_np_float32/single"
                        for var in self.singleVar:
                            input1_file = bucket + input1_timestamp + "-{}.npy".format(var)
                            input2_file = bucket + input2_timestamp + "-{}.npy".format(var)
                            leadtime_file = bucket + leadtime_timestamp + "-{}.npy".format(var)
                            input1_list.append(input1_file)
                            input2_list.append(input2_file)
                            leadtime_list.append(leadtime_file)
                        #here we generate the filelist for the multilevel varibles
                        bucket = "s3://era5_np_float32"
                        for var in self.multiVar:
                            for plevel in self.p_levels:
                                input1_file = bucket + input1_timestamp + "-{}-{}.0.npy".format(var,plevel)
                                input2_file = bucket + input2_timestamp + "-{}-{}.0.npy".format(var,plevel)
                                leadtime_file = bucket + leadtime_timestamp + "-{}-{}.0.npy".format(var,plevel)
                                input1_list.append(input1_file)
                                input2_list.append(input2_file)
                                leadtime_list.append(leadtime_file)
                        #the three file_lists include the file path as ['u10', 'v10', 't2m', 'msl', z50, z100, ..., z1000, q50, q100, ..., q1000, t50, t100, ..., t1000]
                        #the three file_lists have different timestamps

                        temp = {'input1':input1_list,'input2':input2_list,'target':leadtime_list}
                        files_path.append(temp)
        return files_path
                                
    def _get_meanstd(self):
        with open('./datasets/mean_std_single.json',mode='r') as f:
            single_level_mean_std = json.load(f)
        
        with open('./datasets/mean_std.json',mode='r') as f:
            multi_level_mean_std = json.load(f)
        
        mean_std = []
        #mean_std :['u10', 'v10', 't2m', 'msl', z50, z100, ..., z1000, q50, q100, ..., q1000, t50, t100, ..., t1000]
        for var in self.singleVar:
            mean_std.append(np.array([single_level_mean_std['mean'][var][0],single_level_mean_std['std'][var][0]]))
        for var in self.multiVar:
            for plevel in self.p_levels:
                mean_std.append(np.array([multi_level_mean_std['mean'][var][height_levels.index(plevel)],multi_level_mean_std['std'][var][height_levels.index(plevel)]]))
        return mean_std
        

    def get_profiles(self, index):
        input1_list, input2_list, target_list = self.file_names[index]["input1"], self.file_names[index]["input2"], self.file_names[index]["target"]

        with ThreadPool() as pool:
            temp_args = []
            for ivar in range(len(input1_list)):
                mean_std = self.mean_std[ivar]
                mean, std = mean_std[0], mean_std[1]
                temp_args.append((client, input1_list[ivar], mean, std))
                temp_args.append((client, input2_list[ivar], mean, std))
                temp_args.append((client, target_list[ivar], mean, std))

            results = pool.map(get_profile_data_multiprocess, temp_args)

        # 拆分结果
        input1_data = results[::3]
        input2_data = results[1::3]
        target_data = results[2::3]

        input1 = np.stack(input1_data)
        input2 = np.stack(input2_data)
        target = np.stack(target_data)

        return [input1,input2,target]
    
    def _get_profiles(self, index):#可以进一步优化成并行加载
        #read a single batch data
        input1_list,input2_list,target_list = self.file_names[index]["input1"],self.file_names[index]["input2"],self.file_names[index]["target"]
        input1_data = []
        input2_data = []
        target_data = []

        for ivar in range(len(input1_list)):
            mean_std = self.mean_std[ivar]
            mean = mean_std[0]
            std = mean_std[1]
            
            #read input1
            res = client.get(input1_list[ivar])
            buffer = io.BytesIO(res)
            input1_data.append((np.load(buffer)-mean)/std)
            

            #read input2
            res = client.get(input2_list[ivar])
            buffer = io.BytesIO(res)
            input2_data.append((np.load(buffer)-mean)/std)

            #read target
            res = client.get(target_list[ivar])
            buffer = io.BytesIO(res)
            target_data.append((np.load(buffer)-mean)/std)
        
        input1 = np.stack(input1_data)
        input2 = np.stack(input2_data)
        target = np.stack(target_data)
        #the data input to fengwu normaized ?

        return [input1,input2,target]
        
    def __getitem__(self, index):
        res = self.get_profiles(index)
        return res

    def __len__(self):
        return len(self.file_names)

