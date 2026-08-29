import os
import numpy as np
from torch.utils.data import Dataset



class Dataset(Dataset):

    def __init__(self, anno, lines, preload=True,  file_size=500,
                    transform_data=None, transform_label=None,conditioning_key=False):
        if not os.path.exists(anno):
            print(f'Annotation file {anno} does not exists')
        self.preload = preload
        self.file_size = file_size
        with open(anno, 'r') as f:
            self.batches = f.readlines()
        self.batch=self.batches[:lines]
        
        if preload: 
            self.label_list = []
            self.data_list = []
            for batch in self.batch:
                label,data = self.load_every(batch)       

                if label is not None:
                    self.label_list.append(label)
                if data is not None:
                    self.data_list.append(data)       

    def load_every(self, batch):
    
        batch = batch.split('\t')
   
        label_path = batch[0].strip()
        data_path = batch[1].strip()
   

        label = np.load(label_path).astype('float32')
        data = np.load(data_path)
        
        return label,data
    
    def __getitem__(self, idx):
    
 
          
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size

        label = np.copy(self.label_list[batch_idx][sample_idx,:,:]) if len(self.label_list) != 0 else None
        
        data = np.copy(self.data_list[batch_idx][sample_idx,:]) if len(self.data_list) != 0 else None
        label = 2 * (label - 1000) / (5000-1000) - 1

        if np.random.randint(0, 2) == 1:
            label = np.copy(np.flip(label, axis=1))
            

        return label[None,...], data

    
    def __len__(self):
     
        return len(self.batch) * self.file_size



