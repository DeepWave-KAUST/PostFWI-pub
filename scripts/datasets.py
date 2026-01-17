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
            for batch in self.batch:
                label = self.load_every(batch)       
                if label is not None:
                    self.label_list.append(label)
        

    def load_every(self, batch):
    
        batch = batch.split('\t')
        label_path = batch[0]  
        print(label_path)

        label = np.load(label_path)
        label = label.astype('float32')
       
        return label
 
    def __getitem__(self, idx):
    
          
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size

        label = np.copy(self.label_list[batch_idx][sample_idx,:,:]) if len(self.label_list) != 0 else None



        label = 2 * (label - 1478.6156) / (3587.2795  - 1478.6156) - 1
        label = label[:,:608]

        label = label[::2,::2]
        label = label.T
            
        return label[None,:,:],label[None,:,:]
    def __len__(self):
     
        return len(self.batch) * self.file_size



