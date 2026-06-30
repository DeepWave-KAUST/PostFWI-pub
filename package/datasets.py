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
        batch_idx = idx // self.file_size

        label = np.copy(self.label_list[batch_idx][::2, ::2]).astype(np.float32)


        label = 2 * (label - 1000) / 4000 - 1

        label = label[None, None, :320,275+304*0:275+304*2]
        print(np.copy(self.label_list[batch_idx][:, ::1]).astype(np.float32).shape, print(label.shape))
        return label
      
    def __len__(self):
     
        return len(self.batch) * self.file_size



