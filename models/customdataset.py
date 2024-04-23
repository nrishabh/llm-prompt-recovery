import torch
import os
from torch.utils.data import Dataset
import json

class FullDataset(Dataset):
    '''
    A dataset with input text, prompt, output text
    '''
    def __init__(self, json_file):
        '''
        Arguments:
        json_file: json dataset file path
        ---------------------------------
        self.dataset: loaded json file into a list of dictionaries(id->int, input_text->str, prompt->str, ouput_text->str)
        '''
        self.json_file=json_file
        f = open(self.json_file)
        self.dataset=json.load(f) # a list of dicts [{'id':int, 'input_text':str, 'prompt':str, 'output_text':str}]
        f.close()
    
    def __len__(self):
        '''
        Return:
        size of the dataset(int)
        '''
        return len(self.dataset)
    
    def __getitem__(self,idx):
        '''
        Return:
        the IDXth entry in dataset
        each entry has id, input_text, prompt, output_text
        return input_text, prompt, output_text
        '''
        return self.dataset[idx]
