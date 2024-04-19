import torch
from customdataset import FullDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
def load_data():
    full_dataset=FullDataset('./fulldataset/sample_full.json')
    train_dataset, valtest_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])
    val_dataset, test_dataset = torch.utils.data.random_split(valtest_dataset, [0.5,0.5])
    print('full size',len(full_dataset))
    print('train size', len(train_dataset))
    print('val size', len(val_dataset))
    print('test size', len(test_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader

def check_dataloader(train_dataloader, val_dataloader, test_dataloader):
    for index, batch in tqdm(enumerate(val_dataloader)):
        input_texts, prompts, output_texts = batch['input_text'], batch['prompt'], batch['output_text']
        print(input_texts, prompts, output_texts)

if __name__=='__main__':
    train_dataloader, val_dataloader, test_dataloader = load_data()
    check_dataloader(train_dataloader, val_dataloader, test_dataloader)
