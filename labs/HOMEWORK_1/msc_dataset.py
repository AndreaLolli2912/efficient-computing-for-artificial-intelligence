import torch
import torchaudio
import os

class MSCDataset(torch.utils.data.Dataset):
    def __init__(self, dataPath:str, classes:list[str])->None:
        """Initialize the dataset by loading data from the specified path and preparing label mappings.
        
        Args:
            dataPath (str): Path to the dataset file.
            classes (list[str]): List of class labels.
        """
        super(MSCDataset, self).__init__()
        self.__classes = classes
        
        # creating the dictionary associating name label to integer index
        self.__convertedLabels:dict[str, int] = {label.strip().lower():i for i,label in enumerate(classes)}
        # listing all the files in the dataset path along with the converted label, accounting only for .wav files that starts with a valid label
        # sorting by folder name to ensure consistent ordering of samples
        self.__filesPath:list[list[str, int]] = sorted([[os.path.join(dataPath, file), self.__convertedLabels[file.split("_")[0].strip().lower()]] 
                                                for file in os.listdir(dataPath) if file.endswith(".wav") and file.split("_")[0].strip().lower() in self.__convertedLabels], 
                                                    key=lambda x: os.path.split(x[0])[0]) 
        
    @property
    def classes(self)->list[str]:
        """Get the list of class labels.

        Returns:
            classes (list[str]): List of class labels.
        """
        return list(self.__classes)
    
    
    def __len__(self)->int:
        """Return the total number of samples in the dataset.
        
        Returns:
            Number of samples (int): Total number of samples.
        """
        return len(self.__filesPath)


    def __getitem__(self, index:int)->dict[str, int|torch.Tensor]:
        """Retrieve a sample from the dataset at the specified index.
        
        Args:
            index (int): Index of the sample to retrieve.
            
        Returns:
            data (dict): A dictionary containing:
                - 'x' (torch.Tensor): The audio waveform tensor.
                - 'sampling_rate' (int): The sampling rate of the audio.
                - 'label' (int): The integer label corresponding to the audio class.
        """ 
        audio, sampling_rate = torchaudio.load(self.__filesPath[index][0], normalize=False)
        
        return {
            'x': audio,
            'sampling_rate': sampling_rate,
            'label': self.__filesPath[index][1]
        }
    
    def label_to_int(self, label:str)->int:
        """Convert a string label to its corresponding integer label.
        
        Args:
            label (str): The string label to convert.
        """
        return self.__convertedLabels[label.strip().lower()]
    
    
    def getConvertedLabels(self)-> dict [int,str]:
        """Get the mapping of integer labels to string labels.
        
        Returns:
            convertedLabels (dict[int, str]): A dictionary mapping integer labels to string labels.
        """
        return dict(self.__convertedLabels)
    
    def getInvertedConvertedLabels(self)->dict[str, int]:
        """Get the mapping of string labels to integer labels.
        
        Returns:
            invertedConvertedLabels (dict[str, int]): A dictionary mapping string labels to integer labels.
        """
        return {v:k for k,v in self.__convertedLabels.items()}
    
    
if __name__ == "__main__":
    dataset = MSCDataset("./Laboratories/HOMEWORK_1/data/msc-test/", ["down", "no", "go", "yes", "stop", "up", "right", "left"])
    
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"Sample {i}: Label={sample['label']}, Shape={sample['x'].shape}, Sampling Rate={sample['sampling_rate']}")
        assert sample['sampling_rate'] == 16000, "Sampling rate should be 16000"