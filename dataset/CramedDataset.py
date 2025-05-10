import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import random


# class CramedDataset(Dataset):

#     def __init__(self, args, mode='train', class_imbalanced=False):
#         self.args = args
#         self.image = []
#         self.audio = []
#         self.label = []
#         self.mode = mode

#         self.data_root = './data/'
#         class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

#         self.visual_feature_path = r'D:\yunfeng\data\CREMA-D'
#         self.audio_feature_path = r'D:\yunfeng\data\CREMA-D\Audio-1004'
#         # self.audio_feature_path = '../CREMAD/Audio-299'

#         self.train_csv = os.path.join(self.data_root, args.dataset + '/train.csv')
#         self.test_csv = os.path.join(self.data_root, args.dataset + '/test.csv')

#         if mode == 'train':
#             csv_file = self.train_csv
#         else:
#             csv_file = self.test_csv

#         if class_imbalanced and mode == 'train':
#             with open(self.train_csv, encoding='UTF-8-sig') as f:
#                 csv_reader = csv.reader(f)
#                 subset_class = [[] for _ in range(len(class_dict))]  # instances for each class
#                 class_sample_num = [0 for _ in range(len(class_dict))]
#                 for item in csv_reader:
#                     subset_class[class_dict[item[1]]].append(item[0]+' '+item[1])
#                     class_sample_num[class_dict[item[1]]] += 1

#             max_num = class_sample_num[0]
#             num_perclass = []
#             for ii in range(len(class_dict), 0, -1):
#                 num_perclass.append(max_num//len(class_dict)*ii)

#             imbalanced_dataset = []
#             print(num_perclass)
#             # [972, 810, 648, 486, 324, 162]

#             for ii in range(len(class_sample_num)):
#                 random.shuffle(subset_class[ii])
#                 imbalanced_dataset += subset_class[ii][:num_perclass[ii]]

#             random.shuffle(imbalanced_dataset)
#             for item in imbalanced_dataset:
#                 audio_path = os.path.join(self.audio_feature_path, item.split(' ')[0] + '.pkl')
#                 visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.fps), item.split(' ')[0])

#                 if os.path.exists(audio_path) and os.path.exists(visual_path):
#                     self.image.append(visual_path)
#                     self.audio.append(audio_path)
#                     self.label.append(class_dict[item.split(' ')[1]])
#                 else:
#                     continue
#         else:
#             with open(csv_file, encoding='UTF-8-sig') as f2:
#                 csv_reader = csv.reader(f2)
#                 for item in csv_reader:
#                     audio_path = os.path.join(self.audio_feature_path, item[0] + '.pkl')
#                     visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.fps), item[0])

#                     if os.path.exists(audio_path) and os.path.exists(visual_path):
#                         self.image.append(visual_path)
#                         self.audio.append(audio_path)
#                         self.label.append(class_dict[item[1]])
#                     else:
#                         continue


#     def __len__(self):
#         return len(self.image)

#     def __getitem__(self, idx):

#         spectrogram = pickle.load(open(self.audio[idx], 'rb'))

#         if self.mode == 'train':
#             transform1 = transforms.Compose([
#                 transforms.RandomResizedCrop(224),
#                 transforms.ToTensor(),
#             ])
#             transform2 = transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])
#         else:
#             transform1 = transforms.Compose([
#                 transforms.Resize(size=(224, 224)),
#                 transforms.ToTensor(),
#                 # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])
#             transform2 = transforms.Compose([
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])

#         # Visual
#         image_samples = os.listdir(self.image[idx])
#         np.random.seed(999)
#         select_index = np.random.choice(len(image_samples), size=self.args.num_frame, replace=False)
#         select_index.sort()
#         images = torch.zeros((self.args.num_frame, 3, 224, 224))
#         var = self.args.var
#         for i,n in enumerate(select_index):
#             img = Image.open(os.path.join(self.image[idx], image_samples[n])).convert('RGB')
#             img = transform1(img)
#             noise = np.random.normal(0, var, img.shape)
#             img = img + noise
#             img = transform2(img)
#             images[i] = img

#         images = torch.permute(images, (1,0,2,3))

#         # label
#         label = self.label[idx]

#         return spectrogram, images, label
    
    
class CramedDataset(Dataset):

    def __init__(self, mode='train', 
                 train_path='/home/rakib/Multi-modal-Imbalance/data/CREMAD/train.csv',
                 test_path='/home/rakib/Multi-modal-Imbalance/data/CREMAD/test.csv',
                 visual_path='/home/rakib/Multimodal-Datasets/CREMA-D/Image-01-FPS',
                 audio_path='/home/rakib/Multimodal-Datasets/CREMA-D/AudioWAV'):
        
        self.mode = mode
        self.class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}

        self.visual_path = visual_path
        self.audio_path = audio_path

        # Use the appropriate CSV file depending on the mode (train or test)
        csv_file = train_path if mode == 'train' else test_path

        self.image = []
        self.audio = []
        self.label = []

        # Load data from CSV
        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_path, item[0] + '.wav')
                visual_path = os.path.join(self.visual_path, item[0])
                
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(self.class_dict[item[1]])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Load label
        label = self.label[idx]

        ### Audio Processing ###
        # Load and process audio with librosa
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        # Ensure we have 3 seconds of audio by tiling the sample if needed
        resamples = np.tile(samples, 3)[:22050 * 3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        
        # Compute the STFT and log-scale the spectrogram
        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        
        # Convert the spectrogram to a torch tensor
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

        ### Visual Processing ###
        # Define the transformations (different for train and test)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224) if self.mode == 'train' else transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip() if self.mode == 'train' else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Sample 3 frames from the image directory
        visual_path = self.image[idx]
        image_samples = sorted(os.listdir(visual_path))  # Get all image files
        pick_num = 3  # Fixed number of frames like in AVDataset
        seg = int(len(image_samples) / pick_num)  # Evenly spaced frame selection
        
        image_arr = []
        for i in range(pick_num):
            tmp_index = int(seg * i)
            img = Image.open(os.path.join(visual_path, image_samples[tmp_index])).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(1)  # Add a channel dimension for concatenation
            image_arr.append(img)

        # Concatenate the 3 sampled frames into a single tensor (along the channel dimension)
        image_n = torch.cat(image_arr, dim=1)

        ### Return the data in the format required by AVDataset ###
        return spectrogram, image_n, label