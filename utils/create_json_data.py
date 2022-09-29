"""
Creates json files in the Medical Segmentation Decathlon datalist format containing 
dictionaries of image-label pairs for training and testing for each center. 
"""

import os
import json
import argparse
import nibabel as nib
from tqdm import tqdm
import random
import numpy as np

root = "/home/GRAMES.POLYMTL.CA/u114716/duke/projects/ms_brain_spine/data_processing"

parser = argparse.ArgumentParser(description='Code for creating data splits for each center')

parser.add_argument('-se', '--seed', default=42, type=int, help="Seed for reproducibility")
parser.add_argument('-dr', '--data_root', default=root, type=str, help='Path to the data set directory')
parser.add_argument('-ds', '--data_split', default='bwh', type=str, help='name of datasets to include', 
                    choices=['bwh', 'rennes', 'nih', 'amu', 'karo', 'milan', 'montpellier', 'ucsf', 'mix'])

args = parser.parse_args()

save_path = "/home/GRAMES.POLYMTL.CA/u114716/domain_incr_learning/datalists"

seed = args.seed
random.seed(seed)

fraction_test = 0.2

# create one json file with 80-20 train-test split
all_centers_subjects = os.listdir(args.data_root)

if args.data_split == 'amu':
    amu_subjects = [sub for sub in all_centers_subjects if sub.startswith('amu')]

    random.shuffle(amu_subjects); 
    
    test_subjects = amu_subjects[:int(len(amu_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = amu_subjects[int(len(amu_subjects) * fraction_test):]

elif args.data_split == 'bwh':
    bwh_subjects = [sub for sub in all_centers_subjects if sub.startswith('bwh')]

    random.shuffle(bwh_subjects); 
    
    test_subjects = bwh_subjects[:int(len(bwh_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = bwh_subjects[int(len(bwh_subjects) * fraction_test):]

elif args.data_split == 'karo':
    # NOTE: Karo has only T2 labels
    karo_subjects = [sub for sub in all_centers_subjects if sub.startswith('karo')]

    random.shuffle(karo_subjects); 
    
    test_subjects = karo_subjects[:int(len(karo_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = karo_subjects[int(len(karo_subjects) * fraction_test):]

elif args.data_split == 'milan':
    # NOTE: Milan has only T2 labels
    milan_subjects = [sub for sub in all_centers_subjects if sub.startswith('milan')]

    random.shuffle(milan_subjects); 
    
    test_subjects = milan_subjects[:int(len(milan_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = milan_subjects[int(len(milan_subjects) * fraction_test):]

elif args.data_split == 'montpellier':
    montpellier_subjects = [sub for sub in all_centers_subjects if sub.startswith('montpellier')]

    random.shuffle(montpellier_subjects); 
    
    test_subjects = montpellier_subjects[:int(len(montpellier_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = montpellier_subjects[int(len(montpellier_subjects) * fraction_test):]

elif args.data_split == 'rennes':
    renn_subjects = [sub for sub in all_centers_subjects if sub.startswith('rennes')]

    random.shuffle(renn_subjects)

    # takes 20% of the rennes dataset as the held out test-set
    test_subjects = renn_subjects[:int(len(renn_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = renn_subjects[int(len(renn_subjects) * fraction_test):]

elif args.data_split == 'nih':
    nih_subjects = [sub for sub in all_centers_subjects if sub.startswith('nih')]

    random.shuffle(nih_subjects)

    # takes 20% of the rennes dataset as the held out test-set
    test_subjects = nih_subjects[:int(len(nih_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = nih_subjects[int(len(nih_subjects) * fraction_test):]

elif args.data_split == 'ucsf':
    uscf_subjects = [sub for sub in all_centers_subjects if sub.startswith('ucsf')]

    random.shuffle(uscf_subjects)

    # takes 20% of the rennes dataset as the held out test-set
    test_subjects = uscf_subjects[:int(len(uscf_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = uscf_subjects[int(len(uscf_subjects) * fraction_test):]

# Maybe such dataset pairs could be used for pre-training. 
# TODO: Think about it!
elif args.data_split == 'bwh_and_renn':

    bwh_subjects = [sub for sub in all_centers_subjects if sub.startswith('bwh')]
    renn_subjects = [sub for sub in all_centers_subjects if sub.startswith('rennes')]

    random.shuffle(bwh_subjects); random.shuffle(renn_subjects)

    training_subjects = bwh_subjects
    
    # takes 20% of the rennes dataset as the held out test-set
    test_subjects = renn_subjects[:int(len(renn_subjects) * fraction_test)]
    print('Held-out Subjects: ', test_subjects)

elif args.data_split == 'renn_and_bwh':

    bwh_subjects = [sub for sub in all_centers_subjects if sub.startswith('bwh')]
    renn_subjects = [sub for sub in all_centers_subjects if sub.startswith('rennes')]

    random.shuffle(bwh_subjects); random.shuffle(renn_subjects)

    training_subjects = renn_subjects

    # takes 20% of the bwh dataset as the held out test-set
    test_subjects = bwh_subjects[:int(len(bwh_subjects) * fraction_test)]
    print('Held-out Subjects: ', test_subjects)

elif args.data_split == 'mix':
    all_centers_subjects = os.listdir(args.data_root)[1:]   # to exclude .DS_Store file
    
    all_centers_subjects = [sub for sub in all_centers_subjects if not sub.startswith('repro')]

    random.shuffle(all_centers_subjects)

    test_subjects = all_centers_subjects[:int(len(all_centers_subjects) * fraction_test)]
    # print('Held-out Subjects: ', test_subjects)

    # The rest of the subjects will be used for the train and validation phases
    training_subjects = all_centers_subjects[int(len(all_centers_subjects) * fraction_test):]


# keys to be defined in the dataset_0.json
params = {}
params["description"] = "CL for MS"
params["labels"] = {
    "0": "background",
    "1": "ms-lesion"
    }
params["seed_used"] = seed
params["modality"] = {
    "0": "MRI"
    }
params["name"] = f"continual-learning-ms data"
params["numTest"] = len(test_subjects)
params["numTraining"] = len(training_subjects)
params["reference"] = "XX"
params["tensorImageSize"] = "3D"

train_subjects_dict = {"training": training_subjects,} 
test_subjects_dict =  {"test": test_subjects}

# run loop for training and validation subjects
for name, subs_list in train_subjects_dict.items():

    temp_list = []
    for subject_no, subject in enumerate(tqdm(subs_list, desc='Loading Volumes')):
        temp_data = {}        
        
        if subject.startswith('milan'):
            # Turns out it has some corrupted data. check by loading with nibabel before creating datalist
            # Read-in input volumes
            t2 = os.path.join(args.data_root, subject, 'brain', 't2', 't2_mni.nii.gz')
            # Read-in GT volumes (using the consensus GT for now)
            gtc = os.path.join(args.data_root, subject, 'brain', 't2', 't2_lesion_manual_mni.nii.gz')

            try:
                nib.load(t2).get_fdata()
                # store in a temp dictionary
                temp_data["image"] = t2 #.replace(args.data_root+"/", '') # .strip(root)            
                temp_data["label"] = gtc #.replace(args.data_root+"/", '')       # .strip(root)
            except Exception as e:
                print(f"Subject {subject}'s data is corrupted, skipping subject!")
                continue

        
        elif subject.startswith('karo'):
            image = os.path.join(args.data_root, subject, 'brain', 't2', 't2_mni.nii.gz')
            if os.path.exists(image):
                # if T2 image exists (only for some subjects), then use that image-label pair
                temp_data["image"] = image
                temp_data["label"] = os.path.join(args.data_root, subject, 'brain', 't2', 't2_lesion_manual_mni.nii.gz')
            else:
                # if not, then use the flair image-label pair (true for some subjects)
                temp_data["image"] = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_mni.nii.gz')
                temp_data["label"] = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_lesion_manual_mni.nii.gz')

        else:
            # Read-in input volumes
            ses01_flair = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_mni.nii.gz')
            # Read-in GT volumes (using the consensus GT for now)
            gtc = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_lesion_manual_mni.nii.gz')
            
            # store in a temp dictionary
            temp_data["image"] = ses01_flair #.replace(args.data_root+"/", '') # .strip(root)            
            temp_data["label"] = gtc #.replace(args.data_root+"/", '')       # .strip(root)

        temp_list.append(temp_data)
    
    params[name] = temp_list


# run separte loop for testing
for name, subs_list in test_subjects_dict.items():
    temp_list = []
    for subject_no, subject in enumerate(tqdm(subs_list, desc='Loading Volumes')):
    
        temp_data = {}

        if subject.startswith('milan'):
            # Read-in input volumes
            t2 = os.path.join(args.data_root, subject, 'brain', 't2', 't2_mni.nii.gz')
            # Read-in GT volumes (using the consensus GT for now)
            gtc = os.path.join(args.data_root, subject, 'brain', 't2', 't2_lesion_manual_mni.nii.gz')
            
            # store in a temp dictionary
            temp_data["image"] = t2 #.replace(args.data_root+"/", '') # .strip(root)            
            temp_data["label"] = gtc #.replace(args.data_root+"/", '')       # .strip(root)
        
        elif subject.startswith('karo'):
            image = os.path.join(args.data_root, subject, 'brain', 't2', 't2_mni.nii.gz')
            if os.path.exists(image):
                # if T2 image exists (only for some subjects), then use that image-label pair
                temp_data["image"] = image
                temp_data["label"] = os.path.join(args.data_root, subject, 'brain', 't2', 't2_lesion_manual_mni.nii.gz')
            else:
                # if not, then use the flair image-label pair (true for some subjects)
                temp_data["image"] = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_mni.nii.gz')
                temp_data["label"] = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_lesion_manual_mni.nii.gz')

        else:
            # Read-in input volumes
            ses01_flair = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_mni.nii.gz')
            # Read-in GT volumes (using the consensus GT for now)
            gtc = os.path.join(args.data_root, subject, 'brain', 'flair', 'flair_lesion_manual_mni.nii.gz')

            # store in a temp dictionary
            temp_data["image"] = ses01_flair #.replace(args.data_root+"/", '') # .strip(root)
            temp_data["label"] = gtc #.replace(args.data_root+"/", '')       # .strip(root)

        temp_list.append(temp_data)
    
    params[name] = temp_list

final_json = json.dumps(params, indent=4, sort_keys=True)
jsonFile = open(os.path.join(save_path, f"dataset_{args.data_split}.json"), "w")
jsonFile.write(final_json)
jsonFile.close()





