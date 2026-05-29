import os
import numpy as np

DATASET_TRAIN = os.path.join('dataset', 'train')
LABEL_FILE = 'label_classes.npy'
BACKUP_FILE = LABEL_FILE + '.bak'

print('Working dir:', os.getcwd())

# List train folders
if os.path.isdir(DATASET_TRAIN):
    folders = sorted([d for d in os.listdir(DATASET_TRAIN) if os.path.isdir(os.path.join(DATASET_TRAIN, d)) and not d.startswith('.')])
else:
    folders = []
print('Detected class folders under dataset/train:', folders)

# Load existing label file if any
if os.path.exists(LABEL_FILE):
    try:
        labels = np.load(LABEL_FILE, allow_pickle=True)
        print('\nExisting', LABEL_FILE, 'contents:', labels)
    except Exception as e:
        print('\nFailed to load', LABEL_FILE, '->', e)
        labels = None
else:
    print('\nNo existing', LABEL_FILE, 'found.')
    labels = None

# Prepare cleaned labels from folders
clean_labels = np.array(folders, dtype=object)
print('\nClean labels to save:', clean_labels)

# Backup existing label file if present
if os.path.exists(LABEL_FILE):
    try:
        if os.path.exists(BACKUP_FILE):
            print('Backup already exists:', BACKUP_FILE)
        else:
            os.rename(LABEL_FILE, BACKUP_FILE)
            print('Backed up existing', LABEL_FILE, '->', BACKUP_FILE)
    except Exception as e:
        print('Failed to backup existing label file:', e)

# Save cleaned labels
try:
    np.save(LABEL_FILE, clean_labels)
    print('Saved cleaned labels to', LABEL_FILE)
except Exception as e:
    print('Failed to save cleaned labels:', e)

print('\nDone.')
