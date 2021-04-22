import numpy as np
from icecream import ic
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
train_samples = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'dance-jump', 'drift-turn', 'elephant', 'flamingo',
    'hike', 'hockey', 'horsejump-low', 'kite-walk', 'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike', 'paragliding',
    'rhino', 'scooter-gray', 'soccerball', 'stroller', 'surf', 'swing', 'tennis', 'train']


label_path = Path('labels')
spatial_dims = (128, 128)
class_n = 2
labels = np.zeros((len(train_samples), *spatial_dims))
for idx, sample_name in enumerate(train_samples):
    sample_path = label_path / (sample_name + '.npy')
    labels[idx] = np.load(sample_path).astype(np.uint8)[-1][..., 0]

ic(labels.shape)
ic(np.unique(labels, return_counts=True))
unique, count = np.unique(labels, return_counts=True)
class_weights = compute_class_weight(
           'balanced',
            unique, 
            labels.flatten())
ic(class_weights) # class_weights: array([0.54569158, 5.97146725])