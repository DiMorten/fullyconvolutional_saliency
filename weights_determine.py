import numpy as np
from icecream import ic
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import joblib

#train_samples = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'dance-jump', 'drift-turn', 'elephant', 'flamingo',
#    'hike', 'hockey', 'horsejump-low', 'kite-walk', 'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike', 'paragliding',
#    'rhino', 'scooter-gray', 'soccerball', 'stroller', 'surf', 'swing', 'tennis', 'train']

train_samples = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'dance-jump', 'dog-agility', 'drift-turn', 'elephant', 'flamingo',
    'hike', 'hockey', 'horsejump-low', 'kite-walk', 'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike', 'paragliding',
    'rhino', 'rollerblade', 'scooter-gray', 'soccerball', 'stroller', 'surf', 'swing', 'tennis', 'train']
label_path = Path('labels')
data_path = Path('data')
dims = (20, 128, 128)
class_n = 2
labels = np.zeros((len(train_samples), *dims))

data = np.zeros((len(train_samples), *dims, 3))

for idx, sample_name in enumerate(train_samples):
    sample_label_path = label_path / (sample_name + '.npy')
    sample_data_path = data_path / (sample_name + '.npy')
    
    labels[idx] = np.load(sample_label_path).astype(np.uint8)[..., 0]/255.
    data[idx] = np.load(sample_data_path)

ic(labels.shape)
ic(np.unique(labels, return_counts=True))
unique, count = np.unique(labels, return_counts=True)
class_weights = compute_class_weight(
           'balanced',
            unique, 
            labels.flatten())
ic(class_weights) # class_weights: array([0.54569158, 5.97146725])
# array([0.53869264, 6.96117691])

# normalize
scaler = StandardScaler()
data_shape = data.shape
data = np.reshape(data, (-1, data_shape[-1]))
scaler.fit(data)
data = scaler.transform(data)
data = np.reshape(data, data_shape)

ic(data.shape)
ic(np.average(data), np.std(data))
joblib.dump(scaler, 'scaler.save')