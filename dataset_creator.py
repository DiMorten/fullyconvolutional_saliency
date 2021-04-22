import numpy as np
from pathlib import Path
from icecream import ic
import glob
import sys
import cv2
import pdb
#train_list = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'dance-jump', 'dog-agility', 'drift-turn', 'elephant', 'flamingo',
#    'hike', 'hockey', 'horsejump-low', 'kite-walk', 'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike', 'paragliding',
#    'rhino', 'rollerblade', 'scooter-gray', 'soccerball', 'stroller', 'surf', 'swing', 'tennis', 'train']

train_list = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'dance-jump', 'drift-turn', 'elephant', 'flamingo',
    'hike', 'hockey', 'horsejump-low', 'kite-walk', 'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike', 'paragliding',
    'rhino', 'scooter-gray', 'soccerball', 'stroller', 'surf', 'swing', 'tennis', 'train']

test_list = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane',
    'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black',
    'soapbox']


entire_list = train_list + test_list
ic(entire_list)
input_path = Path('dataset/DAVIS/JPEGImages/480p')
label_path = Path('dataset/DAVIS/Annotations/480p')

def jpg_to_npy_video(path, spatial_dim):
    frame_names = []
    frame_name_iterator = path.glob('*')
    for idx, frame_name in enumerate(frame_name_iterator):
        if idx % 2 == 0:
            frame_names.append(frame_name)
            #frame = cv2.imread(frame_name)
    max_len = 20
    if len(frame_names) > max_len:
        frame_names = frame_names[:max_len]
#    ic(frame_names)
#    ic(len(frame_names))
    assert len(frame_names) == max_len
    
    np_video = np.zeros((len(frame_names), *spatial_dim, 3))

    for idx, frame_name in enumerate(frame_names):
        frame = cv2.imread(str(frame_name))
#        ic(frame.dtype)
        frame = cv2.resize(frame, spatial_dim, interpolation = cv2.INTER_NEAREST)
        np_video[idx] = frame
    
    ic(np_video.dtype)     
    ic(np_video.shape)
    return np_video.astype(np.uint8)

spatial_dim = (128, 128)


ic(entire_list)
input_save_path = Path('data')
label_save_path = Path('labels')

input_save_path.mkdir(parents=True, exist_ok=True)
label_save_path.mkdir(parents=True, exist_ok=True)

for sample_name in entire_list:
    sample_input_path = input_path / sample_name
    sample_label_path = label_path / sample_name
    ic(sample_name)
    ic(sample_label_path)
    try:
        input_ = jpg_to_npy_video(sample_input_path, spatial_dim)
        label = jpg_to_npy_video(sample_label_path, spatial_dim)
        ic(np.min(input_), np.median(input_), np.average(input_), np.max(input_))
        ic(np.min(label), np.median(label), np.average(label), np.max(label))
#        ic(np.unique(label, return_counts=True))
        label[...,1] = 255 - label[...,0]  # take only one channel makes it grayscale
        label = label[...,:-1]


        ic(input_.shape)
        ic(label.shape)
#        ic(np.unique(label[...,0], return_counts=True))
#        ic(np.unique(label[...,1], return_counts=True))
#        ic(np.unique(label, return_counts=True))
#        pdb.set_trace()
        np.save(input_save_path / (sample_name+'.npy'), input_)
        np.save(label_save_path / (sample_name+'.npy'), label)

    except AssertionError:
        print("Sample {} too short. Not considered".format(sample_name))
        pdb.set_trace()


#    break

    


