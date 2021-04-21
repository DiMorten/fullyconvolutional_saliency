import numpy as np
from pathlib import Path
from icecream import ic
import glob
import sys
import cv2
train_list = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'dance-jump', 'dog-agility', 'drift-turn', 'elephant', 'flamingo',
    'hike', 'hockey', 'horsejump-low', 'kite-walk', 'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike', 'paragliding',
    'rhino', 'rollerblade', 'scooter-gray', 'soccerball', 'stroller', 'surf', 'swing', 'tennis', 'train']

test_list = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane',
    'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black',
    'soapbox']

input_path = Path('data/DAVIS/JPEGImages/480p')
label_path = Path('data/DAVIS/Annotations/480p')

save_path = Path('data/')
def jpg_to_npy_video(path, spatial_dim):
    frame_names = []
    frame_name_iterator = path.glob('*')
    for idx, frame_name in enumerate(frame_name_iterator):
        if idx % 4 == 0:
            frame_names.append(frame_name)
            #frame = cv2.imread(frame_name)
    max_len = 10
    if len(frame_names) > max_len:
        frame_names = frame_names[:max_len]
    ic(frame_names)
    ic(len(frame_names))

    
    np_video = np.zeros((len(frame_names), *spatial_dim, 3))

    for idx, frame_name in enumerate(frame_names):
        frame = cv2.imread(str(frame_name))
        ic(frame.dtype)
        frame = cv2.resize(frame, spatial_dim, interpolation = cv2.INTER_AREA)
        np_video[idx] = frame
    
    ic(np_video.dtype)     
    ic(np_video.shape)
    ic(np.min(np_video), np.median(np_video), np.max(np_video))
    return np_video.astype(np.uint8)

spatial_dim = (256, 256)


sets = ['test', 'train']
set_lists = [test_list, train_list]
for set_, set_list in zip(sets, set_lists):

    ic(set_list)
    input_save_path = save_path / set_ / 'input_'
    label_save_path = save_path / set_ / 'label'

    input_save_path.mkdir(parents=True, exist_ok=True)
    label_save_path.mkdir(parents=True, exist_ok=True)

    for sample_name in set_list:
        sample_input_path = input_path / sample_name
        sample_label_path = label_path / sample_name
        ic(sample_name)
        ic(sample_label_path)
        input_ = jpg_to_npy_video(sample_input_path, spatial_dim)
        label = jpg_to_npy_video(sample_label_path, spatial_dim)
        
        ic(input_.shape)
        ic(label.shape)

        np.save(input_save_path / (sample_name+'.npy'), input_)
        np.save(label_save_path / (sample_name+'.npy'), label)

    #    break

        


