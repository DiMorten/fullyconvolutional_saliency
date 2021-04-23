import numpy as np
from icecream import ic
from pathlib import Path
import cv2
data = np.load('data/dance-twirl.npy')[0].astype(np.uint8)
label = np.load('labels/dance-twirl.npy')[0].argmax(axis=-1)


label = label.astype(np.uint8)*255

ic(label.shape)
ic(np.unique(label, return_counts=True))

ic(data.shape)
ic(np.min(data), np.average(data), np.max(data))
ic(data.dtype)

observe_sample_path = Path('observe_samples')
observe_sample_path.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(observe_sample_path / 'data.png'), data)
cv2.imwrite(str(observe_sample_path / 'label.png'), label)