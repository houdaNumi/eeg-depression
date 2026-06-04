import numpy as np
import mne
import glob
import pandas as pd
from scipy.stats import entropy





#decoupage en fenetres
window_size= 10*256











#on stock les features
feature_list = []
label_list = []






EC_files = glob.glob("../data/*EC*.edf")
print(EC_files)

for file in EC_files:

    if "MDD" in file:
        label = 1


    else:
        label = 0
    raw = mne.io.read_raw_edf(file, preload=True)
    raw_selected = raw.pick(['EEG Fp1-LE', 'EEG Fp2-LE'])
    raw_selected.notch_filter(50)
    data = raw_selected.get_data()
    nb_windows = data.shape[1] // window_size

    for i in range(nb_windows):
        window = data[:, i * window_size: (i + 1) * window_size]
        # print(window)
        print(f"Fenêtre {i}: début={i * window_size}, fin={(i + 1) * window_size}")
        max_val = np.max(window, axis=1)
        min_val = np.min(window, axis=1)
        mean_val = np.mean(window, axis=1)
        #variance
        var_val = np.var(window, axis=1)
        #range
        range_val = max_val - min_val
        entropy_fp1 = entropy(np.abs(window[0]))
        entropy_fp2 = entropy(np.abs(window[1]))
        entropy_val = np.array([entropy_fp1, entropy_fp2])
        print(max_val, min_val, mean_val, var_val, range_val, entropy_val)
        feature_list.append(np.concatenate([max_val, min_val, mean_val, var_val, range_val, entropy_val]))
        label_list.append(label)

    features_array = np.array(feature_list)
    print(features_array.shape)

    labels_array = np.array(label_list)
    print(labels_array.shape)

    print(pd.Series(labels_array).value_counts())

    np.save("../data/features.npy", features_array)
    np.save("../data/labels.npy", labels_array)