import mne
raw = mne.io.read_raw_edf("../data/MDD S1 EC.edf", preload=True)
raw2= mne.io.read_raw_edf("../data/H S1 EC.edf", preload=True)


print(raw.info)
print(raw.ch_names)
raw_selected = raw.pick(['EEG Fp1-LE', 'EEG Fp2-LE'])
#raw_selected2 = raw2.pick(['EEG Fp1-LE', 'EEG Fp2-LE'])
print(raw_selected.info)



raw_selected.plot(duration=10, title="AVANT filtre - MDD S1", block=False)
raw_selected.notch_filter(50)
raw_selected.plot(duration=10, title="APRES filtre - MDD S1", block=True)


#raw_selected2.plot(duration=10, title="Signal EEG brut - Sujet sain S1", block=True)
