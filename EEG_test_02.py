import numpy as np

import mne

########################Data Importing#########################
raw = mne.io.read_raw_brainvision("./raw/ph02a.vhdr", preload=True)
#If preload=False, data are not read until save
print(raw)
raw.plot() 


# set EOG channel
raw.set_channel_types({'HEOG1': 'eog'})

########################Re-referencing#########################
# set EEG average reference
raw.set_eeg_reference(ref_channels='average')

#A single electrode reference
#raw.set_eeg_reference(ref_channels=['Cz'])

# set EEG reference to the mean of multiple electrodes
#raw.set_eeg_reference(ref_channels=['M1', 'M2'])

######################Artifact Handling#########################

# show power line interference and remove it
raw.plot_psd(tmax=60., average=False)
raw.notch_filter(np.arange(60, 181, 60), fir_design='firwin')
#raw.plot_psd(tmax=60., average=False)

######################Resampling and filtering#########################
raw.resample(250, npad="auto") #set sampling frequency to 100Hz

#####################Define and read epochs##########################

#define parameters
#Define epochs parameters and handle conditions
event_id = dict(standard=4, deviant=11)
tmin = -0.2
tmax = 0.5
#Define the baseline period
baseline = (None, -100)  # means from the first instant to t = 0

#Define peak-to-peak rejection parameters for EOG
reject = dict(eog=250e-6)

#extract events
events = mne.find_events(raw, stim_channel='STI 014')

#exclude = raw.info['bads'] + ['Fc1', 'Cz']
exclude = raw.info['bads']
# pick EEG channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False,
                        eog=True, exclude=exclude)

# Compute epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, preload=False)

epochs.plot()


# compute evoked
evoked = epochs['deviant'].average()
evoked_std = epochs['standard'].average()

######################Get Evoke#########################
# remove physiological artifacts (eyeblinks, heartbeats) using SSP on baseline
evoked.add_proj(mne.compute_proj_evoked(evoked.copy().crop(tmax=0)))
evoked.apply_proj()

# fix stim artifact
mne.preprocessing.fix_stim_artifact(evoked)

# correct delays due to hardware (stim artifact is at 4 ms)
evoked.shift_time(-0.004)


# plot the result
evoked.plot(window_title="Evoked")


########### STD _& DEV (event=4, 11)
########### epochs.['standard'].average()
##### std와 dev 모두 표시한 다음 difference 구하기
## Code 질문
# 1. event_id - dict (standard=4, deviant=11)
#이후, evoked = deviant.average()는 있는데 standard.average()는 없음
#
#2. extract event에서 STI 014의 의미는?
#
#3. compute evoked에서도 마찬가지로 std의 부재
#
#4. concatenate_epoch에서 index와 range의 의미
#
#5. load_data 및 del, difference 확인


# We only use first 40 good epochs from each run. Since we first drop the bad
# epochs, the indices of the epochs are no longer same as in the original
# epochs collection. Investigation of the event timings reveals that first
# epoch from the second run corresponds to index 182.
epochs.drop_bad()
epochs_standard = mne.concatenate_epochs([epochs['standard'][range(40)],
                                          epochs['standard'][182:222]])
epochs_standard.load_data()  # Resampling to save memory.
epochs_deviant = epochs['deviant'].load_data()
del epochs, picks

###############################################################################
# The averages for each conditions are computed.
evoked_std = epochs_standard.average()
evoked_dev = epochs_deviant.average()
del epochs_standard, epochs_deviant

###########################################################
evoked_std.plot(window_title='Standard', gfp=True)
evoked_dev.plot(window_title='Deviant', gfp=True)


evoked_difference = combine_evoked([evoked_dev, -evoked_std], weights='equal')
evoked_difference.plot(window_title='Difference', gfp=True)



########### 그 다음 여기에서 채널 하나만 보려면?



######################Get Peak point######################
evoked.pick_channels(['Cz'])
peak = evoked.get_peak(ch_type='eeg', time_as_index=True)
print(peak)
evoked.plot(window_title="Evoked")



####################PCA & ICA######################

from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
X = epochs.get_data()


#the number of channels == 30
print("==============PCA==================")
pca = UnsupervisedSpatialFilter(PCA(30), average=False)
pca_data = pca.fit_transform(X)
ev = mne.EvokedArray(np.mean(pca_data, axis=0),
                     mne.create_info(30, epochs.info['sfreq'],
                                     ch_types='eeg'), tmin=tmin)
ev.plot(show=False, window_title="PCA")



print("==============ICA==================")
ica = UnsupervisedSpatialFilter(FastICA(30), average=False)
ica_data = ica.fit_transform(X)
ev1 = mne.EvokedArray(np.mean(ica_data, axis=0),
                      mne.create_info(30, epochs.info['sfreq'],
                                      ch_types='eeg'), tmin=tmin)
ev1.plot(show=False, window_title='ICA')

plt.show()


