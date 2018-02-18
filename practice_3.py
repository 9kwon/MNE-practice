import numpy as np

import mne

def for_diff(file_name):
    ########################Data Importing#########################
    raw = mne.io.read_raw_brainvision(file_name, preload=True)
    #If preload=False, data are not read until save
    print(raw)
    # raw.plot()


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
    # raw.plot_psd(tmax=60., average=False)
    raw.notch_filter(np.arange(60, 181, 60), fir_design='firwin')
    #raw.plot_psd(tmax=60., average=False)

    ######################Resampling and filtering#########################
    raw.resample(250, npad="auto") #set sampling frequency to 100Hz

    #####################Define and read epochs##########################

    #define parameters
    #Define epochs parameters and handle conditions
    event_id = dict(standard=11, deviant=1)
    tmin = -0.2
    tmax = 0.5
    #Define the baseline period
    baseline = (None, 0) # means from the first instant to t = 0
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

    evoked_difference = mne.combine_evoked([evoked, -evoked_std], weights='equal')
    evoked_difference.pick_channels(['Pz'])
    # evoked_difference.plot(window_title='Difference_mmn', gfp=True)

    return evoked_difference


file_name = ["./raw/ph01a.vhdr", "./raw/ph02a.vhdr",  "./raw/ph06b.vhdr"]
diff_list = []
for i in file_name:
    diff_list.append(for_diff(i))

evoked_difference = mne.combine_evoked(diff_list, weights='equal')
evoked_difference.plot(window_title='Difference_mmn', gfp=True)
