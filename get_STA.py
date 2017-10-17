import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import decimate


# write function to compute STA

def windowed_avg_rows(dat,f,n_per_win=4):
    n_rows = np.shape(dat)[0]
    n_col = np.shape(dat)[1]
    
    st_points = np.arange(0,n_rows,n_per_win)
    n_rows_new = len(st_points)
    windowed_avg = np.zeros((n_rows_new, n_col))
    f_new = np.zeros(n_rows_new)
                          
    for i in range(n_rows_new):
        these_vals = dat[st_points[i]:st_points[i]+n_per_win,:]
        windowed_avg[i,:] = np.mean(these_vals,0)
        tmp = np.mean(f[st_points[i]:st_points[i]+n_per_win])
        f_new[i] = tmp
    
    return windowed_avg, f_new
                   
def windowed_avg_col(dat,t,ms_per_win = 20.0):
    t_per_win = ms_per_win/1000
    period = t[1]-t[0]
    val_per_win = int(round(t_per_win/period))
    n_row = np.shape(dat)[0]
    n_col = np.shape(dat)[1]

    st_points = np.arange(0,n_col-val_per_win+1,val_per_win)
    n_pts = len(st_points)
    windowed_avg = np.zeros((n_row, n_pts))
    
    for i in range(n_pts):
        these_vals = dat[:,st_points[i]:st_points[i]+val_per_win]
        windowed_avg[:,i] = np.mean(these_vals,1)
    t_new = t[st_points]
    return windowed_avg, t_new

def reduced_spectrogram(song,fs=44100):
    song_d = decimate(song,2,zero_phase = True)
    fs_new = fs/2
    nperseg = 128
    overlap = 64
    nfft = 128
    f, t, ss = signal.spectrogram(song_d,fs = fs_new, nperseg = 128,noverlap = 64,nfft = 128)
    # remove DC component
    f = f[1:]
    ss = ss[1:,:]
    ss_D = 10*np.log10(ss)
    
    ss_Df,f_new = windowed_avg_rows(ss_D,f)
    ss_Dft, t_new = windowed_avg_col(ss_Df,t)
    return f_new, t_new, ss_Dft

def full_spectrogram(song,fs=44100):
    song_d = decimate(song,2,zero_phase = True)
    fs_new = fs/2
    nperseg = 1024
    overlap = 512
    nfft = 1024
    f, t, ss = signal.spectrogram(song_d,fs = fs_new, nperseg = 128,noverlap = 64,nfft = 128)
    ss_D = 10*np.log10(ss)
    return f, t, ss_D

def visualize_spectrogram(f,t,dat,clim):
    plt.pcolormesh(f,t,dat)
    plt.xlabel('Time (s)')
    plt.ylabel('Freq (Hz)')
    try:
        plt.clim(clim)
    except:
        5
    plt.colorbar()
    plt.show()
    
def get_ds_rate(songs):
    n_songs = len(songs)
    ratios = np.zeros(n_songs)
    #print(type(ratios),ratios)
    for i in range(n_songs):
        this_song = songs[i]
        len_song = len(this_song)
        
        _, t, _ = reduced_spectrogram(this_song)
        del _
        len_spect = len(t)
        ratio =len_song/len_spect
        print(len_song,len_spect,ratio)
        ratios[i] = ratio
    my_ratio = np.mean(ratios,dtype=int)
    return my_ratio
        
        
    
    
def visualize_reduced_spectrogram(song,clim=[]):
    f, t, dat = reduced_spectrogram(song)
    visualize_spectrogram(t,f,dat,clim)
    
def visualize_full_spectrogram(song,clim=[]):
    f, t, dat = full_spectrogram(song)
    visualize_spectrogram(t,f,dat,clim)

def get_song_and_time_from_spike_times(spike_times,trials):
    5
    
def get_song_mask(trials,ratio):
    song_names = list(set(trials['stimulus']))
    n_song = len(song_names)
    all_time = trials['stimulus_end'].values[-1]
    
    song_mask = np.zeros(all_time,dtype = np.uint8)
    song_ramp = np.zeros(all_time)
    for i in range(n_song):
        this_song = song_names[i]
        these_trials = trials[trials['stimulus']==this_song]
        this_st = these_trials['time_samples'].values
        this_end = these_trials['stimulus_end'].values
        len_song = this_end[0]-this_st[0]
        n_rep = len(this_st)
        ramp = np.arange(0,len_song,len_song)/ratio
        ramp.astype(int)
        print(len_song)
        for j in range(n_rep):
            song_mask[this_st[j]:this_st[j]+len_song] = i+1
            song_ramp[this_st[j]:this_st[j]+len_song] = ramp
    return song_mask, song_ramp
        
    
def get_STA_from_points(spike_times,this_end,ev_len,song,prt = 1,window = 20):
    # spikes should be ms times of spikes
    # ev_end_times should be starts
    # ev_len
    # song
    # ev_time_ratio
    
    # get PSTH
    len_all = this_end[-1]
    this_indicator = np.zeros((len_all),dtype=int)
    stim_ramp = np.linspace(1,ev_len,ev_len,dtype=int)
    
    for i in range(len(this_end)):
        this_indicator[this_end[i]-ev_len:this_end[i]] = stim_ramp
        
    spike_times = spike_times[spike_times<=len_all]
    spike_points = this_indicator[spike_times]
    spike_trig_points = spike_points[spike_points>0] # PSTH
    
    # get spectrogram
    fs =  44100
    noverlap = 512
    nperseg = 1024
    
    f, t, ss = signal.spectrogram(song,fs =fs, nperseg = nperseg,noverlap = noverlap)
    ss_z = 10*np.log(ss)
    
    corresponding_ind = (np.fix((stim_ramp/noverlap)-1)+1)*2
    spect_trig_timing = (corresponding_ind[spike_trig_points]).astype(int)
    
    # get STA
    cum_sample = np.zeros((len(f),window))
    n_samp = 0
    #print(np.shape(ss_z),max(spect_trig_timing))
    for i in spect_trig_timing:
        if i<window-1 or i>np.shape(ss_z)[1]:
            continue
        n_samp+=1
        my_sample = ss_z[:,i-window:i]
        #print(n_samp,np.shape(my_sample))
        cum_sample += my_sample
    avg_sample = cum_sample/n_samp
    t = np.linspace(1,window,window,dtype = int)
    
    if prt:
        plt.figure(figsize=(15,10))
        plt.title('STA')
        plt.pcolormesh(t*100,f,avg_sample)

        plt.ylabel('Frequecy')
        plt.xlabel('Time (ms*100)')
    else:
        return avg_sample, f, t

def get_STA_mult_song(spike_times,songs,trials,window = 20):

    stim_labels = list(set(trials.stimulus))
    assert(len(stim_labels)==len(songs))
    
    avg_samples = []
    fs = []
    ts = []
    
    for song_ind in range(len(songs)):
        #print(song_ind)
        this_song_name = stim_labels[song_ind]
        song = songs[song_ind]
        
        these_trials = trials[trials['stimulus']==this_song_name]
        this_start = these_trials['time_samples'].values
        this_end = these_trials['stimulus_end'].values
        len_song = this_end[0] - this_start[0]
        
        avg_sample, f, t = get_STA_from_points(spike_times,this_end,len_song,song,0,window)
        
        avg_samples.append(avg_sample)
        fs.append(f)
        ts.append(t)
    return avg_samples, fs, ts

def plot_STA_mult_song(spike_times,songs,trials,window = 20,plt_mode = 1):
    
    avg_sample, f, t = get_STA_mult_song(spike_times,songs,trials,window)
    f = f[1]
    t = t[1]
    dat = np.mean(avg_sample,0)
    if plt_mode==0:
        return dat, f, t
    elif plt_mode==1:
        
        plt.figure(figsize=(15,10))
        plt.title('STA')
        plt.pcolormesh(t*100,f,dat)

        plt.ylabel('Frequecy')
        plt.xlabel('Time (ms*100)')
    elif plt_mode==2:
        n_plts = len(songs)
        plt.figure(figsize=(15,10))
        for i in range(n_plts):
            plt.subplot(1,n_plts,i+1)
            plt.pcolormesh(t*100,f,avg_sample[i])
            if i!=0:
                plt.xticks([])
                plt.yticks([])
            else:
                plt.ylabel('Frequecy')
                plt.xlabel('Time (ms*100)')
                
def plot_STA_mult_neuron(neurons,spikes,songs,trials,window=20):
    cluster_ind = neurons['cluster']
    
    STAs = []
    for i in range(5): #range(len(cluster_ind)):
        print(i)
        my_cluster = neurons['cluster'][i]
        my_spikes = spikes[spikes['cluster']==my_cluster]
        my_times = my_spikes['time_samples']
        
        my_avg_STA, f, t = plot_STA_mult_song(my_times,songs,trials,window,0)
        STAs.append(my_avg_STA)
        
    n_plts = i+1   
    plt.figure(figsize=(15,10))
    print(np.shape(STAs),np.shape(my_avg_STA))
    for i in range(n_plts):
        plt.subplot(1,n_plts,i+1)
        plt.pcolormesh(t*100,f,STAs[i])
        plt.title(cluster_ind[i])
        if i!=0:
            plt.xticks([])
            plt.yticks([])
        else:
            plt.ylabel('Frequecy')
            plt.xlabel('Time (ms*100)')
         

                