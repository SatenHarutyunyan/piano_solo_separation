import scipy.io.wavfile as wavfile
import numpy as np
from preprocessing import Preprocessing

base_dir = "D:\\User\\Desktop\\Saten\\ideal0"
#
# c2_piano = base_dir + "\\C2_label.wav"
# r_music = base_dir + "\\R_input.wav"
# r_piano = base_dir + "\\R_label.wav"
pr = Preprocessing()
w_ratio = pr.window_overlap_ratio
music = "D:\\User\\Desktop\\capstone\\data\\symph_tracks\\0.wav"
piano = "D:\\User\\Desktop\\capstone\\data\\piano_tracks\\0.wav"
m = wavfile.read(music)[1][:30*16000]
p = wavfile.read(piano)[1][:30*16000]
mix_path = base_dir + "\\mix.wav"
piano_path = base_dir + "\\piano.wav"
sym_path = base_dir + "\\sym.wav"
wavfile.write(base_dir + "\\sym.wav", 16000, m)
wavfile.write(base_dir + "\\piano.wav", 16000, p)
wavfile.write(base_dir + "\\mix.wav", 16000, m+p)

def input_to_stft(x_i):
    j = 0
    fts = []
    while j + pr.frame_len <= pr.input_sec * pr.sample_rate:  # j is samples /2 + 1 , dont do the all window, later
        window = x_i[j: j + pr.frame_len] * pr.vorbis_window(pr.frame_len)
        ft = np.fft.rfft(window)
        fts.append(ft)
        j += pr.frame_len - pr.frame_overlap_len
    return np.array(fts)


def get_ideal_fft(m_track, p_track):
    m = pr.track_to_input(m_track)
    p = pr.track_to_input(p_track)
    # min_len = min(len(m), len(p))
    assert len(m) == len(p)
    new_p = []
    for i in range(len(m)):
        ft_m = input_to_stft(m[i])
        ft_p = input_to_stft(p[i])
        rm = np.abs(ft_p) / (np.abs(ft_m) + 1e-7)
        rm = np.clip(rm, 0, 1)
        new_p_ft = ft_m * rm
        new_p.append(new_p_ft)
    return new_p
def get_ideal_fft_rm2(m_track, p_track):
    m = pr.track_to_input(m_track)
    p = pr.track_to_input(p_track)
    # min_len = min(len(m), len(p))
    assert len(m) == len(p)
    new_p = []
    for i in range(len(m)):
        ft_m = input_to_stft(m[i])
        ft_p = input_to_stft(p[i])
        rm = np.sqrt(np.abs(ft_p)**2 / (np.abs(ft_m)**2 + np.abs(ft_p)**2))
        new_p_ft = ft_m * rm
        new_p.append(new_p_ft)
    return new_p
def get_ideal_fft_rm3(m_track, p_track):
    m = pr.track_to_input(m_track)
    p = pr.track_to_input(p_track)
    # min_len = min(len(m), len(p))
    assert len(m) == len(p)
    new_p = []
    for i in range(len(m)):
        ft_m = input_to_stft(m[i])
        ft_p = input_to_stft(p[i])
        rm = np.abs(ft_p)**2 / (np.abs(ft_m)**2 + np.abs(ft_p)**2)
        new_p_ft = ft_m * rm
        new_p.append(new_p_ft)
    return new_p

def produce_time_outputs(m_track, p_track, which_rm):
    if which_rm ==1:
        ft = get_ideal_fft(m_track, p_track)
    elif which_rm == 2:
        ft = get_ideal_fft_rm2(m_track, p_track)
    elif which_rm ==3:
        ft = get_ideal_fft_rm3(m_track, p_track)
    return pr.produce_time_outputs(ft)


def create_ideal_wav(mix_path, label_path,file_name,which_rm):
    time_outputs = produce_time_outputs(mix_path, label_path, which_rm)
    track = pr.time_outputs_into_track(time_outputs,len(p))
    wavfile.write(base_dir + "\\" + file_name + ".wav", 16000, track.astype("int16"))
    return track

def losss(path1, path2):
    tr1 = wavfile.read(path1)[1]
    tr2 = wavfile.read(path2)[1]
    return np.sqrt(np.sum((tr1-tr2)**2))

create_ideal_wav(mix_path, piano_path,"piano_output1",2)
create_ideal_wav(mix_path, sym_path,"sym_output1",2)
print(losss(mix_path, base_dir + "\\piano_output1.wav"))
print(losss(piano_path, base_dir + "\\sym_output1.wav"))
create_ideal_wav(mix_path, piano_path,"piano_output2",2)
create_ideal_wav(mix_path, sym_path,"sym_output2",2)
print(losss(mix_path, base_dir + "\\sym_output2.wav"))
print(losss(piano_path, base_dir + "\\piano_output2.wav"))
create_ideal_wav(mix_path, piano_path,"piano_output3",3)
create_ideal_wav(mix_path, sym_path,"sym_output3",3)
print(losss(mix_path, base_dir + "\\sym_output3.wav"))
print(losss(piano_path, base_dir + "\\piano_output3.wav"))
"""
signal = []
for i in range(10):
    temp = a[i] * vorbis_sq(pr.input_len * pr.sample_rate)
    if i == 0:
        signal.extend(temp)
    else:  # Ttake the overlaping portion of the last frame ad the overlap part of the new one
        signal[int((pr.input_len - pr.input_overlap_len) * pr.input_len *pr.sample_rate* i):int(pr.sample_rate*(pr.input_len - pr.input_overlap_len) * pr.input_len * i)+ pr.input_overlap_len] \
            += temp[:int(pr.sample_rate*pr.input_overlap_len)]
        signal.extend(temp[-int(pr.sample_rate*pr.input_overlap_len):])"""
"""
def get_ideal_fft(m_track,p_track):
    m = pr.track_to_input(m_track)
    p = pr.track_to_input(p_track)
    # min_len = min(len(m), len(p))
    assert len(m) == len(p)
    ideal = []
    for i in range(len(m)):
        m_e_phase = input_to_windows_ampl_phase(m[i])[1]
        p_ampl = input_to_windows_ampl_phase(p[i])[0]
        ideal.append(p_ampl * m_e_phase)
    return np.array(ideal)# has to be [min_len, frame_count, frame_len/2 +1]

"""
"""
def input_to_windows_ampl_phase(x_i):
    j = 0
    abs_ft=[]; phase_ft = []
    while j + pr.frame_len <= pr.input_sec * pr.sample_rate:  # j is samples /2 + 1 , dont do the all window, later
        window = x_i[j: j + pr.frame_len] * pr.vorbis_window(pr.frame_len)
        ft = np.fft.rfft(window)
        e_phase = ft / (np.abs(ft) + 1e-7)
        abs_ft.append(np.abs(ft))
        phase_ft.append(e_phase) # has to have dimensions(frame_count, frame_len/2 +1)
        j += pr.frame_len - pr.frame_overlap_len
    return np.array([abs_ft,phase_ft])

"""