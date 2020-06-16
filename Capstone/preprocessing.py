import scipy.io.wavfile as wavfile
import numpy as np
import glob

class Preprocessing:
    def __init__(self):
        self.sample_rate = 16000  # smaples per second
        self.input_sec = 5  # second duration of X_i
        self.input_overlap_len = 3  # second , overlaping of X1 and X2
        self.db_list = [0, 1, 2, 4, 8, 16]
        self.win_duration = 0.03  # 30ms
        self.window_overlap_ratio = 0.5
        self.frame_overlap_len = int(self.win_duration * self.window_overlap_ratio * self.sample_rate)
        self.frame_len = int(self.win_duration * self.sample_rate)
        self.frame_count = int((self.input_sec - self.win_duration * self.window_overlap_ratio)/(self.win_duration *(1-self.window_overlap_ratio)))

        self.M_paths = glob.glob("/hdd/students/Saten/data/sym_tracks/*.wav")
        self.P_paths = glob.glob("/hdd/students/Saten/data/piano_tracks/**/*.wav", recursive = True) #foraua server
        # self.P_paths = glob.glob("/home/student/Saten/piano_tracks/**/*.wav", recursive = True) #for eph
        # self.M_paths = glob.glob("/home/student/Saten/sym_tracks/*.wav")
        print("len P: ",len(self.P_paths)," len M: ",len(self.M_paths))

    def track_to_input(self, path):
        array = []
        wav = wavfile.read(path)[1]
        j = 0
        while j + self.input_sec <= len(wav) / self.sample_rate:  # j second
            array.append(wav[j * self.sample_rate: (j + self.input_sec) * self.sample_rate])
            j = j + self.input_sec - self.input_overlap_len
        return array



    def vorbis_window(self, N):
        return np.sin((np.pi / 2) * (np.sin(np.pi * np.arange(N) / N)) ** 2)

    def windows_and_fft(self, x_i): #for an input
        j = 0
        x_i_new = []
        while j + self.frame_len <= self.input_sec * self.sample_rate:  # j is samples /2 + 1 , dont do the all window, later
            window = x_i[j: j + self.frame_len] * self.vorbis_window(self.frame_len)
            x_i_new.append(np.abs(np.fft.rfft(window)))  # has to have dimensions(frame_count, frame_len/2 +1)
            j += self.frame_len - self.frame_overlap_len
        return x_i_new

    def E(self, input):
        return np.sum(np.square(input))

    def preprocess(self):
        index = 0
        min_len = min(len(self.P_paths), len(self.M_paths))
        while(index<100001): #for j in range(10):
            np.random.shuffle(self.P_paths)
            for k in range(min_len):
                M = self.track_to_input(self.M_paths[k])  # arrays of inputs
                P = self.track_to_input(self.P_paths[k])
                for i in range(min(len(M), len(P))):#for every input
                    snr = self.E(M[i]) / self.E(P[i])
                    db = np.random.choice(self.db_list)
                    X_i = M[i] + P[i] * snr * 10 ** (-db / 10)
                    X_i = self.windows_and_fft(X_i)
                    P_i = self.windows_and_fft(P[i])
                    dir = "/train"
                    if index >= 80000:#j>8
                        dir = "/validation"
                    np.save("/home/student/Saten/data" + dir+ "/inputs/" + str(index), X_i)
                    np.save("/home/student/Saten/data"+ dir +"/labels/"+ str(index), P_i)
                    # np.save("/hdd/students/Saten/data" + dir + "/inputs/" + str(index), X_i)
                    # np.save("/home/students/Saten/data" + dir + "/labels/" + str(index), P_i)
                    index += 1
                    print("dir: ",dir,", index: ", index)


    def windows_and_fft_for_test(self, x_i): #for an input
        j = 0
        x_i_new = []
        while j + self.frame_len <= self.input_sec * self.sample_rate:  # j is samples /2 + 1 , dont do the all window, later
            window = x_i[j: j + self.frame_len] * self.vorbis_window(self.frame_len)
            x_i_new.append(np.fft.rfft(window))  # has to have dimensions(frame_count, frame_len/2 +1)
            j += self.frame_len - self.frame_overlap_len
        return x_i_new

    def windows_and_fft_for_test_unet(self, x_i): #for an input
        j = 0
        x_i_new = []
        while j + self.frame_len <= len(x_i):  # j is samples /2 + 1 , dont do the all window, later
            window = x_i[j: j + self.frame_len] * self.vorbis_window(self.frame_len)
            x_i_new.append(np.fft.rfft(window))  # has to have dimensions(frame_count, frame_len/2 +1)
            j += self.frame_len - self.frame_overlap_len
        return np.array(x_i_new)

    def track_to_input_unet(self, path, input_sec):
        array = []
        wav = wavfile.read(path)[1]
        j = 0
        while j + input_sec <= len(wav) / self.sample_rate:  # j second
            start = int(j * self.sample_rate)
            end= int((j + input_sec) * self.sample_rate)
            array.append(wav[start: end])
            j = j + input_sec - self.input_overlap_len
        return array

    def preprocess_test_data(self, track_path):
        raw_inputs = self.track_to_input(track_path)
        inputs = []
        for x_i in raw_inputs:
            inputs.append(self.windows_and_fft_for_test(x_i))
        return inputs

    def preprocess_test_data_unet(self, track_path):
        raw_inputs = self.track_to_input_unet(track_path, 4.815)
        inputs = []
        for x_i in raw_inputs:
            window = self.windows_and_fft_for_test_unet(x_i)
            inputs.append(window)
        return np.array(inputs)

    def produce_time_outputs(self, ft):
        time_outputs = []
        for output_num in range(len(ft)):
            output = np.zeros(self.input_sec * self.sample_rate)
            ft_output = ft[output_num]
            for frame_num in range(self.frame_count):
                frame = ft_output[frame_num]
                ift_window = np.fft.irfft(frame)
                ift_window *= self.vorbis_window(self.frame_len)
                start = int(frame_num * self.frame_len * (1 - self.window_overlap_ratio))
                end =int(start + self.frame_len)
                output[start: end] += ift_window
            time_outputs.append(output)
        return np.array(time_outputs)

    def produce_time_outputs_unet(self, ft, input_sec):
        time_outputs = []
        for output_num in range(len(ft)):
            output = np.zeros(int(input_sec * self.sample_rate))
            ft_output = ft[output_num]
            for frame_num in range(320):
                frame = ft_output[frame_num]
                ift_window = np.fft.irfft(frame)
                ift_window *= self.vorbis_window(self.frame_len)
                start = int(frame_num * self.frame_len * (1 - self.window_overlap_ratio))
                end = int(start + self.frame_len)
                output[start: end] += ift_window
            time_outputs.append(output)
        return np.array(time_outputs)


    def time_outputs_into_track(self, time_outputs, len_wav_samples):
        track = np.zeros(len_wav_samples)
        step_sample = (self.input_sec - self.input_overlap_len) * self.sample_rate
        for i in range(len(time_outputs)):
            temp = time_outputs[i] * self.vorbis_window(self.input_sec * self.sample_rate) ** 2
            track[int(i * step_sample):int(i * step_sample + self.input_sec * self.sample_rate)] += temp
        return track

    def time_outputs_into_track_unet(self, time_outputs, len_wav_samples, input_sec):
        track = np.zeros(len_wav_samples)
        print('track.shape: ',track.shape)
        input_len =  int(input_sec * self.sample_rate)
        print("input len: ", input_len)
        step_sample = int((input_sec - self.input_overlap_len) * self.sample_rate)
        print("step in samples: ", step_sample)
        for i in range(len(time_outputs)):
            temp = time_outputs[i] * self.vorbis_window(input_len) ** 2
            start=int(i * step_sample)
            end =int(i * step_sample + input_len)
            track[start:end] += temp
        return track

if __name__ == "__main__":
    print("a")
    prep = Preprocessing()
    prep.preprocess()
