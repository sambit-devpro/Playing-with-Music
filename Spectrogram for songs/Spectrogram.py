import librosa, librosa.display
import matplotlib.pyplot as plt

#DheereDheere by Honey Singh ( You can mount your drive and choose any song you want.)
file="/content/drive/My Drive/songs.mp3"

#waveform
signal, sr =librosa.load(file, sr=22050)
librosa.display.waveplot(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

#Fourier transform
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency =  np.linspace(0,sr, len(magnitude))
left_frequency= frequency[:int(len(frequency)/2)]
left_magnitude= magnitude[:int(len(frequency)/2)]
plt.plot(left_frequency, left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

#Spectrogram

n_fft=2048
hop_length = 512
stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)


Spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(Spectrogram)
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("time")
plt.ylabel("Freq")
plt.colorbar()
plt.show()

#MFCCs

MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length= hop_length, n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()
