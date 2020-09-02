import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000
seconds = 5
dosya_adi = input('ses dosya adi yazar misniz?')
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
print('konus')
sd.wait()
write(dosya_adi+'.wav', fs, myrecording)
