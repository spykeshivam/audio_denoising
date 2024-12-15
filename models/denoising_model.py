import torch

def load_model(model_path):
    # Load a pre-trained PyTorch model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

def denoise_audio(model, audio, sr):
    import librosa
    import numpy as np

    # Apply Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio, n_fft=1024, hop_length=512)
    magnitude, phase = np.abs(stft), np.angle(stft)

    # Use the model to enhance magnitude
    denoised_magnitude = model(torch.tensor(magnitude).unsqueeze(0)).squeeze(0).detach().numpy()

    # Reconstruct the signal
    denoised_stft = denoised_magnitude * np.exp(1j * phase)
    denoised_audio = librosa.istft(denoised_stft, hop_length=512)
    return denoised_audio
