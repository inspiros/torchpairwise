"""Compare already aligned, same-sample-rate audio without loading dependencies."""
import torch
import torchpairwise as tp


def main():
    generator = torch.Generator().manual_seed(123)
    # N recordings, C channels, T time samples; preprocessing is caller-owned.
    reference = torch.randn(3, 1, 4096, generator=generator)
    estimate = .9 * reference + .05 * torch.randn(reference.shape, generator=generator)
    print('Paired SI-SDR dB:', tp.paired_si_sdr_scores(estimate, reference))
    print('Paired signal SNR dB:', tp.paired_signal_snr_scores(estimate, reference))
    print('All-pairs log spectral:', tp.log_spectral_distances(estimate, reference,
                                                            n_fft=256, hop_length=64))
    print('Reference-normalized convergence:',
          tp.paired_spectral_convergence_distances(estimate, reference, n_fft=256))
    print('Multi-resolution discrepancy:',
          tp.paired_multiresolution_stft_distances(estimate, reference,
                                                  resolutions=((256, 64, 256), (512, 128, 512))))
    print('Condensed symmetric self comparisons:',
          tp.condensed_log_spectral_distances(reference, n_fft=256))


if __name__ == '__main__':
    main()
