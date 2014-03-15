# Perceptual Audio Codec

Authors: [AJ Ferrick](http://github.com/ajferrick), [Gabriele Carroti-Sha](http://github.com/weimin59)

This is the final project for [MUSIC 422](https://ccrma.stanford.edu/courses/422/2014/index.htm), taught at CCRMA at Stanford University, by Dr. Marina Bosi. This final project takes a baseline perceptual audio codec built up throughout the course and extends it with additional features.

Specifically, this codec exploits a model of hearing to discard frequency-spectrum data that would be otherwise inaudible. In spite of this distortion, the decoded audio achieves near-transparency on most audio signals, while maintaining the target compression bitrate.

Note that, this code is pretty far from perfect Python code. Time constraints (:clock10:), reuse of code from homework assignments (:recycle:), and multiple contributing students and provided teaching code (:camel:), stylistic purity and architectural integrity were not top priorities (:see_no_evil:). That said, some steps were taken to refactor code to provide true unit tests, remove god objects and side effects, and translate some of the C++ code into a more Pythonic style. Altogether, it reads like a bit of a pidgin (:bird:).


## Features

While the codec is described in more detail in the accompanying paper in the `docs/` folder, here is a brief summary of the features of this codec. The explanations here require some understanding of codec design - see the paper for a more thorough explanation.

#### Frequency Domain Storage

This is a _transform-based codec_: instead of compressing and storing sampled PCM data, this codec opts to take the frequency spectrum of small blocks of data and compress/store the Fourier series. In particular, the codec utilizes the Modfied Discrete Cosine Transform and Kaiser-Bessel Derived windows. The codec also utilized a block floating-point scheme to compress data.

#### Block Switching

A major impairment with transform codecs is the introduction of _pre-echo_, an auditory premonition that precedes a sharp attack. This usually sounds likes a faint attack of the instrument, before the true at	tack occurs. Briefly, this happens because the encoding block size, usually ~50ms, is too long to contain a sharp percussive hit; in the decoding phase, the energy of the percussive attack is spread across these 50ms — thus, hearing a pre-attack before the true (and therefore attenuated) attack.

This codec counters this impairment through _block switching_. Block switching allows the codec to gracefully transition to using short 4ms windows to capture musical attacks as they happen. The details of the implementation can be found in the accompanying paper. Briefly, it works by:

1. Performing a lookahead on the input audio file and detecting incoming musical attacks.
2. Determining the block size needed to both capture the attack and respect overlap-and-add constraints.
3. Altering parameters throughout the codec to adjust for variable block sizes.

#### Perceptual Sidechain

As mentioned above, this codec utlizes a model of hearing to utilizes its rate distortion in lossy compression. The goal of the perceptual sidechain is to identify spectral data that would otherwise be inaudible to due spectral masking phenomenon. This data is discarded by the codec, allowing a higher resolution description of perceived frequencies.

- __Tonal & Noise Masking:__ The sidechain constructs both tonal and noise masking curves for the given sample block. Tonal maskers are created from frequency spectrum peaks and noise maskers are created from the residual spectrum. These masking curves are combined with a threshold of hearing curve   to identify which frequency bands are inaudible in the sample block.

- __Masking Decimation:__ If more than one masking curve is centered in a critical hearing band, only the masking curve with the high peak is used.

- __Rate-Distortion Allocation:__ For the desired compression rate, the codec uses the masking curve to determine which "channels" — i.e. frequency bands — can be distorted. The allocation of bits into channels is implemented with a reverse water-filling algorithm.


## How to Use

Encoding and decoding occurs through the two scripts in the `bin/` folder. You must specify the input and output files.

```
$ python bin/encode.py input.wav coded.pac
$ python bin/decode.py coded.pac output.wav
```

Unit tests can be run either individually or through the [nose](https://nose.readthedocs.org/en/latest/) test suite.

```
$ python tests/test_window.py
$ nosetests
```

## Project Structure

- `bin`: Scripts to kick off encoding and decoding
- `codec`: Package used to perform the coding
	- `bitpack.py`: Helper module provided by Dr. Bosi and Dr. Goldberg to facilitate writing to the PAC file — the generic compressed binary files.
- `docs`: Documents written to accompany the final project
	- `paper.pdf`: Paper written to motivate the project, document the implementation, and provide project results.
	- `listening_tests.pdf`: Some summary data on listening tests conducted with the codec
- `tests`: Unit tests for various modules in the `codec` package. Some test against known cases, and some test against binaries provided by Dr. Bosi and Dr. Goldberg.
	- `provided`: The _[sic]_ provided files. They were provided as binaries in the original work; here, they are source files.


## Notes

#### Apologies

Again, apologies for the sub-par quality of the code. This ain't gonna win any beauty awards. :feelsgood:

#### Future Work 

Many improvements can be made to the current codec. The transient detection algorithm suffers in the face of constant high frequency content, yielding many false positives. This detection could be made more accurate by using an adaptive threshold — potentially one that accumulates and rises with high energy content, and over time dissipates back to its original lower threshold (e.g. a "leaky" integrating threshold). Temporal noise shaping algorithms would be useful in providing better control over quantization noise with respect to the masking threshold. This would result in better quality for speech signals as well as aid block switching in a finer representation of transient signals. Additional work could tighten and improve the bit allocation scheme, which currently under-allocates bits for some samples. Further improvements in compression can be achieved by employing a lossless entropy coding scheme, such as Huffman coding. 

Finally, the codec speed is not at all performant to the point of being able to handle streaming audio.

#### Thanks

Thanks to Dr. Marina Bosi for teaching this course, and to Tim O'Brien, TA extraordinaire, for providing advice on the project.