Extended Codec
===========

To run the encoder or decoder, run these commands from within the projects root directory:

```
$ python bin/encode.py
$ python bin/decode.py
```

The scripts control the encoding and decoding process. Modify the scripts to process different WAV or PAC files.

If you encounter problems, please contact aferrick@stanford.edu or gcarotti@stanford.edu

## Project Structure

- Final paper is in `aferrick_gcarotti_final_paper.pdf`
- Listening test data is in `listening_test_results.pdf`
- Encoded PAC files are in `/pacfiles`
- Encodded and decoded samples are in `/samples`
- The source code contained in `/source`
	- `/bin` - Scripts to start the codec
	- `/codec` - The Python package containing the extended codec
	- `/provided` - Provided code, against which unit tests were written
	- `/tests` - Location of unit tests


