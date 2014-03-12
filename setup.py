from distutils.core import setup

setup(
    name='Perpeptual Audio Compressor',
    version='1.0',
    author='AJ Ferrick, Gabriele Carotti-Sha',
    author_email='ajf@ccrma.stanford.edu, gcarotti@stanford.edu',
    requires=['codec', 'provided.bitpack'],
    description='A perceptual audio compressor based on lessons from MUSIC422 at CCRMA.'
)