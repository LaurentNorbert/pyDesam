# pyDesam
High Resolution spectral analysis, for the estimation and pursuit of modulated sinusoids, for python 3.

TO RUN THE CODE
1. install soundfile python module (*pip install soundfile*)
2. run *python demo.py*
Note: you can change the *filename* variable in demo.py to process some other file (default is *glockenspielShort.wav*), in the *audio* folder.
The outputs are stored in the *output* folder.

Note : in demo.py, for technical reasons, the variable *L* must be odd.

ELB - laurent.benaroya@gmail.com - 01/2019

It is a translation of a part of the Desam toolbox, which is in Matlab and can be downloaded at http://www.tsi.telecom-paristech.fr/aao/2010/03/29/desam-toolbox/.

The python code corresponds in the Desam toolbox to "sinusoidalModels/shortTerm/highResolution_telecomParisTech".

Bibliography :
* "The DESAM Toolbox: spectral analysis of musical audio", Mathieu Lagrange, Roland Badeau, Bertrand David, Nancy Bertin, José Echeveste, Olivier Derrien, Sylvain Marchand et Laurent Daudet, 13th International Conference on Digital Audio Effects DAFx-10, Graz, Austriche, 6-10 septembre 2010.
* "Méthodes à haute résolution pour l'estimation et le suivi de sinusoïdes modulées." Application aux signaux de musique, Roland Badeau, Signal and Image processing. Télécom ParisTech, 2005. in French.

