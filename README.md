# Triangle-BBH
Frequency-domain BBH TDI response template and basic analysis tools. TDI-2.0 response wrapper of 2 waveforms BBHx (https://github.com/mikekatz04/BBHx) and WF4PY (https://github.com/CosmoStatGW/WF4Py). 
The responses are consistent with the time-domain simulations of Triangle-Simulator. 

# Installation 
1. install triangle 

2. install modified WF4PY (if use WF4PY CPU waveform): 

cd WF4PY 

python setup.py install --user  

3. install BBHx (if use BBHx GPU waveform)  
   
see the instructions of BBHx.   

4. install MCMC tools to run the notebooks: 

pip install eryn corner  

