# Triangle-BBH
Frequency-domain BBH TDI response template and basic analysis tools, including TDI-2.0 response wrappers of 2 waveforms: (1) IMRPhenomD \& HM of BBHx (https://github.com/mikekatz04/BBHx, CPU & GPU) and (2) IMRPhenomD \& HM of WF4PY (https://github.com/CosmoStatGW/WF4Py, CPU only). 
The responses are consistent with the time-domain simulations of Triangle-Simulator. 

# Installation 
1. install triangle and activate triangle environment

2. install modified WF4PY (if use WF4PY CPU waveform): 

   cd WF4PY 

   python setup.py install --user  

3. install BBHx (if use BBHx GPU waveform)  
   
   see the instructions of BBHx.   

4. install MCMC tools to run the notebooks: 

   pip install eryn corner  

5. install Triangle_BBH 
   
   pip install -e . (or python setup.py install --user)

# Comparison with time-domain simulation 
![image](Figures/TD_vs_FD.jpg)