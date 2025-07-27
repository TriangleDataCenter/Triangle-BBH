import numpy as np
try:
    import cupy as xp
    print("has cupy")
except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    print("no cupy")

from WF4Py import waveforms as WF
try: 
    from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
    print("has BBHx waveform")
except (ImportError, ModuleNotFoundError) as e:
    print("no BBHx waveform")

from Triangle.Constants import *

def eta_q(q):
    return q / (1. + q) ** 2

def q_eta(eta):
    if eta == 1. / 4.:
        return 1.
    else:
        return (1. - 2. * eta - np.sqrt(1. - 4. * eta)) / 2. / eta
    
def Mc_m1_q(m1, q):
    m2 = m1 * q
    return (m1 * m2) ** (3. / 5.) / (m1 + m2) ** (1. / 5.)

def m1_Mc_q(Mc, q):
    return Mc / (q ** 3 / (1. + q)) ** 0.2

def m1_m2_Mc_q(Mc, q):
    m1 = Mc / (q ** 3 / (1. + q)) ** 0.2
    m2 = m1 * q 
    return m1, m2


class WaveformGenerator():
    def __init__(self, mode='full'):
        """ 
            vectorized wrapper of the PhenomHM waveform 
            using the Fourier transformation convension of S. Marsat
            h_lm(f) = A_lm(f) exp[i Phi_lm(f)]
            t_lm(f) = d Phi_lm(f) / df / 2PI
            with Phi_lm(f) being -Psi_lm(f) in the paper
            
            mode = 'full': HM waveform
            mode = 'primary': D waveform
        """
        self.waveform = WF.IMRPhenomHM(mode=mode)
    
    def __call__(self, parameters, Nfreqs, fmin=1e-5, fmax=1e-1, freqs=None):
        """  
            parameters = {
                "name" : numpy array of shape (Nevents),
                ...
            }
            the parameters include:
            "chirp_mass" [MSUN], "mass_ratio" [1], "spin_1z" [1], "spin_2z" [1], 
            "coalescence_time" [DAY] (SSB), "coalescence_phase" [rad],
            "luminosity_distance" [MPC], "inclination" [rad], 
            "longitude" [rad], "latitude" [rad], "psi" [rad]
            11 parameters in total
            *** NOTE that coalescence time has to be converted to tc at the SSB origin.
            *** NOTE that coalescence phase parameter is ignored and set to 0
            freqs should be None or numpy array of shape (Nevents, Nfreqs)
            
            Calculate the mode-independent frequency grids of shape (Nevents, Nfreqs), 
            and mode-dependent (Nevents, Nfreqs) arrays of GW amplitudes, phases and t-f relationships, 
            stored in dictionaries with the keys being mode numbers.
            e.g. amps_out = {
                (l, m) : (Nevents, Nfreqs) array,
                ...
            }
        """
        # if isinstance(parameters["chirp_mass"], float):
        #     Nevents = 1
        # else:
        Nevents = parameters["chirp_mass"].shape[0]
        
        # conversion of parameters
        parameters_in = {}
        parameters_in["Mc"] = parameters["chirp_mass"]
        parameters_in["eta"] = eta_q(parameters["mass_ratio"])
        parameters_in["chi1z"] = parameters["spin_1z"]
        parameters_in["chi2z"] = parameters["spin_2z"]
        parameters_in["dL"] = parameters["luminosity_distance"] / 1e3 
        
        # calculate frequency grids of shape (Nfreqs, Nevents) (will be transposed later), independent of modes
        # NOTE: for later convenience fgrid must be the same for all events
        if type(freqs) == np.ndarray:
            fgrids = np.transpose(freqs) # (Nfreqs, Nevents)
        else:
            # fcutarr = self.waveform.fcut(**parameters_in)
            # fmaxarr = np.full(fcutarr.shape, min(np.min(fcutarr), fmax))
            # fminarr = np.full(fcutarr.shape, fmin)
            
            fminarr = np.full(Nevents, fmin)
            fmaxarr = np.full(Nevents, fmax)
            
            fgrids = np.geomspace(fminarr, fmaxarr, num=int(Nfreqs))
            
        
        # calculate amplitudes, phases by modes, each mode is originally a (Nfreqs, Nevents) array (will be transposed)
        amplitudes = self.waveform.Ampl(fgrids, **parameters_in) # dict
        phases = self.waveform.Phi(fgrids, **parameters_in) # dict
        # adjust keys to [(2,2), (2, 1), (3, 2), (3, 3), (4, 3), (4, 4)]
        # and shapes to (Nevents, Nfreqs)
        amps_out = {}
        phas_out = {}
        for k, v in amplitudes.items():
            amps_out[(int(k[0]), int(k[1]))] = v.T 
            phas_out[(int(k[0]), int(k[1]))] = phases[k].T 
        fgrids = fgrids.T 
            
        # set t_ref = 0, phi_ref = 0 
        # This step may result in slightly different phases for different resolutions. Therefore it is recommanded to keep consistency in terms of frequency resolution,  
        #  or otherwise, optimization of tc, phic would be required to get coincident results. 
        ind_ref = np.argmax(fgrids ** 2 * amps_out[(2, 2)], axis=1)
        f_ref = fgrids[np.arange(Nevents), ind_ref] # (Nevents)
        f_ref_1 = fgrids[np.arange(Nevents), ind_ref - 1]
        phase_22_ref = phas_out[(2, 2)][np.arange(Nevents), ind_ref] # (Nevents)
        phase_22_ref_1 = phas_out[(2, 2)][np.arange(Nevents), ind_ref - 1]
        dphase_22_ref = (phase_22_ref - phase_22_ref_1) / (f_ref - f_ref_1) # (Nevents)
        phase_correct1 = -(fgrids - f_ref[:, np.newaxis]) * dphase_22_ref[:, np.newaxis] # correct t_ref = 0, i.e. d Phi / df @ f_ref = 0
        # add the extra contribution of coalescence time 
        phase_correct1 += TWOPI * fgrids * parameters['coalescence_time'][:, np.newaxis] * DAY 
        for k, v in phas_out.items():
            phase_correct2 = -k[1] * 0.5 * phase_22_ref # correct phi_ref
            phas_out[k] = v + phase_correct1 + phase_correct2[:, np.newaxis]
        
        # calculate time grids by t-f relationship 
        tfs_out = {}
        dfgrids = fgrids[:, 1:] - fgrids[:, :-1]
        for k, v in phas_out.items():
            tfs_out[k] = np.zeros_like(v)
            dphase = v[:, 1:] - v[:, :-1]
            tfs_out[k][:, :-1] = dphase / dfgrids / TWOPI 
            tfs_out[k][:, -1] = tfs_out[k][:, -2] # fill the last frequency

        return fgrids, amps_out, phas_out, tfs_out # each item of shape (Nevents, Nfreqs)
    


class BBHxWaveformGenerator():
    def __init__(self, mode='full', use_gpu=False):
        """ 
            vectorized wrapper of the BBHx PhenomHM waveform 
            using the Fourier transformation convention of S. Marsat
            h_lm(f) = A_lm(f) exp[i Phi_lm(f)]
            t_lm(f) = d Phi_lm(f) / df / 2PI
            with Phi_lm(f) being -Psi_lm(f) in the paper
            
            mode = 'full': HM waveform
            mode = 'primary': D waveform
        """
        if mode == 'full':
            self.modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
            self.waveform = PhenomHMAmpPhase(use_gpu=use_gpu, run_phenomd=False)
        elif mode == 'primary':
            self.modes = [(2,2)]
            self.waveform = PhenomHMAmpPhase(use_gpu=use_gpu, run_phenomd=True)
        else:
            raise ValueError('mode can only be full or primary.')
        
        self.use_gpu = use_gpu
        if use_gpu:
            self.xp = xp 
        else: 
            self.xp = np 
    
    def __call__(self, parameters, freqs):
        """  
            parameters = {
                "name" : float or numpy array of shape (Nevents),
                ...
            }
            the parameters include:
            "chirp_mass" [MSUN], "mass_ratio" [1], "spin_1z" [1], "spin_2z" [1], 
            "coalescence_time" [DAY] (SSB), "coalescence_phase" [rad],
            "luminosity_distance" [MPC], "inclination" [rad], 
            "longitude" [rad], "latitude" [rad], "polarization" [rad]
            11 parameters in total
            *** NOTE that coalescence time has to be converted to tc at the SSB origin.
            *** NOTE that coalescence phase parameter is ignored and set to 0
            freqs is the array of frequencies where the waveforms will be calculated, shape is (Nfreqs) or (Nevents, Nfreqs). 
            
            Calculate the mode-independent frequency grids of shape (Nevents, Nfreqs), 
            and mode-dependent (Nevents, Nfreqs) arrays of GW amplitudes, phases and t-f relationships, 
            stored in dictionaries with the keys being mode numbers.
            e.g. amps_out = {
                (l, m) : (Nevents, Nfreqs) array,
                ...
            }
        """
        if isinstance(parameters["chirp_mass"], float):
            Nevents = 1
        else:
            Nevents = parameters["chirp_mass"].shape[0]
        Nfreqs = freqs.shape[-1]
        
        # conversion of parameters
        f_ref = np.zeros(Nevents) # f_ref = 0 means that f_ref is set to the frequency at coalescence 
        m1 = m1_Mc_q(parameters["chirp_mass"], parameters["mass_ratio"])
        m2 = m1 * parameters["mass_ratio"]
        a1 = parameters["spin_1z"]
        a2 = parameters["spin_2z"]
        t_ref = parameters['coalescence_time'] * DAY # default tc 
        # t_ref = np.zeros(Nevents) # alternative tc 
        phi_ref = np.zeros(Nevents) # phi_c is set to 0
        dist = parameters["luminosity_distance"] * MPC 
        
        # calculate fgrids and waveforms
        self.waveform(
            m1,
            m2,
            a1,
            a2,
            dist,
            phi_ref,
            f_ref,
            t_ref,
            length=Nfreqs,
            freqs=freqs,
            modes=self.modes,
        )

        fgrids = self.waveform.freqs_shaped  # shape (Nevents, Nfreqs)
        amps = self.waveform.amp  # shape (Nevents, Nmodes, Nfreqs)
        phase = self.waveform.phase  # shape (Nevents, Nmodes, Nfreqs) 
        tf = self.waveform.tf  # shape (Nevents, Nmodes, Nfreqs)
        # tc = self.xp.atleast_1d(parameters["coalescence_time"])[:, self.xp.newaxis, self.xp.newaxis] * DAY # (Nevent) -> (Nevent, 1, 1) alternative tc 
        # phase += TWOPI * fgrids[:, self.xp.newaxis, :] * tc # alternative tc 
        # tf += tc # alternative tc
     
        # save waveforms as dicts 
        amps_out = {}
        phas_out = {}
        tfs_out = {}
        for i, m in enumerate(self.modes):
            amps_out[m] = amps[:, i]
            phas_out[m] = phase[:, i]
            tfs_out[m] = tf[:, i]

        return fgrids, amps_out, phas_out, tfs_out # 
    

        