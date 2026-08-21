import numpy as np 
try:
    import cupy as xp
    # print("has cupy")
except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    # print("no cupy")

from scipy.interpolate import CubicSpline, interp1d

PI = 3.141592653589793
TWOPI = 6.283185307179586
DAY = 86400.0


def FrequencyDomainSNR(h, psd, df):
    """  
    h should be FD waveform 
    psd is an array of the same shape as h
    """
    return np.sqrt(4. * np.sum(np.abs(h) ** 2 / psd) * df)


def FrequencyDomainMismatch(h1, h2, psd):
    """  
    h1 and h2 should be FD waveforms 
    """
    rho1 = np.sqrt(np.sum(np.abs(h1) ** 2 / psd))
    rho2 = np.sqrt(np.sum(np.abs(h2) ** 2 / psd))
    inner12 = np.real(np.sum(h1 * np.conjugate(h2) / psd)) 
    return 1. - inner12 / rho1 / rho2

def FrequencyDomainInnerProduct(h1, h2, psd, df):
    return 4. * np.real(np.sum(df * h1 * np.conjugate(h2) / psd))


def FrequencyDomainCovarianceSNR(data_channels, inv_cov):
    """  
    Args:
        data_channels: numpy array of shape (3, Nf)
        inv_cov: numpy array of shape (Nf, 3, 3), with Cov_IJ = CSD_IJ / 4 / Df, CSD_IJ = 2<I J^*>/T
    Returns: 
        sqrt(d^\dagger C^-1 d), real scalar 
    """
    data_expanded = np.transpose(data_channels)[:,np.newaxis,:] # (Nf, 1, 3)
    data_expanded_transposed = np.transpose(data_expanded, (0, 2, 1)) # (Nf, 3, 1)
    tmp1 = np.matmul(np.conjugate(data_expanded), inv_cov) # (Nf, 1, 3)
    tmp2 = np.matmul(tmp1, data_expanded_transposed) # (Nf, 1, 1)
    return np.sqrt(np.real(np.sum(tmp2)))


def FrequencyDomainCovarianceInnerProduct(data_channels1, data_channels2, inv_cov):
    """  
    Args:
        data_channels1, 2: numpy arrays of shapes (3, Nf)
        inv_cov: numpy array of shape (Nf, 3, 3), with Cov_IJ = CSD_IJ / 4 / Df, CSD_IJ = 2<I J^*>/T
    Returns: 
        d_1^\dagger C^-1 d_2, complex scalar (not the conventional definition of inner product, which should be a real number)
    """
    data_expanded1 = np.transpose(data_channels1)[:,np.newaxis,:] # (Nf, 1, 3)
    data_expanded2 = np.transpose(data_channels2)[:,np.newaxis,:] # (Nf, 1, 3)
    data_expanded_transposed2 = np.transpose(data_expanded2, (0, 2, 1)) # (Nf, 3, 1)
    tmp1 = np.matmul(np.conjugate(data_expanded1), inv_cov) # (Nf, 1, 3)
    tmp2 = np.matmul(tmp1, data_expanded_transposed2) # (Nf, 1, 1)
    return np.sum(tmp2)

def ParamDict2ParamArr(param_dict):
    """ 
    convert parameter dict to parameter array where Mc and dL are in logscale, 
    inclination and latitude are converted to their cosine and sine values 
    """
    return [
        np.log10(param_dict['chirp_mass']),
        param_dict['mass_ratio'],
        param_dict['spin_1z'],
        param_dict['spin_2z'],
        param_dict['coalescence_time'],
        param_dict['coalescence_phase'],
        np.log10(param_dict['luminosity_distance']),
        np.cos(param_dict['inclination']),
        param_dict['longitude'],
        np.sin(param_dict['latitude']),
        param_dict['psi']
    ]

def ParamArr2ParamDict(params):
    p = dict()
    p['chirp_mass'] = np.power(10., params[0])
    p['mass_ratio'] = params[1]
    p['spin_1z'] = params[2]
    p['spin_2z'] = params[3]
    p['coalescence_time'] = params[4]
    p['coalescence_phase'] = params[5]
    p['luminosity_distance'] = np.power(10., params[6])
    p['inclination'] = np.arccos(params[7])
    p['longitude'] = params[8]
    p['latitude'] = np.arcsin(params[9])
    p['psi'] = params[10] 
    return p 

def ParamDict2ParamArrFref(param_dict):
    """ 
    convert parameter dict to parameter array where Mc and dL are in logscale, 
    inclination and latitude are converted to their cosine and sine values 
    """
    return [
        np.log10(param_dict['chirp_mass']),
        param_dict['mass_ratio'],
        param_dict['spin_1z'],
        param_dict['spin_2z'],
        param_dict['reference_time'],
        param_dict['reference_phase'],
        np.log10(param_dict['luminosity_distance']),
        np.cos(param_dict['inclination']),
        param_dict['longitude'],
        np.sin(param_dict['latitude']),
        param_dict['psi']
    ]

def ParamArr2ParamDictFref(params):
    p = dict()
    p['chirp_mass'] = np.power(10., params[0])
    p['mass_ratio'] = params[1]
    p['spin_1z'] = params[2]
    p['spin_2z'] = params[3]
    p['reference_time'] = params[4]
    p['reference_phase'] = params[5]
    p['luminosity_distance'] = np.power(10., params[6])
    p['inclination'] = np.arccos(params[7])
    p['longitude'] = params[8]
    p['latitude'] = np.arcsin(params[9])
    p['psi'] = params[10] 
    return p 



class Likelihood:
    # TODO: mode-by-mode heterodyne for the PhenomHM waveform 
    def __init__(self, response_generator, frequency, data, invserse_covariance_matrix, response_parameters, Fref_waveform=False, use_gpu=False, verbose=0):
        """ 
        Args: 
            response_generator: generate frequency-domain TDI responses for given parameters 
            frequency: frequencies of data, numpy array of shape (Nf,), may not be evenly spaced at high frequencies due to the mask 
            data: frequency-domain tdi data, numpy array of shape (3, Nf)
            invserse_covariance_matrix: numpy array of shape (Nf, 3, 3), with Cov_IJ = CSD_IJ / 4 / Df, CSD_IJ = 2<I J^*>/T
            the SNR of signal h is defined as \sum_f h^\dagger C^-1 h 
            response_parameters: parameters of the response_generator besides source parameters and frequencies.
        """
        if frequency.shape[-1] != data.shape[-1] or frequency.shape[-1] != invserse_covariance_matrix.shape[0]:
            raise ValueError("shapes of data mismatch.")
        
        self.response_generator = response_generator
        self.frequency = frequency
        self.data = data 
        self.invserse_covariance_matrix = invserse_covariance_matrix 
        self.response_kwargs = response_parameters

        self.use_gpu = use_gpu
        if use_gpu: 
            self.xp = xp 
        else: 
            self.xp = np 

        self.frequency = self.xp.array(self.frequency)
        self.data = self.xp.array(data)
        self.invserse_covariance_matrix = self.xp.array(self.invserse_covariance_matrix)
        
        if verbose > 0:
            print("number of freuqncies:", len(frequency))
            print("min and max frequencies:", self.xp.min(frequency), self.xp.max(frequency))
            print("response kwargs:", self.response_kwargs)
        
        self.het_prepare_flag = False 
        
        if Fref_waveform: 
            self.ParamDict2ParamArr = ParamDict2ParamArrFref
            self.ParamArr2ParamDict = ParamArr2ParamDictFref
        else: 
            self.ParamDict2ParamArr = ParamDict2ParamArr
            self.ParamArr2ParamDict = ParamArr2ParamDict
        
    def full_log_like(self, parameter_array): 
        """ 
        Args: 
            parameter_array: parameters given as an array.
            the order is: ['log_chirp_mass', 'mass_ratio', 'spin_1z', 'spin_2z', 'coalescence_time', 'coalescence_phase', 'log_luminosity_distance', 'cos_inclination', 'longitude', 'sin_latitude', 'psi']
        Returns: 
            loglike 
        """
        template = self.response_generator.Response(
            parameters=self.ParamArr2ParamDict(parameter_array),
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (3, Nf)
        residual = self.data - template # (3, Nf)
        
        if self.use_gpu:
            return (-0.5 * self.FrequencyDomainCovarianceSNR(data_channels=residual, inv_cov=self.invserse_covariance_matrix) ** 2).get() 
        else:
            return -0.5 * self.FrequencyDomainCovarianceSNR(data_channels=residual, inv_cov=self.invserse_covariance_matrix) ** 2 
        
    def full_log_like_vectorized(self, parameter_array): 
        """ 
        Args: 
            parameter_array: parameters given as an array of shape (Nparams, Nevents).
            the order is: ['log_chirp_mass', 'mass_ratio', 'spin_1z', 'spin_2z', 'coalescence_time', 'coalescence_phase', 'log_luminosity_distance', 'cos_inclination', 'longitude', 'sin_latitude', 'psi']
        Returns: 
            loglike of shape (Nevents)
        """
        template = self.response_generator.Response(
            parameters=self.ParamArr2ParamDict(parameter_array),
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (3, Nevents, Nf)
        residual = self.xp.transpose(self.data - self.xp.transpose(template, (1, 0, 2)), (0, 2, 1)) # (Nevents, 3, Nf) -> (Nevents, Nf, 3)
        residual_dagger = self.xp.conjugate(residual[:, :, self.xp.newaxis, :]) # (Nevents, Nf, 1, 3)
        residual = residual[:, :, :, self.xp.newaxis] # (Nevents, Nf, 3, 1)
        loglikes = self.xp.sum(self.xp.matmul(self.xp.matmul(residual_dagger, self.invserse_covariance_matrix), residual), axis=(1,2,3))
        
        if self.use_gpu: 
            return np.real(-0.5 * loglikes.get()) 
        else: 
            return np.real(-0.5 * loglikes)
    
    def prepare_het_log_like(self, base_parameters=None, base_waveform=None, num_het_frequency=128):
        """ 
        Args: 
            base_parameters: None or numpy array of shape (Nparams)
            base_waveform: None or numpy array of shape (3, Nf)
            num_het_frequency: number of sparse frequencies to calculate the waveform perturbation, N_het_f = Nb + 1 
            NOTE: we recommend only using base parameters rather than base waveform 
        """
        if base_waveform is None:
            self.h0 = self.response_generator.Response(
                parameters=self.ParamArr2ParamDict(base_parameters),
                freqs=self.frequency,
                **self.response_kwargs,
            ) # (3, Nf)
        else: 
            raise NotImplementedError("heterodyned likelihood with base waveform not implemented yet.")
        
        # create sparse grid of frequencies (1st try)
        FMIN, FMAX = self.xp.min(self.frequency) * 0.999999999999, self.xp.max(self.frequency) * 1.000000000001
        self.het_frequency = self.xp.logspace(self.xp.log10(FMIN), self.xp.log10(FMAX), num_het_frequency) # N_het_f
        
        # calculate base waveform at the sparse grid (1st try)
        if base_waveform is None:
            self.het_h0 = self.response_generator.Response(
                parameters=self.ParamArr2ParamDict(base_parameters),
                freqs=self.het_frequency,
                **self.response_kwargs,
            ) # (3, N_het_f)
        else: 
            raise NotImplementedError("heterodyned likelihood with base waveform not implemented yet.")
        
        # refine the sparse grid to ensure no zero waveforms 
        # valid_idx = self.xp.where(self.xp.abs(self.het_h0[0]) != 0.)[0]
        valid_idx = self.xp.where(self.xp.abs(self.het_h0[0]) > 1e-25)[0]
        tmpf = self.het_frequency[valid_idx]
        
        # create sparse grid of frequencies (final)
        self.het_frequency = self.xp.logspace(self.xp.log10(tmpf[0]), self.xp.log10(tmpf[-1]), num_het_frequency) # N_het_f
        
        # calculate base waveform at the sparse grid (final)
        if base_waveform is None: 
            self.het_h0 = self.response_generator.Response(
                parameters=self.ParamArr2ParamDict(base_parameters),
                freqs=self.het_frequency,
                **self.response_kwargs,
            ) # (3, N_het_f)
        else: 
            raise NotImplementedError("heterodyned likelihood with base waveform not implemented yet.")
        
        self.het_h0[self.xp.abs(self.het_h0)<1e-25] = 1e-25 
        
        # confine the frequency and data to be within the boundaries of sparse grid 
        inband_idx = self.xp.where((self.frequency >= self.het_frequency[0]) & (self.frequency <= self.het_frequency[-1]))[0]
        self.dense_frequency = self.frequency[inband_idx] # (Nf)
        self.dense_data = self.data[:, inband_idx] # (3, Nf)
        self.dense_h0 = self.h0[:, inband_idx] # (3, Nf)
        self.dense_invserse_covariance_matrix = self.invserse_covariance_matrix[inband_idx] # (Nf, 3, 3)
        
        # group the dense frequencies with the sparse frequency grid, return the left idx of each dense frequency, each bin is labeled by this 
        group_idx = self.xp.searchsorted(self.het_frequency, self.dense_frequency, "right") - 1 # (Nf)
        dense_frequency_offset = self.dense_frequency - self.het_frequency[group_idx] # (Nf)
        
        # start pre-calculating the coefficients of heterodyned likelihood 
        # 1) h_h terms
        B0_pre = self.xp.matmul(self.xp.transpose(self.xp.conjugate(self.dense_h0))[:, :, self.xp.newaxis], self.xp.transpose(self.dense_h0)[:, self.xp.newaxis, :]) * self.dense_invserse_covariance_matrix # (Nf, 3, 1) * (Nf, 1, 3) -> (Nf, 3, 3)
        B1_pre = B0_pre * dense_frequency_offset[:, self.xp.newaxis, self.xp.newaxis] # (Nf, 3, 3)
        # 2) d_h terms 
        A0_pre = self.xp.matmul(self.xp.transpose(self.xp.conjugate(self.dense_data))[:, :, self.xp.newaxis], self.xp.transpose(self.dense_h0)[:, self.xp.newaxis, :]) * self.dense_invserse_covariance_matrix # (Nf, 3, 1) * (Nf, 1, 3) -> (Nf, 3, 3)
        A1_pre = A0_pre * dense_frequency_offset[:, self.xp.newaxis, self.xp.newaxis] # (Nf, 3, 3)
        
        # sum all the coefficients in sparse grids  
        self.Nbin = num_het_frequency - 1 
        if self.response_kwargs.get("drop_T", False): 
            self.B0 = self.xp.zeros((self.Nbin, 2, 2), dtype=self.xp.complex128)
            self.B1 = self.xp.zeros((self.Nbin, 2, 2), dtype=self.xp.complex128)
            self.A0 = self.xp.zeros((self.Nbin, 2, 2), dtype=self.xp.complex128)
            self.A1 = self.xp.zeros((self.Nbin, 2, 2), dtype=self.xp.complex128)
        else:
            self.B0 = self.xp.zeros((self.Nbin, 3, 3), dtype=self.xp.complex128)
            self.B1 = self.xp.zeros((self.Nbin, 3, 3), dtype=self.xp.complex128)
            self.A0 = self.xp.zeros((self.Nbin, 3, 3), dtype=self.xp.complex128)
            self.A1 = self.xp.zeros((self.Nbin, 3, 3), dtype=self.xp.complex128)
        for ibin in self.xp.unique(group_idx): # loop over the left idx of sparse grids 
            inbin_idx = group_idx == ibin 
            self.B0[ibin] = self.xp.sum(B0_pre[inbin_idx], axis=0) # (3, 3)
            self.B1[ibin] = self.xp.sum(B1_pre[inbin_idx], axis=0)
            self.A0[ibin] = self.xp.sum(A0_pre[inbin_idx], axis=0)
            self.A1[ibin] = self.xp.sum(A1_pre[inbin_idx], axis=0)
            
        self.het_df = self.het_frequency[1:] - self.het_frequency[:-1] # (Nb)
        self.het_prepare_flag = True 
        
    def het_log_like(self, parameter_array):
        """ 
        Args: 
            parameter_array: parameters given as an array.
            the order is: ['log_chirp_mass', 'mass_ratio', 'spin_1z', 'spin_2z', 'coalescence_time', 'coalescence_phase', 'log_luminosity_distance', 'cos_inclination', 'longitude', 'sin_latitude', 'psi']
        Returns: 
            heterodyned loglike 
        """
        if not self.het_prepare_flag: 
            raise NotImplementedError("heterodyne not prepared, run preparation first.")
            
        # calculate sparse template 
        het_h = self.response_generator.Response(
            parameters=self.ParamArr2ParamDict(parameter_array),
            freqs=self.het_frequency,
            **self.response_kwargs,
        ) # (3, N_het_f)
        
        # calculate heterodyne 
        het_r = self.xp.transpose(het_h / self.het_h0) # (N_het_f, 3)
        het_r = self.xp.nan_to_num(het_r, 0.) # deal with the nan caused by divide 0 
        het_r0 = het_r[:-1] # (Nb, 3)
        het_r1 = (het_r[1:] - het_r0) / self.het_df[:, self.xp.newaxis] # (Nb, 3)
        
        # calculate likelihood 
        # 1) h_h term 
        LL1 = self.xp.sum(self.xp.matmul(self.xp.conjugate(het_r0)[:, :, self.xp.newaxis], het_r0[:, self.xp.newaxis, :]) * self.B0) # sum((Nb, 3, 3) * (Nb, 3, 3)) -> scalar 
        tmp_mat = self.xp.matmul(self.xp.conjugate(het_r0)[:, :, self.xp.newaxis], het_r1[:, self.xp.newaxis, :]) # (Nb, 3, 3)
        LL1 += self.xp.sum((tmp_mat + self.xp.transpose((self.xp.conjugate(tmp_mat)), (0, 2, 1))) * self.B1) # sum((Nb, 3, 3) * (Nb, 3, 3)) -> scalar 
        # 2_ d_h term 
        LL2 = self.xp.sum(self.xp.matmul(self.A0, het_r0[:, :, self.xp.newaxis]))
        LL2 += self.xp.sum(self.xp.matmul(self.A1, het_r1[:, :, self.xp.newaxis]))

        # self.output = [LL1, self.xp.real(LL2)]
        
        res = self.xp.nan_to_num(self.xp.real(-0.5 * LL1 + LL2), nan=-self.xp.infty)
        if self.use_gpu: 
            return res.get()
        else:
            return res
        
        
    def het_log_like_vectorized(self, parameter_array):
        """ 
        Args: 
            parameter_array: parameters given as an array of shape (Nparams, Nevents)
            the order is: ['log_chirp_mass', 'mass_ratio', 'spin_1z', 'spin_2z', 'coalescence_time', 'coalescence_phase', 'log_luminosity_distance', 'cos_inclination', 'longitude', 'sin_latitude', 'psi']
        Returns: 
            numpy array of heterodyned log-likelihoods 
        """
        if not self.het_prepare_flag: 
            raise NotImplementedError("heterodyne not prepared, run preparation first.")
            
        # calculate sparse template 
        het_h = self.xp.transpose(self.response_generator.Response(
            parameters=self.ParamArr2ParamDict(parameter_array),
            freqs=self.het_frequency,
            **self.response_kwargs,
        ), (1, 0, 2)) # (Nevents, 3, N_het_f)
        
        # calculate heterodyne 
        het_r = self.xp.transpose(het_h / self.het_h0, (0, 2, 1)) # (Nevents, N_het_f, 3)
        het_r = self.xp.nan_to_num(het_r, 0.) # deal with the nan caused by divide 0 
        het_r0 = het_r[:, :-1, :] # (Nevents, Nb, 3)
        het_r1 = (het_r[:, 1:, :] - het_r0) / self.het_df[:, self.xp.newaxis] # (Nevents, Nb, 3)
        
        # calculate likelihood 
        # 1) h_h term 
        LL1 = self.xp.sum(self.xp.matmul(self.xp.conjugate(het_r0)[:, :, :, self.xp.newaxis], het_r0[:, :, self.xp.newaxis, :]) * self.B0, axis=(1, 2, 3)) # sum((Nevents, Nb, 3, 3) * (Nb, 3, 3)) -> (Nevents)
        tmp_mat = self.xp.matmul(self.xp.conjugate(het_r0)[:, :, :, self.xp.newaxis], het_r1[:, :, self.xp.newaxis, :]) # (Nevents, Nb, 3, 3)
        LL1 += self.xp.sum((tmp_mat + self.xp.transpose((self.xp.conjugate(tmp_mat)), (0, 1, 3, 2))) * self.B1, axis=(1, 2, 3)) # sum((Nevents, Nb, 3, 3) * (Nb, 3, 3)) -> (Nevents)
        # 2_ d_h term 
        LL2 = self.xp.sum(self.xp.matmul(self.A0[self.xp.newaxis, :, :, :], het_r0[:, :, :, self.xp.newaxis]), axis=(1, 2, 3)) # (Nevents)
        LL2 += self.xp.sum(self.xp.matmul(self.A1[self.xp.newaxis, :, :, :], het_r1[:, :, :, self.xp.newaxis]), axis=(1, 2, 3)) # (Nevents)

        # self.output_vec = [LL1, self.xp.real(LL2)]
        
        res = self.xp.nan_to_num(self.xp.real(-0.5 * LL1 + LL2), nan=-self.xp.infty)
        if self.use_gpu: 
            return res.get()
        else:
            return res
        
    def marginal_log_like(self, parameter_array):
        """ 
        Args: 
            parameter_array: parameters given as an array.
            the order is: ['log_chirp_mass', 'mass_ratio', 'spin_1z', 'spin_2z', 'coalescence_time', 'coalescence_phase', 'cos_inclination', 'longitude', 'sin_latitude', 'psi']
        Returns: 
            loglike marginalized over luminosity distance 
        """
        # p = dict()
        # p['chirp_mass'] = np.power(10., parameter_array[0])
        # p['mass_ratio'] = parameter_array[1]
        # p['spin_1z'] = parameter_array[2]
        # p['spin_2z'] = parameter_array[3]
        # p['coalescence_time'] = parameter_array[4]
        # p['coalescence_phase'] = parameter_array[5]
        # p['luminosity_distance'] = 1. 
        # p['inclination'] = np.arccos(parameter_array[6])
        # p['longitude'] = parameter_array[7]
        # p['latitude'] = np.arcsin(parameter_array[8])
        # p['psi'] = parameter_array[9] 
        full_parameter_array = np.array(list(parameter_array[:6])+[0.,]+list(parameter_array[6:]))
        p = self.ParamArr2ParamDict(full_parameter_array)

        template = self.response_generator.Response(
            parameters=p,
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (3, Nf)
        B_term = self.xp.real(self.FrequencyDomainCovarianceInnerProduct(data_channels1=template, data_channels2=self.data, inv_cov=self.invserse_covariance_matrix))
        if B_term < 0.:
            result = 0. 
        else:
            C_term = self.FrequencyDomainCovarianceSNR(data_channels=template, inv_cov=self.invserse_covariance_matrix) ** 2
            result = B_term ** 2 / C_term / 2.

        if self.use_gpu:
            return result.get() 
        else:
            return result
        
    def calculate_marginalized_parameter(self, parameter_array):
        """ 
        Args: 
            parameter_array: parameters given as an array.
            the order is: ['log_chirp_mass', 'mass_ratio', 'spin_1z', 'spin_2z', 'coalescence_time', 'coalescence_phase', 'cos_inclination', 'longitude', 'sin_latitude', 'psi']
        Returns: 
            calculate luminosity distance in [MPC] given the maximum estimation of other parameters  
        """
        p = dict()
        p['chirp_mass'] = np.power(10., parameter_array[0])
        p['mass_ratio'] = parameter_array[1]
        p['spin_1z'] = parameter_array[2]
        p['spin_2z'] = parameter_array[3]
        p['coalescence_time'] = parameter_array[4]
        p['coalescence_phase'] = parameter_array[5]
        p['luminosity_distance'] = 1. 
        p['inclination'] = np.arccos(parameter_array[6])
        p['longitude'] = parameter_array[7]
        p['latitude'] = np.arcsin(parameter_array[8])
        p['psi'] = parameter_array[9] 

        template = self.response_generator.Response(
            parameters=p,
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (3, Nf)
        B_term = self.xp.real(self.FrequencyDomainCovarianceInnerProduct(data_channels1=template, data_channels2=self.data, inv_cov=self.invserse_covariance_matrix))
        C_term = self.FrequencyDomainCovarianceSNR(data_channels=template, inv_cov=self.invserse_covariance_matrix) ** 2 

        if self.use_gpu:
            return np.abs((C_term / B_term).get())
        else:
            return np.abs(C_term / B_term)
        
    def FrequencyDomainCovarianceSNR(self, data_channels, inv_cov):
        """  
        Args:
            data_channels: numpy array of shape (3, Nf)
            inv_cov: numpy array of shape (Nf, 3, 3), with Cov_IJ = CSD_IJ / 4 / Df, CSD_IJ = 2<I J^*>/T
        Returns: 
            sqrt(d^\dagger C^-1 d), real scalar 
        """
        data_expanded = self.xp.transpose(data_channels)[:,self.xp.newaxis,:] # (Nf, 1, 3)
        data_expanded_transposed = self.xp.transpose(data_expanded, (0, 2, 1)) # (Nf, 3, 1)
        tmp1 = self.xp.matmul(self.xp.conjugate(data_expanded), inv_cov) # (Nf, 1, 3)
        tmp2 = self.xp.matmul(tmp1, data_expanded_transposed) # (Nf, 1, 1)
        return self.xp.sqrt(self.xp.real(self.xp.sum(tmp2)))
    
    def FrequencyDomainCovarianceInnerProduct(self, data_channels1, data_channels2, inv_cov):
        """  
        Args:
            data_channels1, 2: numpy arrays of shapes (3, Nf)
            inv_cov: numpy array of shape (Nf, 3, 3), with Cov_IJ = CSD_IJ / 4 / Df, CSD_IJ = 2<I J^*>/T
        Returns: 
            d_1^\dagger C^-1 d_2, complex scalar 
        """
        data_expanded1 = self.xp.transpose(data_channels1)[:,self.xp.newaxis,:] # (Nf, 1, 3)
        data_expanded2 = self.xp.transpose(data_channels2)[:,self.xp.newaxis,:] # (Nf, 1, 3)
        data_expanded_transposed2 = self.xp.transpose(data_expanded2, (0, 2, 1)) # (Nf, 3, 1)
        tmp1 = self.xp.matmul(self.xp.conjugate(data_expanded1), inv_cov) # (Nf, 1, 3)
        tmp2 = self.xp.matmul(tmp1, data_expanded_transposed2) # (Nf, 1, 1)
        return self.xp.sum(tmp2)
    
    
class HMLikelihood(Likelihood):
    def __init__(self, response_generator, frequency, data, invserse_covariance_matrix, response_parameters, Fref_waveform=False, use_gpu=False, verbose=0):
        super().__init__(response_generator, frequency, data, invserse_covariance_matrix, response_parameters, Fref_waveform, use_gpu, verbose)
        self.NX = self.xp.newaxis
        self.TRANS = self.xp.transpose
        
        # force output by mode 
        self.response_kwargs["output_by_mode"] = True 
        
    def prepare_het_log_like(self, base_parameters, num_het_frequency=128):
        """  
            base_parameters: dictionary of parameters 
        """
        self.h0 = self.response_generator.Response(
            # parameters=self.ParamArr2ParamDict(base_parameters),
            parameters=base_parameters, 
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (Nchannels, Nmodes, Nf)
        
        self.Nchannels = len(self.h0)
        self.Nmodes = len(self.h0[0])
        
        # create sparse grid of frequencies (1st try)
        FMIN, FMAX = self.xp.min(self.frequency) * 0.999999999999, self.xp.max(self.frequency) * 1.000000000001
        self.het_frequency = self.xp.logspace(self.xp.log10(FMIN), self.xp.log10(FMAX), num_het_frequency) # N_het_f
        
        # calculate base waveform at the sparse grid (1st try)
        self.het_h0 = self.response_generator.Response(
            # parameters=self.ParamArr2ParamDict(base_parameters),
            parameters=base_parameters, 
            freqs=self.het_frequency,
            **self.response_kwargs,
        ) # (Nchannels, Nmodes, N_het_f)
        
        # refine the sparse grid to ensure no zero waveforms 
        valid_idx = self.xp.where(self.xp.abs(self.het_h0[0][0]) > 1e-25)[0]
        tmpf = self.het_frequency[valid_idx]
        
        # create sparse grid of frequencies (final)
        self.het_frequency = self.xp.logspace(self.xp.log10(tmpf[0]), self.xp.log10(tmpf[-1]), num_het_frequency) # N_het_f
        
        # calculate base waveform at the sparse grid (final)
        self.het_h0 = self.response_generator.Response(
            # parameters=self.ParamArr2ParamDict(base_parameters),
            parameters=base_parameters, 
            freqs=self.het_frequency,
            **self.response_kwargs,
        ) # (Nchannels, Nmodes, N_het_f)
        
        # avoid singularity when calculating r = h / h0
        self.het_h0[self.xp.abs(self.het_h0)<1e-25] = 1e-25
        # self.het_h0[self.xp.abs(self.het_h0)<1e-23] = 1e-23 
        
        # confine the frequency and data to be within the boundaries of sparse grid 
        inband_idx = self.xp.where((self.frequency >= self.het_frequency[0]) & (self.frequency <= self.het_frequency[-1]))[0]
        self.dense_frequency = self.frequency[inband_idx] # (Nf)
        self.dense_data = self.data[:, inband_idx] # (Nchannels, Nf)
        self.dense_h0 = self.h0[:, :, inband_idx] # (Nchannels, Nmodes, Nf)
        self.dense_invserse_covariance_matrix = self.invserse_covariance_matrix[inband_idx] # (Nf, Nchannels, Nchannels)
        self.Nfreqs_dense = len(self.dense_frequency)
        
        # group the dense frequencies with the sparse frequency grid, return the left sparse idx of each dense frequency
        group_idx = self.xp.searchsorted(self.het_frequency, self.dense_frequency, "right") - 1 # (Nf)
        
        # calculate the coefficients of heterodyned likelihood 
        # 1) the (h|h) term 
        A_dense = self.xp.zeros((self.Nmodes, self.Nmodes, self.Nchannels, self.Nchannels, self.Nfreqs_dense), dtype=self.xp.complex128)
        B_dense = self.xp.zeros((self.Nmodes, self.Nmodes, self.Nchannels, self.Nchannels, self.Nfreqs_dense), dtype=self.xp.complex128)
        C_dense = self.xp.zeros((self.Nmodes, self.Nmodes, self.Nchannels, self.Nchannels, self.Nfreqs_dense), dtype=self.xp.complex128)
        for lmode in range(self.Nmodes):
            for lpmode in range(self.Nmodes):
                A_dense[lmode][lpmode] = self.inner_product_frequency_array(h1=self.dense_frequency*self.dense_h0[:, lpmode], h2=self.dense_frequency*self.dense_h0[:, lmode], inv_cov=self.dense_invserse_covariance_matrix) # (Nchannels, Nchannels, Nf)
                B_dense[lmode][lpmode] = self.inner_product_frequency_array(h1=self.dense_frequency*self.dense_h0[:, lpmode], h2=self.dense_h0[:, lmode], inv_cov=self.dense_invserse_covariance_matrix) # (Nchannels, Nchannels, Nf)
                C_dense[lmode][lpmode] = self.inner_product_frequency_array(h1=self.dense_h0[:, lpmode], h2=self.dense_h0[:, lmode], inv_cov=self.dense_invserse_covariance_matrix) # (Nchannels, Nchannels, Nf)
        self.Nbin = num_het_frequency - 1 
        self.A_sparse = self.xp.zeros((self.Nmodes, self.Nmodes, self.Nchannels, self.Nchannels, self.Nbin), dtype=self.xp.complex128)
        self.B_sparse = self.xp.zeros((self.Nmodes, self.Nmodes, self.Nchannels, self.Nchannels, self.Nbin), dtype=self.xp.complex128)
        self.C_sparse = self.xp.zeros((self.Nmodes, self.Nmodes, self.Nchannels, self.Nchannels, self.Nbin), dtype=self.xp.complex128)
        for ibin in range(self.Nbin): # ibin for left index of bin 
            group_idx_in_bin = self.xp.where(group_idx==ibin)[0]
            self.A_sparse[:, :, :, :, ibin] = self.xp.sum(A_dense[:, :, :, :, group_idx_in_bin], axis=4) # (Nmodes, Nmodes, Nchannels, Nchannels)
            self.B_sparse[:, :, :, :, ibin] = self.xp.sum(B_dense[:, :, :, :, group_idx_in_bin], axis=4)
            self.C_sparse[:, :, :, :, ibin] = self.xp.sum(C_dense[:, :, :, :, group_idx_in_bin], axis=4)
        # 2) the (d|h) term 
        D_dense = self.xp.zeros((self.Nmodes, self.Nchannels, self.Nchannels, self.Nfreqs_dense), dtype=self.xp.complex128)
        E_dense = self.xp.zeros((self.Nmodes, self.Nchannels, self.Nchannels, self.Nfreqs_dense), dtype=self.xp.complex128)
        for lmode in range(self.Nmodes): 
            D_dense[lmode] = self.inner_product_frequency_array(h1=self.dense_frequency*self.dense_data, h2=self.dense_h0[:, lmode], inv_cov=self.dense_invserse_covariance_matrix) # (Nchannels, Nchannels, Nf)
            E_dense[lmode] = self.inner_product_frequency_array(h1=self.dense_data, h2=self.dense_h0[:, lmode], inv_cov=self.dense_invserse_covariance_matrix) # (Nchannels, Nchannels, Nf)
        self.D_sparse = self.xp.zeros((self.Nmodes, self.Nchannels, self.Nchannels, self.Nbin), dtype=self.xp.complex128)
        self.E_sparse = self.xp.zeros((self.Nmodes, self.Nchannels, self.Nchannels, self.Nbin), dtype=self.xp.complex128)
        for ibin in range(self.Nbin): 
            group_idx_in_bin = self.xp.where(group_idx==ibin)[0]
            self.D_sparse[:, :, :, ibin] = self.xp.sum(D_dense[:, :, :, group_idx_in_bin], axis=3) # (Nmodes, Nchannels, Nchannels)
            self.E_sparse[:, :, :, ibin] = self.xp.sum(E_dense[:, :, :, group_idx_in_bin], axis=3)
            
        self.het_df = self.het_frequency[1:] - self.het_frequency[:-1] # Nbins = N_het_f - 1
        self.het_prepare_flag = True 
        
    def het_log_like(self, parameter_array):
        """ 
        Parameters: 
            parameter_array: parameters given as an array, the order is: ['log_chirp_mass', 'mass_ratio', 'spin_1z', 'spin_2z', 'coalescence_time', 'coalescence_phase', 'log_luminosity_distance', 'cos_inclination', 'longitude', 'sin_latitude', 'psi']
        Returns: 
            heterodyned log-likelihood (scalar)
        """
        if not self.het_prepare_flag: 
            raise NotImplementedError("Heterodyne not prepared.")
        
        # calculate sparse template 
        het_h = self.response_generator.Response(
            parameters=self.ParamArr2ParamDict(parameter_array),
            freqs=self.het_frequency,
            **self.response_kwargs,
        ) # (Nchannels, Nmodes, N_het_f)
        
        # calculate heterodyne 
        # het_r = self.xp.nan_to_num(het_h / self.het_h0, nan=0.) # (Nchannels, Nmodes, N_het_f)
        het_r = het_h / self.het_h0 # (Nchannels, Nmodes, N_het_f)
        het_r_right = het_r[:, :, 1:] # (Nchannels, Nmodes, Nbins)
        het_r_left = het_r[:, :, :-1] # (Nchannels, Nmodes, Nbins)
        alpha = self.TRANS((het_r_right - het_r_left) / self.het_df, axes=(1, 0, 2)) # (Nchannels, Nmodes, Nbins) -> (Nmodes, Nchannels, Nbins)
        alpha_star = self.xp.conjugate(alpha)
        beta = self.TRANS((het_r_left * self.het_frequency[1:] - het_r_right * self.het_frequency[:-1]) / self.het_df, axes=(1, 0, 2)) # (Nchannels, Nmodes, Nbins) -> (Nmodes, Nchannels, Nbins)
        beta_star = self.xp.conjugate(beta)
        
        # calculate likelihood 
        # 1) (h|h)
        hh_term = alpha[:, self.NX, :, self.NX] * alpha_star[self.NX, :, self.NX, :] * self.A_sparse # (Nmodes, Nmodes, Nchannels, Nchannels, Nbins)
        hh_term += (alpha[:, self.NX, :, self.NX] * beta_star[self.NX, :, self.NX, :] + beta[:, self.NX, :, self.NX] * alpha_star[self.NX, :, self.NX, :]) * self.B_sparse # (Nmodes, Nmodes, Nchannels, Nchannels, Nbins)
        hh_term += beta[:, self.NX, :, self.NX] * beta_star[self.NX, :, self.NX, :] * self.C_sparse # (Nmodes, Nmodes, Nchannels, Nchannels, Nbins)
        hh_term = self.xp.real(self.xp.sum(hh_term)) # scalar
        # 2) (d|h)
        dh_term = self.xp.real(self.xp.sum(alpha[:, :, self.NX] * self.D_sparse + beta[:, :, self.NX] * self.E_sparse)) # scalar 
        
        return dh_term - 0.5 * hh_term
    
    def het_log_like_vectorized(self, parameter_array):
        """ 
        Parameters: 
            parameter_array: parameters given as a (Nparams, Nevents) array, the order is: ['log_chirp_mass', 'mass_ratio', 'spin_1z', 'spin_2z', 'coalescence_time', 'coalescence_phase', 'log_luminosity_distance', 'cos_inclination', 'longitude', 'sin_latitude', 'psi']
        Returns: 
            heterodyned log-likelihoods (Nevents,)
        """
        if not self.het_prepare_flag: 
            raise NotImplementedError("Heterodyne not prepared.")
        
        # calculate sparse template 
        het_h = self.TRANS(self.response_generator.Response(
            parameters=self.ParamArr2ParamDict(parameter_array),
            freqs=self.het_frequency,
            **self.response_kwargs,
        ), axes=(2, 0, 1, 3)) # (Nchannels, Nmodes, Nevents, N_het_f) -> (Nevents, Nchannels, Nmodes, N_het_f)
        
        # calculate heterodyne 
        # het_r = self.xp.nan_to_num(het_h / self.het_h0, nan=0.) # (Nevents, Nchannels, Nmodes, N_het_f)
        het_r = het_h / self.het_h0 # (Nevents, Nchannels, Nmodes, N_het_f)
        het_r_right = het_r[:, :, :, 1:] # (Nevents, Nchannels, Nmodes, Nbins)
        het_r_left = het_r[:, :, :, :-1] # (Nevents, Nchannels, Nmodes, Nbins)
        alpha = self.TRANS((het_r_right - het_r_left) / self.het_df, axes=(0, 2, 1, 3)) # (Nevents, Nchannels, Nmodes, Nbins) -> (Nevents, Nmodes, Nchannels, Nbins)
        alpha_star = self.xp.conjugate(alpha)
        beta = self.TRANS((het_r_left * self.het_frequency[1:] - het_r_right * self.het_frequency[:-1]) / self.het_df, axes=(0, 2, 1, 3)) # (Nevents, Nchannels, Nmodes, Nbins) -> (Nevents, Nmodes, Nchannels, Nbins)
        beta_star = self.xp.conjugate(beta)
        
        # calculate likelihood 
        # 1) (h|h)
        hh_term = alpha[:, :, self.NX, :, self.NX, :] * alpha_star[:, self.NX, :, self.NX, :, :] * self.A_sparse # (Nevents, Nmodes, Nmodes, Nchannels, Nchannels, Nbins)
        hh_term += (alpha[:, :, self.NX, :, self.NX, :] * beta_star[:, self.NX, :, self.NX, :, :] + beta[:, :, self.NX, :, self.NX, :] * alpha_star[:, self.NX, :, self.NX, :, :]) * self.B_sparse # (Nevents, Nmodes, Nmodes, Nchannels, Nchannels, Nbins)
        hh_term += beta[:, :, self.NX, :, self.NX, :] * beta_star[:, self.NX, :, self.NX, :, :] * self.C_sparse # (Nevents, Nmodes, Nmodes, Nchannels, Nchannels, Nbins)
        hh_term = self.xp.real(self.xp.sum(hh_term, axis=(1, 2, 3, 4, 5))) # (Nevents)
        # 2) (d|h)
        dh_term = self.xp.real(self.xp.sum(alpha[:, :, :, self.NX, :] * self.D_sparse + beta[:, :, :, self.NX, :] * self.E_sparse, axis=(1, 2, 3, 4))) # (Nevents)
        
        if self.use_gpu: 
            return (dh_term - 0.5 * hh_term).get() # (Nevents)
        else: 
            return dh_term - 0.5 * hh_term # (Nevents)
        
    def inner_product_frequency_array(self, h1, h2, inv_cov): 
        """  
        Parameters: 
            h1, h2: fourier array of shape (Nchannels, Nf,)
            inv_cov: inverse covariance matrix of shape (Nf, Nchannels, Nchannels)
        Returns: 
            h1^\dagger C^-1 h2, neither summed over channels nor frequencies, the shape is (Nchannels, Nchannels, Nf)
        """
        # tmp1 = self.xp.sum(self.TRANS(self.xp.conjugate(h1))[:, :, self.NX] * inv_cov, axis=1) # (Nf, Nchannels) 
        # tmp2 = self.xp.sum(tmp1 * self.TRANS(h2), axis=1) # (Nf,)
        # return tmp2
        tmp1 = self.TRANS(self.xp.conjugate(h1))[:, :, self.NX] * inv_cov # (Nf, Nchannels, Nchannels) 
        tmp2 = tmp1 * self.TRANS(h2)[:, self.NX, :] # (Nf, Nchannels, Nchannels)
        return self.TRANS(tmp2, axes=(1, 2, 0)) # (Nchannels, Nchannels, Nf)
        
    

import copy 

class Fstatistics(Likelihood):
    extrinsic_parameter_names = [
        "luminosity_distance", 
        "inclination", 
        "coalescence_phase", 
        "psi"
        ]
    intrinsic_parameter_names = [
        'chirp_mass',
        'mass_ratio',
        'spin_1z',
        'spin_2z',
        'coalescence_time',
        'longitude',
        'latitude'
        ]
    # def __init__(self, response_generator, frequency, data, invserse_covariance_matrix, response_parameters, use_gpu=False):
    #     super().__init__(response_generator, frequency, data, invserse_covariance_matrix, response_parameters, use_gpu)
    def __init__(self, response_generator, frequency, data, invserse_covariance_matrix, response_parameters, Fref_waveform=False, use_gpu=False, verbose=0):
        super().__init__(response_generator, frequency, data, invserse_covariance_matrix, response_parameters, Fref_waveform, use_gpu, verbose)
        self.SUM = self.xp.sum 
        self.CONJ = self.xp.conjugate
        self.RE = self.xp.real
        self.NX = self.xp.newaxis 
        self.MATMUL = self.xp.matmul
        self.TRANS = self.xp.transpose

    def self_inner_product_vectorized(self, template_channels):
        """ 
            template_channels: shape (Nevent, Nchannel, Nfreq)
        """
        residual = self.TRANS(template_channels, (0, 2, 1)) # (Nevent, 3, Nf) -> (Nevent, Nf, 3)
        residual_dagger = self.CONJ(residual[:, :, self.NX, :]) # (Nevent, Nf, 1, 3)
        residual = residual[:, :, :, self.NX] # (Nevent, Nf, 3, 1)
        inners = self.SUM(self.MATMUL(self.MATMUL(residual_dagger, self.invserse_covariance_matrix), residual), axis=(1,2,3)) # (Nevent)
        return self.RE(inners) # (Nevent)
    
    def inner_product_vectorized(self, template_channels1, template_channels2):
        """ 
            template_channels1: shape (Nevent, Nchannel, Nfreq)
            template_channels2: shape (Nevent, Nchannel, Nfreq)
        """
        residual1 = self.TRANS(template_channels1, (0, 2, 1)) # (Nevent, 3, Nf) -> (Nevent, Nf, 3)
        residual_dagger1 = self.CONJ(residual1[:, :, self.NX, :]) # (Nevent, Nf, 1, 3)

        residual2 = self.TRANS(template_channels2, (0, 2, 1)) # (Nevent, 3, Nf) -> (Nevent, Nf, 3)
        residual2 = residual2[:, :, :, self.NX] # (Nevent, Nf, 3, 1)

        inners = self.SUM(self.MATMUL(self.MATMUL(residual_dagger1, self.invserse_covariance_matrix), residual2), axis=(1,2,3)) # (Nevent)
        return self.RE(inners) # (Nevent)
    
    def calculate_Fstat(self, intrinsic_parameters, return_a=False, return_recovered_wave=False):
        """  
        calculate F-statistics for a batch of events TODO: expand to HM waveform 
        Args: 
            intrinsic_parameters: dictionary of intrinsic parameters (except for D, iota, phic, psi), each parameter is a float number. 
        Returns: 
            F-statistics
        """        
        full_parameters1 = intrinsic_parameters.copy() 
        full_parameters1["luminosity_distance"] = 0.25 
        full_parameters1["coalescence_phase"] = 0.
        full_parameters1["inclination"] = PI / 2. 
        full_parameters1["psi"] = 0.

        temp1 = self.response_generator.Response(
            parameters=full_parameters1,
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (Nchannel=3, Nfreq)
        
        full_parameters2 = full_parameters1.copy() 
        full_parameters2["psi"] = PI / 4. 

        temp2 = self.response_generator.Response(
            parameters=full_parameters2,
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (Nchannel=3, Nfreq)

        X1 = temp1 # (Nchannel, Nfreq)
        X2 = 1.j * X1 # (Nchannel, Nfreq)
        X3 = temp2 # (Nchannel, Nfreq)
        X4 = 1.j * X3 # (Nchannel, Nfreq) 
        # print("shape of X1:", X1.shape) # TEST 
        
        Nvector = self.RE(self.xp.array([
            FrequencyDomainCovarianceInnerProduct(self.data, X1, self.invserse_covariance_matrix), 
            FrequencyDomainCovarianceInnerProduct(self.data, X2, self.invserse_covariance_matrix),
            FrequencyDomainCovarianceInnerProduct(self.data, X3, self.invserse_covariance_matrix),
            FrequencyDomainCovarianceInnerProduct(self.data, X4, self.invserse_covariance_matrix),
        ])) # (4,) all real numbers 
        # print("shape of N vector:", Nvector.shape) # TEST 
        
        M12 = FrequencyDomainCovarianceInnerProduct(X1, X2, self.invserse_covariance_matrix)
        M13 = FrequencyDomainCovarianceInnerProduct(X1, X3, self.invserse_covariance_matrix)
        M14 = FrequencyDomainCovarianceInnerProduct(X1, X4, self.invserse_covariance_matrix)
        M23 = FrequencyDomainCovarianceInnerProduct(X2, X3, self.invserse_covariance_matrix)
        M24 = FrequencyDomainCovarianceInnerProduct(X2, X4, self.invserse_covariance_matrix)
        M34 = FrequencyDomainCovarianceInnerProduct(X3, X4, self.invserse_covariance_matrix)
        Mmatrix = self.RE(self.xp.array([
            [FrequencyDomainCovarianceInnerProduct(X1, X1, self.invserse_covariance_matrix), M12, M13, M14], 
            [M12, FrequencyDomainCovarianceInnerProduct(X2, X2, self.invserse_covariance_matrix), M23, M24], 
            [M13, M23, FrequencyDomainCovarianceInnerProduct(X3, X3, self.invserse_covariance_matrix), M34], 
            [M14, M24, M34, FrequencyDomainCovarianceInnerProduct(X4, X4, self.invserse_covariance_matrix)]
        ])) # (4, 4) all real numbers 
        # print("shape of M matrix:", Mmatrix.shape) # TEST 
        
        # invMmatrix = self.xp.linalg.inv(Mmatrix) # (4, 4)
        # NM = self.MATMUL(invMmatrix, Nvector) # (4,)
        # NMN = self.MATMUL(Nvector, NM) # float 
        NM = np.linalg.solve(Mmatrix, Nvector) # (4,)
        res = 0.5 * Nvector @ NM # float, Fstat 0.5 * N^T M^{-1} N
        
        if return_a:
            res_a = NM
            return res_a # (4,)
            
        if return_recovered_wave: 
            res_a = NM # (4,)
            res_wf = res_a[0] * X1 + res_a[1] * X2 + res_a[2] * X3 + res_a[3] * X4 # (Nchannel, Nfreq)
            return res_wf # (Nchannel, Nfreq)

        return float(res) # float 

    def calculate_Fstat_vectorized(self, intrinsic_parameters, return_a=False, return_recovered_wave=False):
        """  
        calculate F-statistics for a batch of events TODO: expand to HM waveform 
        Args: 
            intrinsic_parameters: dictionary of intrinsic parameters (except for D, iota, phic, psi), each item is a numpy array of shape (Nevent). 
        Returns: 
            F-statistics of events 
        """
        Nevent = len(np.atleast_1d(intrinsic_parameters["chirp_mass"]))
        
        full_parameters1 = copy.deepcopy(intrinsic_parameters)
        full_parameters1["luminosity_distance"] = np.ones(Nevent) * 0.25 
        full_parameters1["coalescence_phase"] = np.zeros(Nevent)
        full_parameters1["inclination"] = np.ones(Nevent) * PI / 2. 
        full_parameters1["psi"] = np.zeros(Nevent)
        # print("1st parameter set:") # TEST 
        # print(full_parameters1) # TEST 

        temp1 = self.response_generator.Response(
            parameters=full_parameters1,
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (Nchannel=3, Nevent, Nfreq)
        
        full_parameters2 = copy.deepcopy(full_parameters1)
        full_parameters2["psi"] = np.ones(Nevent) * PI / 4. 
        # print("2nd parameter set:") # TEST 
        # print(full_parameters2) # TEST 

        temp2 = self.response_generator.Response(
            parameters=full_parameters2,
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (Nchannel=3, Nevent, Nfreq)

        if Nevent == 1:
            temp1 = temp1[:, self.NX, :]
            temp2 = temp2[:, self.NX, :]

        X1 = self.TRANS(temp1, axes=(1, 0, 2)) # (Nevent, Nchannel, Nfreq)
        X2 = 1.j * X1 # (Nevent, Nchannel, Nfreq)
        X3 = self.TRANS(temp2, axes=(1, 0, 2)) # (Nevent, Nchannel, Nfreq)
        X4 = 1.j * X3 # (Nevent, Nchannel, Nfreq) 
        # print("shape of X1:", X1.shape) # TEST 
        
        data_expand = self.data[self.NX, :, :] # (1, Nchannel, Nfreq)
        Nvector = self.TRANS(self.xp.array([
            self.inner_product_vectorized(data_expand, X1), 
            self.inner_product_vectorized(data_expand, X2), 
            self.inner_product_vectorized(data_expand, X3), 
            self.inner_product_vectorized(data_expand, X4), 
        ])) # (4, Nevent) -> (Nevent, 4) inner products, all real numbers 
        # print("shape of N vector:", Nvector.shape) # TEST 
        
        M12 = self.inner_product_vectorized(X1, X2) # (Nevent), real numbers 
        M13 = self.inner_product_vectorized(X1, X3)
        M14 = self.inner_product_vectorized(X1, X4)
        M23 = self.inner_product_vectorized(X2, X3)
        M24 = self.inner_product_vectorized(X2, X4)
        M34 = self.inner_product_vectorized(X3, X4)
        Mmatrix = self.TRANS(self.xp.array([
            [self.self_inner_product_vectorized(X1), M12, M13, M14], 
            [M12, self.self_inner_product_vectorized(X2), M23, M24], 
            [M13, M23, self.self_inner_product_vectorized(X3), M34], 
            [M14, M24, M34, self.self_inner_product_vectorized(X4)]
        ]), axes=(2, 0, 1)) # (4, 4, Nevent) -> (Nevent, 4, 4) inner products, all real numbers 
        # print("shape of M matrix:", Mmatrix.shape) # TEST 
        
        invMmatrix = self.xp.linalg.inv(Mmatrix) # (Nevent, 4, 4)
        Nvector_col = Nvector[..., self.NX] # (Nevent, 4, 1)
        NM = self.MATMUL(invMmatrix, Nvector_col) # (Nevent, 4, 1)
        Nvector_row = Nvector[:, self.NX, :] # (Nevent, 1, 4)
        NMN = self.MATMUL(Nvector_row, NM) # (Nevent, 1, 1)
        
        res = 0.5 * NMN[:, 0, 0] # (Nevent) Fstat 0.5 * N^T M^{-1} N
        
        if return_a:
            res_a = NM.squeeze(axis=-1) # (Nevent, 4)
            if self.use_gpu:
                return res_a.get() # (Nevent, 4)
            else: 
                return res_a # (Nevent, 4)
            
        if return_recovered_wave: 
            res_a = NM.squeeze(axis=-1) # (Nevent, 4)
            res_wf = res_a[:, 0] * self.TRANS(X1, axes=(1, 2, 0)) # (Nchannel, Nfreq, Nevent)
            res_wf += res_a[:, 1] * self.TRANS(X2, axes=(1, 2, 0))
            res_wf += res_a[:, 2] * self.TRANS(X3, axes=(1, 2, 0))
            res_wf += res_a[:, 3] * self.TRANS(X4, axes=(1, 2, 0)) 
            if Nevent == 1: 
                return res_wf[:, :, 0] # (Nchannel, Nfreq)
            else:
                return self.TRANS(res_wf, (0, 2, 1)) # (Nchannel, Nevent, Nfreq)

        # else:
        if self.use_gpu:
            if Nevent == 1:
                return res.get()[0]
            else:
                return res.get() # (Nevent)
        else: 
            if Nevent == 1: 
                return res[0]
            else: 
                return res 

    @staticmethod
    def a_to_extrinsic(a):
        """ 
        TODO: expand to HM waveform 
        Args: 
            a: (4), numpy array of the a coefficients 
        Returns: 
            dictionary of extrinsic parameters 
        """
        extrinsic_parameters = dict()
        
        P = np.linalg.norm(a) ** 2 # float 
        Q = a[1] * a[2] - a[0] * a[3] # float 
        Delta = np.sqrt(P ** 2 - 4. * Q ** 2) # float 
        Aplus = np.sqrt((P + Delta) / 2.) # float
        Across = np.sign(Q) * np.sqrt((P - Delta) / 2.) # float
        
        tmp = Aplus + np.sqrt(Aplus ** 2 - Across ** 2) # float 
        extrinsic_parameters["luminosity_distance"] = 0.5 / tmp # float 
        extrinsic_parameters["inclination"] = np.arccos(Across / tmp) # float 
        # extrinsic_parameters["coalescence_phase"] = -np.arctan(2. * (a[:, 0] * a[:, 1] + a[:, 2] * a[:, 3]) / (a[:, 0] ** 2 + a[:, 2] ** 2 - a[:, 1] ** 2 - a[:, 3] ** 2)) / 2. # (Nevent), one possible solution 
        # extrinsic_parameters["psi"] = np.arctan(2. * (a[:, 0] * a[:, 2] + a[:, 1] * a[:, 3]) / (a[:, 0] ** 2 + a[:, 1] ** 2 - a[:, 2] ** 2 - a[:, 3] ** 2)) / 4. # (Nevent), one possible solution 

        P = np.sqrt((a[0] + a[3])**2 + (a[1] - a[2])**2) # float 
        Q = np.sqrt((a[0] - a[3])**2 + (a[1] + a[2])**2) # float 
        Aplus = P + Q # float 
        Across = P - Q # float 
        extrinsic_parameters["psi"] = 0.5 * np.arctan2(Aplus*a[3] - Across*a[0], Aplus*a[1] + Across*a[2]) # float (-PI/2, PI/2)
        sgns2p = np.sign(np.sin(2. * extrinsic_parameters["psi"]))
        extrinsic_parameters["coalescence_phase"] = -0.5*np.arctan2((Aplus*a[3] - Across*a[0])*sgns2p, (Aplus*a[2] + Across*a[1])*sgns2p) # float (-PI/2, PI/2)
        if extrinsic_parameters["psi"] < 0: 
            extrinsic_parameters["psi"] += PI # (0, PI)
        if extrinsic_parameters["coalescence_phase"] < 0.: 
            extrinsic_parameters["coalescence_phase"] += PI # (0, PI)
           
        return extrinsic_parameters
        
    @staticmethod
    def a_to_extrinsic_vectorized(a):
        """ 
        TODO: expand to HM waveform 
        Args: 
            a: (Nevent, 4), numpy array of the a coefficients 
        Returns: 
            dictionary of extrinsic parameters 
        """
        extrinsic_parameters = dict()
        
        P = np.linalg.norm(a, axis=1) ** 2 # (Nevent)
        Q = a[:, 1] * a[:, 2] - a[:, 0] * a[:, 3] # (Nevent)
        Delta = np.sqrt(P ** 2 - 4. * Q ** 2) # (Nevent)
        Aplus = np.sqrt((P + Delta) / 2.) # (Nevent)
        Across = np.sign(Q) * np.sqrt((P - Delta) / 2.) # (Nevent)
        
        tmp = Aplus + np.sqrt(Aplus ** 2 - Across ** 2) # (Nevent)
        extrinsic_parameters["luminosity_distance"] = 0.5 / tmp # (Nevent)
        extrinsic_parameters["inclination"] = np.arccos(Across / tmp) # (Nevent)
        # extrinsic_parameters["coalescence_phase"] = -np.arctan(2. * (a[:, 0] * a[:, 1] + a[:, 2] * a[:, 3]) / (a[:, 0] ** 2 + a[:, 2] ** 2 - a[:, 1] ** 2 - a[:, 3] ** 2)) / 2. # (Nevent), one possible solution 
        # extrinsic_parameters["psi"] = np.arctan(2. * (a[:, 0] * a[:, 2] + a[:, 1] * a[:, 3]) / (a[:, 0] ** 2 + a[:, 1] ** 2 - a[:, 2] ** 2 - a[:, 3] ** 2)) / 4. # (Nevent), one possible solution 

        P = np.sqrt((a[:, 0] + a[:, 3])**2 + (a[:, 1] - a[:, 2])**2)
        Q = np.sqrt((a[:, 0] - a[:, 3])**2 + (a[:, 1] + a[:, 2])**2)
        Aplus = P + Q 
        Across = P - Q 
        extrinsic_parameters["psi"] = 0.5 * np.arctan2(Aplus*a[:, 3] - Across*a[:, 0], Aplus*a[:, 1] + Across*a[:, 2]) # (Nevent)
        sgns2p = np.sign(np.sin(2. * extrinsic_parameters["psi"])) # (Nevent)
        extrinsic_parameters["coalescence_phase"] = -0.5*np.arctan2((Aplus*a[:, 3] - Across*a[:, 0]) * sgns2p, (Aplus*a[:, 2] + Across*a[:, 1]) * sgns2p) # (Nevent)
        extrinsic_parameters["psi"][extrinsic_parameters["psi"]<0.] += PI 
        extrinsic_parameters["coalescence_phase"][extrinsic_parameters["coalescence_phase"]<0.] += PI 
        
        if a.shape[0] == 1:
            extrinsic_parameters_out = dict() 
            for k, v in extrinsic_parameters.items():
                extrinsic_parameters_out[k] = float(v[0])
            return extrinsic_parameters_out
        else:              
            return extrinsic_parameters
    
    @staticmethod
    def IntParamDict2ParamArr(param_dict):
        return np.array([
            np.log10(param_dict['chirp_mass']),
            param_dict['mass_ratio'],
            param_dict['spin_1z'],
            param_dict['spin_2z'],
            param_dict['coalescence_time'],
            param_dict['longitude'],
            np.sin(param_dict['latitude']),
        ]) # (Nparams, Nevent)

    @staticmethod
    def IntParamArr2ParamDict(params):
        p = dict()
        p['chirp_mass'] = np.power(10., params[0])
        p['mass_ratio'] = params[1]
        p['spin_1z'] = params[2]
        p['spin_2z'] = params[3]
        p['coalescence_time'] = params[4]
        p['longitude'] = params[5]
        p['latitude'] = np.arcsin(params[6])
        return p 
    
class FstatisticsFref(Likelihood):
    extrinsic_parameter_names = [
        "luminosity_distance", 
        "inclination", 
        "reference_phase", 
        "psi"
        ]
    intrinsic_parameter_names = [
        'chirp_mass',
        'mass_ratio',
        'spin_1z',
        'spin_2z',
        'reference_time',
        'longitude',
        'latitude'
        ]
    def __init__(self, response_generator, frequency, data, invserse_covariance_matrix, response_parameters, use_gpu=False, verbose=0):
        super().__init__(response_generator, frequency, data, invserse_covariance_matrix, response_parameters, True, use_gpu, verbose)
        self.SUM = self.xp.sum 
        self.CONJ = self.xp.conjugate
        self.RE = self.xp.real
        self.NX = self.xp.newaxis 
        self.MATMUL = self.xp.matmul
        self.TRANS = self.xp.transpose

    def self_inner_product_vectorized(self, template_channels):
        """ 
            template_channels: shape (Nevent, Nchannel, Nfreq)
        """
        residual = self.TRANS(template_channels, (0, 2, 1)) # (Nevent, 3, Nf) -> (Nevent, Nf, 3)
        residual_dagger = self.CONJ(residual[:, :, self.NX, :]) # (Nevent, Nf, 1, 3)
        residual = residual[:, :, :, self.NX] # (Nevent, Nf, 3, 1)
        inners = self.SUM(self.MATMUL(self.MATMUL(residual_dagger, self.invserse_covariance_matrix), residual), axis=(1,2,3)) # (Nevent)
        return self.RE(inners) # (Nevent)
    
    def inner_product_vectorized(self, template_channels1, template_channels2):
        """ 
            template_channels1: shape (Nevent, Nchannel, Nfreq)
            template_channels2: shape (Nevent, Nchannel, Nfreq)
        """
        residual1 = self.TRANS(template_channels1, (0, 2, 1)) # (Nevent, 3, Nf) -> (Nevent, Nf, 3)
        residual_dagger1 = self.CONJ(residual1[:, :, self.NX, :]) # (Nevent, Nf, 1, 3)

        residual2 = self.TRANS(template_channels2, (0, 2, 1)) # (Nevent, 3, Nf) -> (Nevent, Nf, 3)
        residual2 = residual2[:, :, :, self.NX] # (Nevent, Nf, 3, 1)

        inners = self.SUM(self.MATMUL(self.MATMUL(residual_dagger1, self.invserse_covariance_matrix), residual2), axis=(1,2,3)) # (Nevent)
        return self.RE(inners) # (Nevent)
    
    def calculate_Fstat(self, intrinsic_parameters, return_a=False, return_recovered_wave=False):
        """  
        calculate F-statistics for a batch of events TODO: expand to HM waveform 
        Args: 
            intrinsic_parameters: dictionary of intrinsic parameters (except for D, iota, phic, psi), each parameter is a float number. 
        Returns: 
            F-statistics
        """        
        full_parameters1 = intrinsic_parameters.copy() 
        full_parameters1["luminosity_distance"] = 0.25 
        full_parameters1["reference_phase"] = 0.
        full_parameters1["inclination"] = PI / 2. 
        full_parameters1["psi"] = 0.

        temp1 = self.response_generator.Response(
            parameters=full_parameters1,
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (Nchannel=3, Nfreq)
        
        full_parameters2 = full_parameters1.copy() 
        full_parameters2["psi"] = PI / 4. 

        temp2 = self.response_generator.Response(
            parameters=full_parameters2,
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (Nchannel=3, Nfreq)

        X1 = temp1 # (Nchannel, Nfreq)
        X2 = 1.j * X1 # (Nchannel, Nfreq)
        X3 = temp2 # (Nchannel, Nfreq)
        X4 = 1.j * X3 # (Nchannel, Nfreq) 
        # print("shape of X1:", X1.shape) # TEST 
        
        Nvector = self.RE(self.xp.array([
            FrequencyDomainCovarianceInnerProduct(self.data, X1, self.invserse_covariance_matrix), 
            FrequencyDomainCovarianceInnerProduct(self.data, X2, self.invserse_covariance_matrix),
            FrequencyDomainCovarianceInnerProduct(self.data, X3, self.invserse_covariance_matrix),
            FrequencyDomainCovarianceInnerProduct(self.data, X4, self.invserse_covariance_matrix),
        ])) # (4,) all real numbers 
        # print("shape of N vector:", Nvector.shape) # TEST 
        
        M12 = FrequencyDomainCovarianceInnerProduct(X1, X2, self.invserse_covariance_matrix)
        M13 = FrequencyDomainCovarianceInnerProduct(X1, X3, self.invserse_covariance_matrix)
        M14 = FrequencyDomainCovarianceInnerProduct(X1, X4, self.invserse_covariance_matrix)
        M23 = FrequencyDomainCovarianceInnerProduct(X2, X3, self.invserse_covariance_matrix)
        M24 = FrequencyDomainCovarianceInnerProduct(X2, X4, self.invserse_covariance_matrix)
        M34 = FrequencyDomainCovarianceInnerProduct(X3, X4, self.invserse_covariance_matrix)
        Mmatrix = self.RE(self.xp.array([
            [FrequencyDomainCovarianceInnerProduct(X1, X1, self.invserse_covariance_matrix), M12, M13, M14], 
            [M12, FrequencyDomainCovarianceInnerProduct(X2, X2, self.invserse_covariance_matrix), M23, M24], 
            [M13, M23, FrequencyDomainCovarianceInnerProduct(X3, X3, self.invserse_covariance_matrix), M34], 
            [M14, M24, M34, FrequencyDomainCovarianceInnerProduct(X4, X4, self.invserse_covariance_matrix)]
        ])) # (4, 4) all real numbers 
        # print("shape of M matrix:", Mmatrix.shape) # TEST 
        
        # invMmatrix = self.xp.linalg.inv(Mmatrix) # (4, 4)
        # NM = self.MATMUL(invMmatrix, Nvector) # (4,)
        # NMN = self.MATMUL(Nvector, NM) # float 
        NM = np.linalg.solve(Mmatrix, Nvector) # (4,)
        res = 0.5 * Nvector @ NM # float, Fstat 0.5 * N^T M^{-1} N
        
        if return_a:
            res_a = NM
            return res_a # (4,)
            
        if return_recovered_wave: 
            res_a = NM # (4,)
            res_wf = res_a[0] * X1 + res_a[1] * X2 + res_a[2] * X3 + res_a[3] * X4 # (Nchannel, Nfreq)
            return res_wf # (Nchannel, Nfreq)

        return float(res) # float 

    def calculate_Fstat_vectorized(self, intrinsic_parameters, return_a=False, return_recovered_wave=False):
        """  
        calculate F-statistics for a batch of events TODO: expand to HM waveform 
        Args: 
            intrinsic_parameters: dictionary of intrinsic parameters (except for D, iota, phic, psi), each item is a numpy array of shape (Nevent). 
        Returns: 
            F-statistics of events 
        """
        Nevent = len(np.atleast_1d(intrinsic_parameters["chirp_mass"]))
        
        full_parameters1 = copy.deepcopy(intrinsic_parameters)
        full_parameters1["luminosity_distance"] = np.ones(Nevent) * 0.25 
        full_parameters1["reference_phase"] = np.zeros(Nevent)
        full_parameters1["inclination"] = np.ones(Nevent) * PI / 2. 
        full_parameters1["psi"] = np.zeros(Nevent)
        # print("1st parameter set:") # TEST 
        # print(full_parameters1) # TEST 

        temp1 = self.response_generator.Response(
            parameters=full_parameters1,
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (Nchannel=3, Nevent, Nfreq)
        
        full_parameters2 = copy.deepcopy(full_parameters1)
        full_parameters2["psi"] = np.ones(Nevent) * PI / 4. 
        # print("2nd parameter set:") # TEST 
        # print(full_parameters2) # TEST 

        temp2 = self.response_generator.Response(
            parameters=full_parameters2,
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (Nchannel=3, Nevent, Nfreq)

        if Nevent == 1:
            temp1 = temp1[:, self.NX, :]
            temp2 = temp2[:, self.NX, :]

        X1 = self.TRANS(temp1, axes=(1, 0, 2)) # (Nevent, Nchannel, Nfreq)
        X2 = 1.j * X1 # (Nevent, Nchannel, Nfreq)
        X3 = self.TRANS(temp2, axes=(1, 0, 2)) # (Nevent, Nchannel, Nfreq)
        X4 = 1.j * X3 # (Nevent, Nchannel, Nfreq) 
        # print("shape of X1:", X1.shape) # TEST 
        
        data_expand = self.data[self.NX, :, :] # (1, Nchannel, Nfreq)
        Nvector = self.TRANS(self.xp.array([
            self.inner_product_vectorized(data_expand, X1), 
            self.inner_product_vectorized(data_expand, X2), 
            self.inner_product_vectorized(data_expand, X3), 
            self.inner_product_vectorized(data_expand, X4), 
        ])) # (4, Nevent) -> (Nevent, 4) inner products, all real numbers 
        # print("shape of N vector:", Nvector.shape) # TEST 
        
        M12 = self.inner_product_vectorized(X1, X2) # (Nevent), real numbers 
        M13 = self.inner_product_vectorized(X1, X3)
        M14 = self.inner_product_vectorized(X1, X4)
        M23 = self.inner_product_vectorized(X2, X3)
        M24 = self.inner_product_vectorized(X2, X4)
        M34 = self.inner_product_vectorized(X3, X4)
        Mmatrix = self.TRANS(self.xp.array([
            [self.self_inner_product_vectorized(X1), M12, M13, M14], 
            [M12, self.self_inner_product_vectorized(X2), M23, M24], 
            [M13, M23, self.self_inner_product_vectorized(X3), M34], 
            [M14, M24, M34, self.self_inner_product_vectorized(X4)]
        ]), axes=(2, 0, 1)) # (4, 4, Nevent) -> (Nevent, 4, 4) inner products, all real numbers 
        # print("shape of M matrix:", Mmatrix.shape) # TEST 
        
        invMmatrix = self.xp.linalg.inv(Mmatrix) # (Nevent, 4, 4)
        Nvector_col = Nvector[..., self.NX] # (Nevent, 4, 1)
        NM = self.MATMUL(invMmatrix, Nvector_col) # (Nevent, 4, 1)
        Nvector_row = Nvector[:, self.NX, :] # (Nevent, 1, 4)
        NMN = self.MATMUL(Nvector_row, NM) # (Nevent, 1, 1)
        
        res = 0.5 * NMN[:, 0, 0] # (Nevent) Fstat 0.5 * N^T M^{-1} N
        
        if return_a:
            res_a = NM.squeeze(axis=-1) # (Nevent, 4)
            if self.use_gpu:
                return res_a.get() # (Nevent, 4)
            else: 
                return res_a # (Nevent, 4)
            
        if return_recovered_wave: 
            res_a = NM.squeeze(axis=-1) # (Nevent, 4)
            res_wf = res_a[:, 0] * self.TRANS(X1, axes=(1, 2, 0)) # (Nchannel, Nfreq, Nevent)
            res_wf += res_a[:, 1] * self.TRANS(X2, axes=(1, 2, 0))
            res_wf += res_a[:, 2] * self.TRANS(X3, axes=(1, 2, 0))
            res_wf += res_a[:, 3] * self.TRANS(X4, axes=(1, 2, 0)) 
            if Nevent == 1: 
                return res_wf[:, :, 0] # (Nchannel, Nfreq)
            else:
                return self.TRANS(res_wf, (0, 2, 1)) # (Nchannel, Nevent, Nfreq)

        # else:
        if self.use_gpu:
            if Nevent == 1:
                return res.get()[0]
            else:
                return res.get() # (Nevent)
        else: 
            if Nevent == 1: 
                return res[0]
            else: 
                return res 

    @staticmethod
    def a_to_extrinsic(a):
        """ 
        TODO: expand to HM waveform 
        Args: 
            a: (4), numpy array of the a coefficients 
        Returns: 
            dictionary of extrinsic parameters 
        """
        extrinsic_parameters = dict()
        
        P = np.linalg.norm(a) ** 2 # float 
        Q = a[1] * a[2] - a[0] * a[3] # float 
        Delta = np.sqrt(P ** 2 - 4. * Q ** 2) # float 
        Aplus = np.sqrt((P + Delta) / 2.) # float
        Across = np.sign(Q) * np.sqrt((P - Delta) / 2.) # float
        
        tmp = Aplus + np.sqrt(Aplus ** 2 - Across ** 2) # float 
        extrinsic_parameters["luminosity_distance"] = 0.5 / tmp # float 
        extrinsic_parameters["inclination"] = np.arccos(Across / tmp) # float 

        P = np.sqrt((a[0] + a[3])**2 + (a[1] - a[2])**2) # float 
        Q = np.sqrt((a[0] - a[3])**2 + (a[1] + a[2])**2) # float 
        Aplus = P + Q # float 
        Across = P - Q # float 
        extrinsic_parameters["psi"] = 0.5 * np.arctan2(Aplus*a[3] - Across*a[0], Aplus*a[1] + Across*a[2]) # float (-PI/2, PI/2)
        sgns2p = np.sign(np.sin(2. * extrinsic_parameters["psi"]))
        extrinsic_parameters["reference_phase"] = -0.5*np.arctan2((Aplus*a[3] - Across*a[0])*sgns2p, (Aplus*a[2] + Across*a[1])*sgns2p) # float (-PI/2, PI/2)
        if extrinsic_parameters["psi"] < 0: 
            extrinsic_parameters["psi"] += PI # (0, PI)
        if extrinsic_parameters["reference_phase"] < 0.: 
            extrinsic_parameters["reference_phase"] += PI # (0, PI)
           
        return extrinsic_parameters
        
    @staticmethod
    def a_to_extrinsic_vectorized(a):
        """ 
        TODO: expand to HM waveform 
        Args: 
            a: (Nevent, 4), numpy array of the a coefficients 
        Returns: 
            dictionary of extrinsic parameters 
        """
        extrinsic_parameters = dict()
        
        P = np.linalg.norm(a, axis=1) ** 2 # (Nevent)
        Q = a[:, 1] * a[:, 2] - a[:, 0] * a[:, 3] # (Nevent)
        Delta = np.sqrt(P ** 2 - 4. * Q ** 2) # (Nevent)
        Aplus = np.sqrt((P + Delta) / 2.) # (Nevent)
        Across = np.sign(Q) * np.sqrt((P - Delta) / 2.) # (Nevent)
        
        tmp = Aplus + np.sqrt(Aplus ** 2 - Across ** 2) # (Nevent)
        extrinsic_parameters["luminosity_distance"] = 0.5 / tmp # (Nevent)
        extrinsic_parameters["inclination"] = np.arccos(Across / tmp) # (Nevent)
        # extrinsic_parameters["reference_phase"] = -np.arctan(2. * (a[:, 0] * a[:, 1] + a[:, 2] * a[:, 3]) / (a[:, 0] ** 2 + a[:, 2] ** 2 - a[:, 1] ** 2 - a[:, 3] ** 2)) / 2. # (Nevent), one possible solution 
        # extrinsic_parameters["psi"] = np.arctan(2. * (a[:, 0] * a[:, 2] + a[:, 1] * a[:, 3]) / (a[:, 0] ** 2 + a[:, 1] ** 2 - a[:, 2] ** 2 - a[:, 3] ** 2)) / 4. # (Nevent), one possible solution 

        P = np.sqrt((a[:, 0] + a[:, 3])**2 + (a[:, 1] - a[:, 2])**2)
        Q = np.sqrt((a[:, 0] - a[:, 3])**2 + (a[:, 1] + a[:, 2])**2)
        Aplus = P + Q 
        Across = P - Q 
        extrinsic_parameters["psi"] = 0.5 * np.arctan2(Aplus*a[:, 3] - Across*a[:, 0], Aplus*a[:, 1] + Across*a[:, 2]) # (Nevent)
        sgns2p = np.sign(np.sin(2. * extrinsic_parameters["psi"])) # (Nevent)
        extrinsic_parameters["reference_phase"] = -0.5*np.arctan2((Aplus*a[:, 3] - Across*a[:, 0]) * sgns2p, (Aplus*a[:, 2] + Across*a[:, 1]) * sgns2p) # (Nevent)
        extrinsic_parameters["psi"][extrinsic_parameters["psi"]<0.] += PI 
        extrinsic_parameters["reference_phase"][extrinsic_parameters["reference_phase"]<0.] += PI 
        
        if a.shape[0] == 1:
            extrinsic_parameters_out = dict() 
            for k, v in extrinsic_parameters.items():
                extrinsic_parameters_out[k] = float(v[0])
            return extrinsic_parameters_out
        else:              
            return extrinsic_parameters
    
    @staticmethod
    def IntParamDict2ParamArr(param_dict):
        return np.array([
            np.log10(param_dict['chirp_mass']),
            param_dict['mass_ratio'],
            param_dict['spin_1z'],
            param_dict['spin_2z'],
            param_dict['reference_time'],
            param_dict['longitude'],
            np.sin(param_dict['latitude']),
        ]) # (Nparams, Nevent)

    @staticmethod
    def IntParamArr2ParamDict(params):
        p = dict()
        p['chirp_mass'] = np.power(10., params[0])
        p['mass_ratio'] = params[1]
        p['spin_1z'] = params[2]
        p['spin_2z'] = params[3]
        p['reference_time'] = params[4]
        p['longitude'] = params[5]
        p['latitude'] = np.arcsin(params[6])
        return p 
    
    
class HMFstatistics_4params(Fstatistics):
    """  
        For waveforms with higher-order modes, in principle only 2 parameters can be analytically marginalized with F-stat.
        While, this less rigorous 4-dimensional reduction can be used to quickly determine the intrinsic parameters.
    """
    
    all_mode_factors = {
        "21": np.sqrt(5./PI)/4., 
        "22": np.sqrt(5./PI)/8., 
        "33": -np.sqrt(21./PI) / 8. / np.sqrt(2.), 
        "44": np.sqrt(7./PI) * 3. / 16., 
        "32": -np.sqrt(7./PI) / 4., 
        "43": np.sqrt(14./PI) * 3. / 16., 
    }

    def __init__(self, response_generator, frequency, data, invserse_covariance_matrix, response_parameters, Fref_waveform=False, use_gpu=False, verbose=0):
        super().__init__(response_generator, frequency, data, invserse_covariance_matrix, response_parameters, Fref_waveform, use_gpu, verbose) 
        self.response_kwargs = response_parameters.copy() 
        self.response_kwargs["output_by_mode"] = True 
        if self.response_kwargs.get("modes", None) is None: 
            self.modes = [(2, 2), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3)] 
        else: 
            self.modes = self.response_kwargs["modes"]
        self.Nmodes = len(self.modes)
        self.mode_factors = self.xp.array([self.all_mode_factors[str(mode[0])+str(mode[1])] for mode in self.modes])

        self.Nfreqs = len(frequency)
        
        if self.response_kwargs.get("drop_T", False): 
            self.Nchannels = 2 
        else: 
            self.Nchannels = 3 
            
        if Fref_waveform: 
            self.IntParamArr2ParamDictConversion = self.IntParamArr2ParamDictFRef
            self.IntParamDict2ParamArrConversion = self.IntParamDict2ParamArrFRef
            self.phase_name = "reference_phase"
        else: 
            self.IntParamArr2ParamDictConversion = self.IntParamArr2ParamDict
            self.IntParamDict2ParamArrConversion = self.IntParamDict2ParamArr
            self.phase_name = "coalescence_phase"

    def HM_calculate_Fstat(self, intrinsic_parameters, return_a=False, return_recovered_wave=False):

        Nevents = len(np.atleast_1d(intrinsic_parameters["chirp_mass"]))

        full_parameters1 = copy.deepcopy(intrinsic_parameters)
        full_parameters1["luminosity_distance"] = np.ones(Nevents)
        full_parameters1[self.phase_name] = np.zeros(Nevents)
        full_parameters1["inclination"] = np.ones(Nevents) * PI / 2. 
        full_parameters1["psi"] = np.zeros(Nevents)

        temp1 = self.response_generator.Response(
            parameters=full_parameters1,
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (Nchannels, Nmodes, Nevents, Nfreqs) or (Nchannels, Nmodes, Nfreqs) if Nevents == 1

        full_parameters2 = copy.deepcopy(full_parameters1)
        full_parameters2["psi"] = np.ones(Nevents) * PI / 4. 

        temp2 = self.response_generator.Response(
            parameters=full_parameters2,
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (Nchannels, Nmodes, Nevents, Nfreqs) or (Nchannels, Nmodes, Nfreqs) if Nevents == 1

        if Nevents == 1:
            temp1 = temp1[:, :, self.NX, :] # (Nchannels, Nmodes, 1, Nfreqs)
            temp2 = temp2[:, :, self.NX, :]
        # print("template shape:", temp1.shape, temp2.shape)

        X1 = self.TRANS(temp1, axes=(2, 0, 1, 3)) # (Nevents, Nchannels, Nmodes, Nfreqs)
        X1 *= 1. / self.mode_factors[:, self.NX] # (Nevents, Nchannels, Nmodes, Nfreqs) / (Nmodes, 1) = (Nevents, Nchannels, Nmodes, Nfreqs)
        X2 = 1.j * X1
        X3 = self.TRANS(temp2, axes=(2, 0, 1, 3)) 
        X3 *= 1. / self.mode_factors[:, self.NX]
        X4 = 1.j * X3

        Xvector = self.TRANS(self.xp.array([X1, X2, X3, X4]), axes=(1, 0, 3, 2, 4)) # (4, Nevents, Nchannels, Nmodes, Nfreqs) -> (Nevents, 4, Nmodes, Nchannels, Nfreqs)
        Xvector = Xvector.reshape(Nevents, 4*self.Nmodes, self.Nchannels, self.Nfreqs) # (Nevents, 4*Nmodes, Nchannels, Nfreqs)
        # print("X vector shape:", Xvector.shape)

        data_expand = self.data[self.NX, self.NX, :, :] # (1, 1, Nchannels, Nfreqs)
        Nvector = self.HM_inner_product_vectorized(data_expand, Xvector) # (Nevents, 4*Nmodes)
        Mmatrix = self.HM_inner_product_matrix(Xvector, Xvector) # (Nevents, 4*Nmodes, 4*Nmodes)
        # print("N vector shape:", Nvector.shape)
        # print("M matrix shape:", Mmatrix.shape)

        invMmatrix = self.xp.linalg.inv(Mmatrix) # (Nevents, 4, 4)
        Nvector_col = Nvector[..., self.NX] # (Nevents, 4, 1)
        NM = self.MATMUL(invMmatrix, Nvector_col) # (Nevents, 4, 1)
        Nvector_row = Nvector[:, self.NX, :] # (Nevents, 1, 4)
        NMN = self.SUM(self.MATMUL(Nvector_row, NM), axis=(1,2)) # (Nevents, 1, 1) -> (Nevents)

        res = 0.5 * NMN # (Nevents) Fstat 0.5 * N^T M^{-1} N

        if return_a:
            res_a = NM.squeeze(axis=-1) # (Nevents, 4)
            if self.use_gpu:
                return res_a.get() # (Nevents, 4)
            else: 
                return res_a # (Nevents, 4)
            
        if return_recovered_wave: 
            res_a = NM.squeeze(axis=-1) # (Nevents, 4)
            res_wf = self.SUM(res_a[:, :, self.NX, self.NX] * Xvector, axis=1) # (Nevents, Nchannels, Nfreqs)
            if Nevents == 1: 
                return res_wf[0] # (Nchannels, Nfreqs)
            else:
                return self.TRANS(res_wf, (1, 0, 2)) # (Nchannels, Nevents, Nfreqs)

        # else:
        if self.use_gpu:
            if Nevents == 1:
                return res.get()[0]
            else:
                return res.get() # (Nevents)
        else: 
            if Nevents == 1: 
                return res[0]
            else: 
                return res 

    def HM_inner_product_vectorized(self, template_channels1, template_channels2):
        """ 
            template_channels1: shape (Nevent, Nrows, Nchannel, Nfreq)
            template_channels2: shape (Nevent, Nrows, Nchannel, Nfreq)
        """
        residual1 = self.TRANS(template_channels1, (0, 1, 3, 2)) # (Nevents, Nrows, Nfreqs, Nchannels)
        residual_dagger1 = self.CONJ(residual1[:, :, :, self.NX, :]) # (Nevents, Nrows, Nfreqs, 1, Nchannels), conjugate 

        residual2 = self.TRANS(template_channels2, (0, 1, 3, 2)) # (Nevents, Nrows, Nfreqs, Nchannels)
        residual2 = residual2[:, :, :, :, self.NX] # (Nevents, Nrows, Nfreqs, Nchannels, 1)

        inners = self.SUM(self.MATMUL(self.MATMUL(residual_dagger1, self.invserse_covariance_matrix), residual2), axis=(2, 3, 4)) # (Nevents, Nrows, Nfreqs, 1, Nchannels) -> (Nevents, Nrows, Nfreqs, 1, 1) -> (Nevents, Nrows)
        return self.RE(inners)
    
    def HM_inner_product_matrix(self, template_channels1, template_channels2):
        """ 
            template_channels1: shape (Nevent, Nrows, Nchannel, Nfreq)
            template_channels2: shape (Nevent, Nrows, Nchannel, Nfreq)
        """
        residual1 = self.TRANS(template_channels1, (0, 1, 3, 2)) # (Nevents, Nrows, Nfreqs, Nchannels)
        residual_dagger1 = self.CONJ(residual1[:, :, self.NX, :, self.NX, :]) # (Nevents, Nrows, 1, Nfreqs, 1, Nchannels), conjugate 

        residual2 = self.TRANS(template_channels2, (0, 1, 3, 2)) # (Nevents, Nrows, Nfreqs, Nchannels)
        residual2 = residual2[:, self.NX, :, :, :, self.NX] # (Nevents, 1, Nrows, Nfreqs, Nchannels, 1)

        inners = self.SUM(self.MATMUL(self.MATMUL(residual_dagger1, self.invserse_covariance_matrix), residual2), axis=(3, 4, 5)) # (Nevents, Nrows, Nrows)
        return self.RE(inners)
    
    @staticmethod
    def IntParamDict2ParamArr(param_dict):
        return np.array([
            np.log10(param_dict['chirp_mass']),
            param_dict['mass_ratio'],
            param_dict['spin_1z'],
            param_dict['spin_2z'],
            param_dict['coalescence_time'],
            param_dict['longitude'],
            np.sin(param_dict['latitude']),
        ]) # (Nparams, Nevent)

    @staticmethod
    def IntParamArr2ParamDict(params):
        p = dict()
        p['chirp_mass'] = np.power(10., params[0])
        p['mass_ratio'] = params[1]
        p['spin_1z'] = params[2]
        p['spin_2z'] = params[3]
        p['coalescence_time'] = params[4]
        p['longitude'] = params[5]
        p['latitude'] = np.arcsin(params[6])
        return p 
    
    @staticmethod
    def IntParamDict2ParamArrFRef(param_dict):
        return np.array([
            np.log10(param_dict['chirp_mass']),
            param_dict['mass_ratio'],
            param_dict['spin_1z'],
            param_dict['spin_2z'],
            param_dict['reference_time'],
            param_dict['longitude'],
            np.sin(param_dict['latitude']),
        ]) # (Nparams, Nevent)

    @staticmethod
    def IntParamArr2ParamDictFRef(params):
        p = dict()
        p['chirp_mass'] = np.power(10., params[0])
        p['mass_ratio'] = params[1]
        p['spin_1z'] = params[2]
        p['spin_2z'] = params[3]
        p['reference_time'] = params[4]
        p['longitude'] = params[5]
        p['latitude'] = np.arcsin(params[6])
        return p

class HMFstatistics_2params(Fstatistics):
    def __init__(self, response_generator, frequency, data, invserse_covariance_matrix, response_parameters, Fref_waveform=False, use_gpu=False, verbose=0):
        super().__init__(response_generator, frequency, data, invserse_covariance_matrix, response_parameters, Fref_waveform, use_gpu, verbose) 
        self.response_kwargs = response_parameters.copy() 
        self.response_kwargs["output_by_mode"] = False 
        if self.response_kwargs.get("modes", None) is None: 
            self.modes = [(2, 2), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3)] 
        else: 
            self.modes = self.response_kwargs["modes"]
        self.Nmodes = len(self.modes)
        self.Nfreqs = len(frequency)
        if self.response_kwargs.get("drop_T", False): 
            self.Nchannels = 2 
        else: 
            self.Nchannels = 3 
        
        if Fref_waveform: 
            self.IntParamArr2ParamDictConversion = self.IntParamArr2ParamDictFRef
            self.IntParamDict2ParamArrConversion = self.IntParamDict2ParamArrFRef
        else: 
            self.IntParamArr2ParamDictConversion = self.IntParamArr2ParamDict
            self.IntParamDict2ParamArrConversion = self.IntParamDict2ParamArr

    def HM_calculate_Fstat(self, intrinsic_parameters, return_a=False, return_recovered_wave=False):
        """  
            intrinsic parameters: mbhb parameters except for luminosity distance and polarization angle 
        """

        Nevents = len(np.atleast_1d(intrinsic_parameters["chirp_mass"]))

        full_parameters1 = copy.deepcopy(intrinsic_parameters)
        full_parameters1["luminosity_distance"] = np.ones(Nevents)
        full_parameters1["psi"] = np.zeros(Nevents)

        temp1 = self.response_generator.Response(
            parameters=full_parameters1,
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (Nchannels, Nevents, Nfreqs) or (Nchannels, Nfreqs) if Nevents == 1

        full_parameters2 = copy.deepcopy(full_parameters1)
        full_parameters2["psi"] = np.ones(Nevents) * PI / 4. 

        temp2 = self.response_generator.Response(
            parameters=full_parameters2,
            freqs=self.frequency,
            **self.response_kwargs,
        ) # (Nchannels, Nevents, Nfreqs) or (Nchannels, Nfreqs) if Nevents == 1

        if Nevents == 1:
            temp1 = temp1[:, self.NX, :] # (Nchannels, 1, Nfreqs)
            temp2 = temp2[:, self.NX, :]
        # print("template shape:", temp1.shape, temp2.shape)

        X1 = self.TRANS(temp1, axes=(1, 0, 2)) # (Nevents, Nchannels, Nfreqs)
        X2 = self.TRANS(temp2, axes=(1, 0, 2)) # (Nevents, Nchannels, Nfreqs)
        
        data_expand = self.data[self.NX, :, :] # (1, Nchannels, Nfreqs)
        
        Nvector = self.TRANS(self.xp.array([
            self.inner_product_vectorized(data_expand, X1), 
            self.inner_product_vectorized(data_expand, X2), 
        ])) # (Nevents, 2)
        # print("shape of N vector:", Nvector.shape) 
        
        M12 = self.inner_product_vectorized(X1, X2) # (Nevent), real numbers 
        Mmatrix = self.TRANS(self.xp.array([
            [self.self_inner_product_vectorized(X1), M12], 
            [M12, self.self_inner_product_vectorized(X2)], 
        ]), axes=(2, 0, 1)) # (Nevents, 2, 2) 
        # print("shape of M matrix:", Mmatrix.shape) 

        invMmatrix = self.xp.linalg.inv(Mmatrix) # (Nevents, 2, 2)
        Nvector_col = Nvector[..., self.NX] # (Nevents, 2, 1)
        NM = self.MATMUL(invMmatrix, Nvector_col) # (Nevents, 2, 1)
        Nvector_row = Nvector[:, self.NX, :] # (Nevents, 1, 2)
        NMN = self.SUM(self.MATMUL(Nvector_row, NM), axis=(1,2)) # (Nevents, 1, 1) -> (Nevents)

        res = 0.5 * NMN # (Nevents) Fstat 0.5 * N^T M^{-1} N

        if return_a:
            res_a = NM.squeeze(axis=-1) # (Nevents, 2)
            if self.use_gpu:
                return res_a.get() # (Nevents, 2)
            else: 
                return res_a # (Nevents, 2)
            
        if return_recovered_wave: 
            res_a = NM.squeeze(axis=-1) # (Nevents, 2)
            Xvector = self.TRANS(self.xp.array([X1, X2]), axes=(1, 0, 2, 3)) # (Nevents, 2, Nchannels, Nfreqs)
            res_wf = self.SUM(res_a[:, :, self.NX, self.NX] * Xvector, axis=1) # (Nevents, Nchannels, Nfreqs)
            if Nevents == 1: 
                return res_wf[0] # (Nchannels, Nfreqs)
            else:
                return self.TRANS(res_wf, (1, 0, 2)) # (Nchannels, Nevents, Nfreqs)

        # else:
        if self.use_gpu:
            if Nevents == 1:
                return res.get()[0]
            else:
                return res.get() # (Nevents)
        else: 
            if Nevents == 1: 
                return res[0]
            else: 
                return res 
            
    @staticmethod
    def IntParamDict2ParamArr(param_dict):
        return np.array([
            np.log10(param_dict['chirp_mass']), # 0
            param_dict['mass_ratio'], # 1
            param_dict['spin_1z'], # 2
            param_dict['spin_2z'], # 3
            param_dict['coalescence_time'], # 4
            param_dict['coalescence_phase'], # 5
            np.cos(param_dict['inclination']), # 6
            param_dict['longitude'], # 7
            np.sin(param_dict['latitude']), # 8
        ]) # (Nparams, Nevent)

    @staticmethod
    def IntParamArr2ParamDict(params):
        p = dict()
        p['chirp_mass'] = np.power(10., params[0])
        p['mass_ratio'] = params[1]
        p['spin_1z'] = params[2]
        p['spin_2z'] = params[3]
        p['coalescence_time'] = params[4]
        p['coalescence_phase'] = params[5]
        p['inclination'] = np.arccos(params[6])
        p['longitude'] = params[7]
        p['latitude'] = np.arcsin(params[8])
        return p
    
    @staticmethod
    def IntParamDict2ParamArrFRef(param_dict):
        return np.array([
            np.log10(param_dict['chirp_mass']), # 0
            param_dict['mass_ratio'], # 1
            param_dict['spin_1z'], # 2
            param_dict['spin_2z'], # 3
            param_dict['reference_time'], # 4
            param_dict['reference_phase'], # 5
            np.cos(param_dict['inclination']), # 6
            param_dict['longitude'], # 7
            np.sin(param_dict['latitude']), # 8
        ]) # (Nparams, Nevent)

    @staticmethod
    def IntParamArr2ParamDictFRef(params):
        p = dict()
        p['chirp_mass'] = np.power(10., params[0])
        p['mass_ratio'] = params[1]
        p['spin_1z'] = params[2]
        p['spin_2z'] = params[3]
        p['reference_time'] = params[4]
        p['reference_phase'] = params[5]
        p['inclination'] = np.arccos(params[6])
        p['longitude'] = params[7]
        p['latitude'] = np.arcsin(params[8])
        return p
    
    @staticmethod
    def a_to_extrinsic(a):
        """   
            a: numpy array of shape (Nevents, 2)
        """
        Nevents = len(a)
        extrinsic_parameters = dict()
        extrinsic_parameters["luminosity_distance"] = 1. / np.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2) # (Nevents)
        # extrinsic_parameters["psi"] = 0.5 * np.arctan(a[:, 1] / a[:, 0]) # (Nevents)
        extrinsic_parameters["psi"] = 0.5 * np.arctan2(a[:, 1], a[:, 0]) # (Nevents)
        if Nevents == 1: 
            for k, v in extrinsic_parameters.items(): 
                extrinsic_parameters[k] = v[0]
        return extrinsic_parameters

    

def DetectorBasisInSSB(orbit_time_SI, orbit):
    n21 = orbit.ArmVectorfunctions()["21"](orbit_time_SI)
    n31 = orbit.ArmVectorfunctions()["31"](orbit_time_SI)
    z_det = np.cross(n31, n21)
    z_norm = np.linalg.norm(z_det)
    z_det = z_det / z_norm
    x_det = n31 
    y_det = np.cross(z_det, x_det)
    y_norm = np.linalg.norm(y_det)
    y_det = y_det / y_norm
    return x_det, y_det, z_det 

def DetectorSSBRotationMatrices(orbit_time_SI, orbit):
    x_det, y_det, z_det = DetectorBasisInSSB(orbit_time_SI, orbit)
    R_Ssb2Det = np.array([x_det, y_det, z_det])
    R_Det2Ssb = R_Ssb2Det.T 
    return R_Ssb2Det, R_Det2Ssb 

def SSBPosToDetectorFrame(lon_ssb, lat_ssb, psi_ssb, orbit_time_SI, orbit):
    R_Ssb2Det, _ = DetectorSSBRotationMatrices(orbit_time_SI, orbit)
    x_in_ssb = np.cos(lat_ssb) * np.cos(lon_ssb)
    y_in_ssb = np.cos(lat_ssb) * np.sin(lon_ssb)
    z_in_ssb = np.sin(lat_ssb)
    x_in_det, y_in_det, z_in_det = np.matmul(R_Ssb2Det, np.array([x_in_ssb, y_in_ssb, z_in_ssb]))
    lon_det = np.arctan2(y_in_det, x_in_det)
    lat_det = np.arcsin(z_in_det)
    
    n12_ssb = orbit.ArmVectorfunctions()["12"](orbit_time_SI)
    u_ssb = np.array([np.sin(lon_ssb), -np.cos(lon_ssb), 0.])
    v_ssb = np.array([-np.sin(lat_ssb)*np.cos(lon_ssb), -np.sin(lat_ssb)*np.sin(lon_ssb), np.cos(lat_ssb)])
    n12u = np.dot(n12_ssb, u_ssb) 
    n12v = np.dot(n12_ssb, v_ssb)
    xi_plus_12_ssb = n12u ** 2 - n12v ** 2 
    xi_cross_12_ssb = 2. * n12u * n12v
    xi_12_ssb = xi_plus_12_ssb + 1.j * xi_cross_12_ssb
    
    xi_plus_12_det = 0.5 * np.cos(lat_det) ** 2 + 0.5 * np.cos(2. * lon_det + PI / 3.) * (1. + np.sin(lat_det) ** 2)
    xi_cross_12_det = np.sin(2. * lon_det + PI / 3.) * np.sin(lat_det)
    xi_12_det = xi_plus_12_det + 1.j * xi_cross_12_det
    
    zeta_12 = xi_12_ssb * np.exp(-2.j * psi_ssb)
    # psi_det = np.arccos(np.real(zeta_12 / xi_12_det)) / 2. # cos(2psi) -> psi in [0, PI]
    c2p = np.real(zeta_12 / xi_12_det)
    s2p = -np.imag(zeta_12 / xi_12_det)
    psi_det = np.arctan2(s2p, c2p) / 2. # (-PI, PI) -> (-PI/2, PI/2)
    if psi_det < 0:
        psi_det += PI 
        
    return lon_det, lat_det, psi_det

def DetectorPosToSSBFrame(lon_det, lat_det, psi_det, orbit_time_SI, orbit):
    _, R_Det2Ssb = DetectorSSBRotationMatrices(orbit_time_SI, orbit)
    x_in_det = np.cos(lat_det) * np.cos(lon_det)
    y_in_det = np.cos(lat_det) * np.sin(lon_det)
    z_in_det = np.sin(lat_det)
    x_in_ssb, y_in_ssb, z_in_ssb = np.matmul(R_Det2Ssb, np.array([x_in_det, y_in_det, z_in_det]))
    lon_ssb = np.arctan2(y_in_ssb, x_in_ssb) % TWOPI
    lat_ssb = np.arcsin(z_in_ssb)
    
    n12_ssb = orbit.ArmVectorfunctions()["12"](orbit_time_SI)
    u_ssb = np.array([np.sin(lon_ssb), -np.cos(lon_ssb), 0.])
    v_ssb = np.array([-np.sin(lat_ssb)*np.cos(lon_ssb), -np.sin(lat_ssb)*np.sin(lon_ssb), np.cos(lat_ssb)])
    n12u = np.dot(n12_ssb, u_ssb) 
    n12v = np.dot(n12_ssb, v_ssb)
    xi_plus_12_ssb = n12u ** 2 - n12v ** 2 
    xi_cross_12_ssb = 2. * n12u * n12v
    xi_12_ssb = xi_plus_12_ssb + 1.j * xi_cross_12_ssb
    
    xi_plus_12_det = 0.5 * np.cos(lat_det) ** 2 + 0.5 * np.cos(2. * lon_det + PI / 3.) * (1. + np.sin(lat_det) ** 2)
    xi_cross_12_det = np.sin(2. * lon_det + PI / 3.) * np.sin(lat_det)
    xi_12_det = xi_plus_12_det + 1.j * xi_cross_12_det
    
    zeta_12 = xi_12_det * np.exp(-2.j * psi_det)
    # psi_ssb = np.arccos(np.real(zeta_12 / xi_12_ssb)) / 2. # cos(2psi) -> psi in [0, PI]
    c2p = np.real(zeta_12 / xi_12_ssb)
    s2p = -np.imag(zeta_12 / xi_12_ssb)
    psi_ssb = np.arctan2(s2p, c2p) / 2. # (-PI, PI) -> (-PI/2, PI/2)
    if psi_ssb < 0:
        psi_ssb += PI 
        
    return lon_ssb, lat_ssb, psi_ssb

def get_reflected_parameter_dict(searched_params, orbit):
    lon_ssb = searched_params["longitude"]
    lat_ssb = searched_params["latitude"]
    psi_ssb = searched_params["psi"]
    lon_det, lat_det, psi_det = SSBPosToDetectorFrame(lon_ssb, lat_ssb, psi_ssb, searched_params["coalescence_time"]*DAY, orbit)
    lat_det = -lat_det  # reflect latitude
    psi_det = PI - psi_det  # reflect psi
    searched_ref_params = copy.deepcopy(searched_params)
    searched_ref_params["longitude"], searched_ref_params["latitude"], searched_ref_params["psi"] = DetectorPosToSSBFrame(lon_det, lat_det, psi_det, searched_params["coalescence_time"]*DAY, orbit)
    searched_ref_params["inclination"] = PI - searched_params["inclination"] # reflect inclination 
    return searched_ref_params

def get_reflected_parameter_dict_Fref(searched_params, orbit, tc=None):
    lon_ssb = searched_params["longitude"]
    lat_ssb = searched_params["latitude"]
    psi_ssb = searched_params["psi"]
    if tc is None:
        lon_det, lat_det, psi_det = SSBPosToDetectorFrame(lon_ssb, lat_ssb, psi_ssb, searched_params["reference_time"]*DAY, orbit)
    else: 
        lon_det, lat_det, psi_det = SSBPosToDetectorFrame(lon_ssb, lat_ssb, psi_ssb, tc*DAY, orbit)
    lat_det = -lat_det  # reflect latitude
    psi_det = PI - psi_det  # reflect psi
    searched_ref_params = copy.deepcopy(searched_params)
    if tc is None: 
        searched_ref_params["longitude"], searched_ref_params["latitude"], searched_ref_params["psi"] = DetectorPosToSSBFrame(lon_det, lat_det, psi_det, searched_params["reference_time"]*DAY, orbit)
    else: 
        searched_ref_params["longitude"], searched_ref_params["latitude"], searched_ref_params["psi"] = DetectorPosToSSBFrame(lon_det, lat_det, psi_det, tc*DAY, orbit)
    searched_ref_params["inclination"] = PI - searched_params["inclination"] # reflect inclination 
    return searched_ref_params

def get_reflected_parameters(original_lon, original_lat, original_psi, original_inc, orbit_time_SI, orbit):
    lon_det, lat_det, psi_det = SSBPosToDetectorFrame(original_lon, original_lat, original_psi, orbit_time_SI, orbit)
    lat_det = -lat_det  # reflect latitude
    psi_det = PI - psi_det  # reflect psi
    reflected_lon, reflected_lat, reflected_psi = DetectorPosToSSBFrame(lon_det, lat_det, psi_det, orbit_time_SI, orbit)
    reflected_inc = PI - original_inc
    return reflected_lon, reflected_lat, reflected_psi, reflected_inc





# for Space-Based GW Detector Network 

def network_covariance_inner_product(h1, h2, inv_cov):
    """Compute h1† C^{-1} h2 for joint network data vectors.

    This is the network generalisation of
    `FrequencyDomainCovarianceInnerProduct` in Utils.py.

    Parameters
    ----------
    h1, h2 : (Nch, Nf) complex arrays
        Joint data vectors across all detectors/channels.
    inv_cov : (Nf, Nch, Nch) complex array
        Inverse network covariance per frequency.

    Returns
    -------
    inner : complex scalar
        Σ_f h1(f)† C^{-1}(f) h2(f)
    """
    d1 = h1.T[:, np.newaxis, :]          # (Nf, 1, Nch)
    d2 = h2.T[:, :, np.newaxis]          # (Nf, Nch, 1)
    tmp = np.matmul(np.conjugate(d1), inv_cov)  # (Nf, 1, Nch)
    result = np.matmul(tmp, d2)           # (Nf, 1, 1)
    return np.sum(result)


def network_covariance_snr(h, inv_cov):
    """Compute sqrt(h† C^{-1} h) for joint network data.

    Parameters
    ----------
    h : (Nch, Nf) complex array
    inv_cov : (Nf, Nch, Nch) complex array

    Returns
    -------
    snr : real scalar
    """
    return np.sqrt(np.real(network_covariance_inner_product(h, h, inv_cov)))


class NetworkHMLikelihood(HMLikelihood):
    """Heterodyned likelihood for a network of space-based GW detectors.

    Uses a single :class:`FDTDIResponseGeneratorFRefNetwork` that
    internally handles all detectors and returns a stacked joint
    response of shape ``(Ndet*Nchan, Nmode, Nf)``.

    The joint data vector at each frequency is:

        d(f) = [det0_A, det0_E, det1_A, det1_E, ..., detN_A, detN_E]ᵀ

    with a (2N × 2N) covariance matrix C(f).

    Parameters
    ----------
    response_generator : FDTDIResponseGeneratorFRefNetwork
        Single network response generator that stacks all detectors.
    frequency : (Nf,) array
        Frequency grid shared by all detectors.
    data : (Ndet*Nchan, Nf) complex array
        Stacked frequency-domain data.
    inverse_covariance_matrix : (Nf, Ndet*Nchan, Ndet*Nchan) complex array
        Pre-computed C^{-1}(f) at each frequency.
    response_kwargs : dict
        Keyword arguments forwarded to ``response_generator.Response()``.
        **Copied** on input — the caller's dict is not mutated.
    Fref_waveform : bool
        Whether the waveform uses reference_time / reference_phase.
    use_gpu : bool
        Use CuPy for GPU acceleration.
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        response_generator,
        frequency,
        data,
        inverse_covariance_matrix,
        response_kwargs,
        Fref_waveform=False,
        use_gpu=False,
        verbose=0,
    ):
        # Copy kwargs so the parent's mutation of output_by_mode
        # does not affect the caller's dict.
        kwargs = response_kwargs.copy()

        super().__init__(
            response_generator,
            frequency,
            data,
            inverse_covariance_matrix,
            kwargs,
            Fref_waveform=Fref_waveform,
            use_gpu=use_gpu,
            verbose=verbose,
        )

        if verbose > 0:
            n_chan = data.shape[0]
            print(f"NetworkHMLikelihood: {n_chan} joint channels")

    # ============================================================
    # Full (non-heterodyned) log-likelihood — mode-summed network
    # ============================================================

    def full_log_like(self, parameter_array):
        """Full log-likelihood for a single event (network, mode-summed).

        log L = -½ (d - h)† C⁻¹ (d - h)

        where h = Σₘ hₘ is the sum over waveform modes.
        Handles both (Nch, Nmodes, Nf) and (Nch, Nf) responses.
        """
        params = self.ParamArr2ParamDict(parameter_array)
        template_raw = self.response_generator.Response(
            parameters=params,
            freqs=self.frequency,
            **self.response_kwargs,
        )

        # Handle both per-mode (Nch, Nmodes, Nf) and mode-summed (Nch, Nf)
        if template_raw.ndim == 3:
            template = self.xp.sum(template_raw, axis=1)  # sum over modes → (Nch, Nf)
        else:
            template = template_raw                        # already (Nch, Nf)

        residual = self.data - template

        # residual† C⁻¹ residual  (vectorised over frequencies)
        r = self.xp.transpose(residual)                     # (Nf, Nch)
        rd = self.xp.conjugate(r)[:, self.xp.newaxis, :]    # (Nf, 1, Nch)
        rv = r[:, :, self.xp.newaxis]                       # (Nf, Nch, 1)
        tmp = self.xp.matmul(rd, self.invserse_covariance_matrix)  # (Nf, 1, Nch)
        snr2 = self.xp.real(self.xp.sum(self.xp.matmul(tmp, rv)))

        loglike = -0.5 * snr2
        if self.use_gpu:
            return loglike.get()
        return loglike

    def full_log_like_vectorized(self, parameter_array):
        """Full log-likelihood for multiple events (network, mode-summed).

        Parameters
        ----------
        parameter_array : (Nparams, Nevents) array

        Returns
        -------
        loglikes : (Nevents,) array
        """
        params = self.ParamArr2ParamDict(parameter_array)
        template_raw = self.response_generator.Response(
            parameters=params,
            freqs=self.frequency,
            **self.response_kwargs,
        )

        # Handle both (Nch, Nmodes, Nevents, Nf) and (Nch, Nevents, Nf)
        if template_raw.ndim == 4:
            template = self.xp.sum(template_raw, axis=1)  # (Nch, Nevents, Nf)
        else:
            template = template_raw                        # (Nch, Nevents, Nf)

        # residual: broadcast data (Nch, Nf) against template (Nch, Nev, Nf)
        # → (Nevents, Nch, Nf) → (Nevents, Nf, Nch)
        residual = self.xp.transpose(
            self.data[self.xp.newaxis, :, :]
            - self.xp.transpose(template, (1, 0, 2)),
            (0, 2, 1),
        )  # (Nevents, Nf, Nch)

        inv_cov = self.invserse_covariance_matrix  # (Nf, Nch, Nch)

        # r† C⁻¹ r per event, per frequency, then sum
        rd = self.xp.conjugate(residual[:, :, self.xp.newaxis, :])  # (Nev, Nf, 1, Nch)
        rv = residual[:, :, :, self.xp.newaxis]                     # (Nev, Nf, Nch, 1)
        tmp = self.xp.matmul(rd, inv_cov)                            # (Nev, Nf, 1, Nch)
        snr2 = self.xp.sum(self.xp.matmul(tmp, rv), axis=(1, 2, 3)) # (Nevents,)
        loglikes = -0.5 * self.xp.real(snr2)

        if self.use_gpu:
            return loglikes.get()
        return loglikes
        
        
        