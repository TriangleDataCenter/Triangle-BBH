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

def FrequencyDomainInnterProduct(h1, h2, psd, df):
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
        data_channels1, 2: numpy arraies of shapes (3, Nf)
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
    convert parameter dict to parameter array where Mc and D are in logscalse, 
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



class Likelihood:
    # TODO: mode-by-mode heterodyne for the PhenonmHM waveform 
    def __init__(self, response_generator, frequency, data, invserse_covariance_matrix, response_parameters, use_gpu=False, verbose=0):
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
        
    def full_log_like(self, parameter_array): 
        """ 
        Args: 
            parameter_array: parameters given as an array.
            the order is: ['log_chirp_mass', 'mass_ratio', 'spin_1z', 'spin_2z', 'coalescence_time', 'coalescence_phase', 'log_luminosity_distance', 'cos_inclination', 'longitude', 'sin_latitude', 'psi']
        Returns: 
            loglike 
        """
        template = self.response_generator.Response(
            parameters=ParamArr2ParamDict(parameter_array),
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
            parameters=ParamArr2ParamDict(parameter_array),
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
            NOTE: currently we only recommand the use of base parameters rather than base waveform 
        """
        if base_waveform is None:
            self.h0 = self.response_generator.Response(
                parameters=ParamArr2ParamDict(base_parameters),
                freqs=self.frequency,
                **self.response_kwargs,
            ) # (3, Nf)
        else: 
            raise NotImplementedError("heterodyned likelihood with base waveform not implemented yet.")
        
        # create sparce grid of frequencies (1st try)
        FMIN, FMAX = self.xp.min(self.frequency) * 0.999999999999, self.xp.max(self.frequency) * 1.000000000001
        self.het_frequency = self.xp.logspace(self.xp.log10(FMIN), self.xp.log10(FMAX), num_het_frequency) # N_het_f
        
        # calculate base waveform at the sparce grid (1st try)
        if base_waveform is None:
            self.het_h0 = self.response_generator.Response(
                parameters=ParamArr2ParamDict(base_parameters),
                freqs=self.het_frequency,
                **self.response_kwargs,
            ) # (3, N_het_f)
        else: 
            raise NotImplementedError("heterodyned likelihood with base waveform not implemented yet.")
        
        # refine the sparse grid to ensure no zero waveforms 
        # valid_idx = self.xp.where(self.xp.abs(self.het_h0[0]) != 0.)[0]
        valid_idx = self.xp.where(self.xp.abs(self.het_h0[0]) > 1e-25)[0]
        tmpf = self.het_frequency[valid_idx]
        
        # create sparce grid of frequencies (final)
        self.het_frequency = self.xp.logspace(self.xp.log10(tmpf[0]), self.xp.log10(tmpf[-1]), num_het_frequency) # N_het_f
        
        # calculate base waveform at the sparce grid (final)
        if base_waveform is None: 
            self.het_h0 = self.response_generator.Response(
                parameters=ParamArr2ParamDict(base_parameters),
                freqs=self.het_frequency,
                **self.response_kwargs,
            ) # (3, N_het_f)
        else: 
            raise NotImplementedError("heterodyned likelihood with base waveform not implemented yet.")
        
        self.het_h0[self.het_h0==0.] = 1e-25 
        
        # confine the frequency and data to be within the boundaries of sparce grid 
        inband_idx = self.xp.where((self.frequency >= self.het_frequency[0]) & (self.frequency <= self.het_frequency[-1]))[0]
        self.dense_frequency = self.frequency[inband_idx] # (Nf)
        self.dense_data = self.data[:, inband_idx] # (3, Nf)
        self.dense_h0 = self.h0[:, inband_idx] # (3, Nf)
        self.dense_invserse_covariance_matrix = self.invserse_covariance_matrix[inband_idx] # (Nf, 3, 3)
        
        # group the dense frequencies with the sparce frequency grid, return the left idx of each dense frequency, each bin is labeled by this 
        group_idx = self.xp.searchsorted(self.het_frequency, self.dense_frequency, "right") - 1 # (Nf)
        dense_frequency_offset = self.dense_frequency - self.het_frequency[group_idx] # (Nf)
        
        # start pre-calculating the coefficients of heterodyned likelihood 
        # 1) h_h terms
        B0_pre = self.xp.matmul(self.xp.transpose(self.xp.conjugate(self.dense_h0))[:, :, self.xp.newaxis], self.xp.transpose(self.dense_h0)[:, self.xp.newaxis, :]) * self.dense_invserse_covariance_matrix # (Nf, 3, 1) * (Nf, 1, 3) -> (Nf, 3, 3)
        B1_pre = B0_pre * dense_frequency_offset[:, self.xp.newaxis, self.xp.newaxis] # (Nf, 3, 3)
        # 2) d_h terms 
        A0_pre = self.xp.matmul(self.xp.transpose(self.xp.conjugate(self.dense_data))[:, :, self.xp.newaxis], self.xp.transpose(self.dense_h0)[:, self.xp.newaxis, :]) * self.dense_invserse_covariance_matrix # (Nf, 3, 1) * (Nf, 1, 3) -> (Nf, 3, 3)
        A1_pre = A0_pre * dense_frequency_offset[:, self.xp.newaxis, self.xp.newaxis] # (Nf, 3, 3)
        
        # sum all the coefficients in sparce grids  
        self.Nbin = num_het_frequency - 1 
        self.B0 = self.xp.zeros((self.Nbin, 3, 3), dtype=self.xp.complex128)
        self.B1 = self.xp.zeros((self.Nbin, 3, 3), dtype=self.xp.complex128)
        self.A0 = self.xp.zeros((self.Nbin, 3, 3), dtype=self.xp.complex128)
        self.A1 = self.xp.zeros((self.Nbin, 3, 3), dtype=self.xp.complex128)
        for ibin in self.xp.unique(group_idx): # loop over the left idx of sparce grids 
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
            
        # calculate sparce template 
        het_h = self.response_generator.Response(
            parameters=ParamArr2ParamDict(parameter_array),
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

        self.output = [LL1, self.xp.real(LL2)]
        
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
            numpy array of heterodyned loglikes 
        """
        if not self.het_prepare_flag: 
            raise NotImplementedError("heterodyne not prepared, run preparation first.")
            
        # calculate sparce template 
        het_h = self.xp.transpose(self.response_generator.Response(
            parameters=ParamArr2ParamDict(parameter_array),
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

        self.output_vec = [LL1, self.xp.real(LL2)]
        
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
            data_channels1, 2: numpy arraies of shapes (3, Nf)
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
    def __init__(self, response_generator, frequency, data, invserse_covariance_matrix, response_parameters, use_gpu=False):
        super().__init__(response_generator, frequency, data, invserse_covariance_matrix, response_parameters, use_gpu)
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
    lat_det = -lat_det # reflect latitutde 
    psi_det = PI - psi_det # reflect psi 
    searched_ref_params = copy.deepcopy(searched_params)
    searched_ref_params["longitude"], searched_ref_params["latitude"], searched_ref_params["psi"] = DetectorPosToSSBFrame(lon_det, lat_det, psi_det, searched_params["coalescence_time"]*DAY, orbit)
    searched_ref_params["inclination"] = PI - searched_params["inclination"] # reflect inclination 
    return searched_ref_params

def get_reflected_parameters(original_lon, original_lat, original_psi, original_inc, orbit_time_SI, orbit):
    lon_det, lat_det, psi_det = SSBPosToDetectorFrame(original_lon, original_lat, original_psi, orbit_time_SI, orbit)
    lat_det = -lat_det # reflect latitutde 
    psi_det = PI - psi_det # reflect psi 
    reflected_lon, reflected_lat, reflected_psi = DetectorPosToSSBFrame(lon_det, lat_det, psi_det, orbit_time_SI, orbit)
    reflected_inc = PI - original_inc
    return reflected_lon, reflected_lat, reflected_psi, reflected_inc