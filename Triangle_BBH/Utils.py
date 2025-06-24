import numpy as np 
try:
    import cupy as xp
    # print("has cupy")
except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    # print("no cupy")


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
    def __init__(self, response_generator, frequency, data, invserse_covariance_matrix, response_parameters, use_gpu=False):
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
        
        print("number of freuqncies:", len(frequency))
        print("min and max frequencies:", self.xp.min(frequency), self.xp.max(frequency))
        print("response kwargs:", self.response_kwargs)
        
        self.parameter_names = ['log_chirp_mass', 'mass_ratio', 'spin_1z', 'spin_2z', 'coalescence_time', 'coalescence_phase', 'log_luminosity_distance', 'cos_inclination', 'longitude', 'sin_latitude', 'psi']
        self.channel_names = ["X2", "Y2", "Z2"]
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
        """
        if base_waveform == None:
            self.h0 = self.response_generator.Response(
                parameters=ParamArr2ParamDict(base_parameters),
                freqs=self.frequency,
                **self.response_kwargs,
            ) # (3, Nf)
        else: 
            self.h0 = base_waveform.copy() 
            if self.h0.shape != self.data.shape:
                raise ValueError("shapes of waveforms mismatch.")
        
        # create sparce grid of frequencies (1st try)
        FMIN, FMAX = self.xp.min(self.frequency) * 0.999999999999, self.xp.max(self.frequency) * 1.000000000001
        self.het_frequency = self.xp.logspace(self.xp.log10(FMIN), self.xp.log10(FMAX), num_het_frequency) # N_het_f
        
        # calculate base waveform at the sparce grid (1st try)
        self.het_h0 = self.response_generator.Response(
            parameters=ParamArr2ParamDict(base_parameters),
            freqs=self.het_frequency,
            **self.response_kwargs,
        ) # (3, N_het_f)
        
        # refine the sparse grid to ensure no zero waveforms 
        # valid_idx = self.xp.where(self.xp.abs(self.het_h0[0]) != 0.)[0]
        valid_idx = self.xp.where(self.xp.abs(self.het_h0[0]) > 1e-23)[0]
        tmpf = self.het_frequency[valid_idx]
        
        # create sparce grid of frequencies (final)
        self.het_frequency = self.xp.logspace(self.xp.log10(tmpf[0]), self.xp.log10(tmpf[-1]), num_het_frequency) # N_het_f
        
        # calculate base waveform at the sparce grid (final)
        self.het_h0 = self.response_generator.Response(
            parameters=ParamArr2ParamDict(base_parameters),
            freqs=self.het_frequency,
            **self.response_kwargs,
        ) # (3, N_het_f)
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

        # # test only 
        # self.LL1 = LL1 
        # self.LL2 = LL2 
        # self.het_h = het_h
        # self.het_r = het_r
        # self.het_r0 = het_r0
        # self.het_r1 = het_r1
        
        # test only 
        # LL1 = 0.  
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
        # print("shape of het_h", het_h.shape)
        
        # calculate heterodyne 
        het_r = self.xp.transpose(het_h / self.het_h0, (0, 2, 1)) # (Nevents, N_het_f, 3)
        het_r = self.xp.nan_to_num(het_r, 0.) # deal with the nan caused by divide 0 
        het_r0 = het_r[:, :-1, :] # (Nevents, Nb, 3)
        het_r1 = (het_r[:, 1:, :] - het_r0) / self.het_df[:, self.xp.newaxis] # (Nevents, Nb, 3)
        # print("shape of r, r0, r1", het_r.shape, het_r0.shape, het_r1.shape)
        
        # calculate likelihood 
        # 1) h_h term 
        LL1 = self.xp.sum(self.xp.matmul(self.xp.conjugate(het_r0)[:, :, :, self.xp.newaxis], het_r0[:, :, self.xp.newaxis, :]) * self.B0, axis=(1, 2, 3)) # sum((Nevents, Nb, 3, 3) * (Nb, 3, 3)) -> (Nevents)
        tmp_mat = self.xp.matmul(self.xp.conjugate(het_r0)[:, :, :, self.xp.newaxis], het_r1[:, :, self.xp.newaxis, :]) # (Nevents, Nb, 3, 3)
        LL1 += self.xp.sum((tmp_mat + self.xp.transpose((self.xp.conjugate(tmp_mat)), (0, 1, 3, 2))) * self.B1, axis=(1, 2, 3)) # sum((Nevents, Nb, 3, 3) * (Nb, 3, 3)) -> (Nevents)
        # 2_ d_h term 
        LL2 = self.xp.sum(self.xp.matmul(self.A0[self.xp.newaxis, :, :, :], het_r0[:, :, :, self.xp.newaxis]), axis=(1, 2, 3)) # (Nevents)
        LL2 += self.xp.sum(self.xp.matmul(self.A1[self.xp.newaxis, :, :, :], het_r1[:, :, :, self.xp.newaxis]), axis=(1, 2, 3)) # (Nevents)
        # print("shape of tmpmat", tmp_mat.shape)

        # test 
        # LL1 = 0. # test 
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
            