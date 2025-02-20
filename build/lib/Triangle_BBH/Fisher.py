import numpy as np 
from Triangle.FFTTools import *
from Triangle_BBH.Utils import *
from Triangle.Noise import * 


class Fisher():
    def __init__(self, waveform_generator, param_dict, analyze_param_step_dict, 
                 frequency, psd=None, verbose=0):
        """
        waveform_generator should be a function: param_dict, frequency -> numpy array of TDI channels of shape (Nchannel, Nfreq)
        param_dict = {"param name": param value}
        analyze_param_step_dict = {"param name": param step} (keys of the latter dict should be a subset of the former)
        frequency is a 1D array of shape (Nfreq,)
        psd should be of the same shape as generated Frequency Domain waveform
        waveform data and derivatives are numpy arrays (N_channel, N_sampling_point)
        NOTE: set undetermined steps to nagative values whose absolute values should be interpreted as the initial step and run auto_test_steps 
        NOTE: the indices of Fisher matrix follows the order of analyze_param_step_dict
        """
        
        self.waveform_generator = waveform_generator

        # parameters
        self.param_dict = param_dict.copy()
        self.analyze_param_step_dict = analyze_param_step_dict.copy()
        self.num_params = len(param_dict.keys())
        self.num_analyze_params = len(analyze_param_step_dict.keys())
        self.param_names = list(param_dict.keys())
        self.analyze_param_names = list(analyze_param_step_dict.keys())

        # Fisher matrix
        self.Fisher = np.zeros((self.num_analyze_params, self.num_analyze_params))
        self.waveform_fiducial = self.waveform_generator(param_dict, frequency)
        self.num_channels = len(self.waveform_fiducial)
        
        # PSD and FFT
        self.PSD = psd
        if psd.shape != self.waveform_fiducial.shape:
            raise ValueError('the shape of PSD and waveform mismatch.')
        self.frequency = frequency
        self.df = self.frequency[1] - self.frequency[0]

        self.verbose = verbose
        if self.verbose > 0:
            print(self.num_params, 'parameters in total:', self.param_names)
            print(self.num_analyze_params, 'analyzed parameters:', self.analyze_param_names)
            test_params = []
            for k, v in self.analyze_param_step_dict.items():
                if v < 0:
                    test_params.append(k)
            print('parameters whose steps should be tested:', test_params)

    def get_derivative(self, param_name, param_shift):
        if not (param_name in self.analyze_param_names):
            raise ValueError(param_name + " is not in analyze parameters.")
      
        param_dict_shifted = self.param_dict.copy()
        param_dict_shifted[param_name] += param_shift
        waveform_shifted = self.waveform_generator(param_dict_shifted, self.frequency)
        # shape: (Nchannel, Nfreqs)
        return (waveform_shifted - self.waveform_fiducial) / param_shift
    
    def test_step(self, param_name, init_shift, factor, threshold):
        """
        shift = shift / factor for each iteration
        iteration ends when relative difference of Fisher element is less than the threshold
        """
        if not (param_name in self.analyze_param_names):
            raise ValueError(param_name + " is not in analyze parameters.")
        
        rel_deff = 1.
        shift = init_shift
        derivative1 = self.get_derivative(param_name=param_name, param_shift=shift)
        Fisher_element1 = 0
        for i in range(self.num_channels):
            # Fisher_element1 += (FrequencyDomainSNR(h=derivative1[i], psd=self.PSD[i], df=self.df)) ** 2
            Fisher_element1 += FrequencyDomainInnterProduct(h1=derivative1[i], h2=derivative1[i], psd=self.PSD[i], df=self.df)
        error1 = 1. / np.sqrt(Fisher_element1)

        num_iterate = 0
        while(rel_deff > threshold):
            num_iterate += 1
            if num_iterate >= 50:
                raise ValueError("Not converged after 50 iterations.")
            print('Iteration No.', num_iterate)
            shift = shift / factor
            derivative2 = self.get_derivative(param_name=param_name, param_shift=shift)
            Fisher_element2 = 0
            for i in range(self.num_channels):
                # Fisher_element2 += (FrequencyDomainSNR(h=derivative2[i], psd=self.PSD[i], df=self.df)) ** 2
                Fisher_element2 += FrequencyDomainInnterProduct(h1=derivative2[i], h2=derivative2[i], psd=self.PSD[i], df=self.df)
            error2 = 1. / np.sqrt(Fisher_element2)
            print('errors =', error1, error2)

            rel_deff = np.abs((error2 - error1) / error1)
            error1 = error2
            print('shift =', shift, 'relative diff =', rel_deff)
    
        self.analyze_param_step_dict[param_name] = shift
        print('shift of parameter', param_name, 'is', shift)

    def auto_test_step(self, factor, threshold=1e-4):
        """ 
        shift = shift / factor for each iteration
        """
        for k, v in self.analyze_param_step_dict.items():
            if v < 0:
                print('\n ========= testing parameter', k, '==========')
                self.test_step(param_name=k, init_shift=np.abs(v), factor=factor, threshold=threshold)
        print('all parameters tested.')

    def calculate_Fisher(self):
        """
        the index of Fisher matrix is in coincidence with that of analyze parameter
        """
        for k, v in self.analyze_param_step_dict.items():
            if v <= 0:
                raise ValueError("steps should be positive")
            
        self.param_derivatives = {}
        for k, v in self.analyze_param_step_dict.items():
            self.param_derivatives[k] = self.get_derivative(param_name=k, param_shift=v)

        for i1 in range(self.num_analyze_params):
            name1 = self.analyze_param_names[i1]
            for i2 in range(i1, self.num_analyze_params):
                name2 = self.analyze_param_names[i2]
                for i3 in range(self.num_channels):
                    self.Fisher[i1][i2] += FrequencyDomainInnterProduct(
                        h1=self.param_derivatives[name1][i3], 
                        h2=self.param_derivatives[name2][i3],
                        psd=self.PSD[i3],
                        df=self.df
                        )
        self.Fisher += self.Fisher.T - np.diag(self.Fisher.diagonal())
        if self.verbose > 0:
            print('index of Fisher:', self.analyze_param_names)
        # return self.Fisher


    def calculate_errors(self):
        self.param_errors = {}
        Covariance = np.linalg.inv(self.Fisher)
        errors = np.sqrt(np.diagonal(Covariance))
        for i in range(self.num_analyze_params):
            name = self.analyze_param_names[i]
            self.param_errors[name] = errors[i]
        self.errors = errors
        self.CovMatrix = Covariance
        if self.verbose > 0:
            print('errors:')
            print(self.param_errors)
        # return self.param_errors




class MultiChannelFisher():
    def __init__(self, waveform_generator, param_dict, analyze_param_step_dict, frequency, inverse_covariance, verbose=0):
        """
        Args: 
            waveform_generator should be a function: param_dict, frequency -> numpy array of TDI channels of shape (Nchannel, Nfreq)
            param_dict = {"param name": param value}
            analyze_param_step_dict = {"param name": param step} (keys of the latter dict should be a subset of the former)
            frequency is a 1D array of shape (Nfreq,)
            inverse_covariance should be a 3D numpy array of shape (Nfreq, 3, 3)

        waveform data and derivatives are numpy arrays (Nchannel, Nfreq)
        NOTE: set undetermined steps to nagative values whose absolute values should be interpreted as the initial step and run auto_test_steps 
        NOTE: the indices of Fisher matrix follows the order of analyze_param_step_dict
        """
        
        self.waveform_generator = waveform_generator

        # parameters
        self.param_dict = param_dict.copy()
        self.analyze_param_step_dict = analyze_param_step_dict.copy()
        self.num_params = len(param_dict.keys())
        self.num_analyze_params = len(analyze_param_step_dict.keys())
        self.param_names = list(param_dict.keys())
        self.analyze_param_names = list(analyze_param_step_dict.keys())

        # Fisher matrix
        self.Fisher = np.zeros((self.num_analyze_params, self.num_analyze_params))
        self.waveform_fiducial = self.waveform_generator(param_dict, frequency)
        self.num_channels = len(self.waveform_fiducial)
        
        # PSD and FFT
        self.frequency = frequency
        self.df = self.frequency[1] - self.frequency[0]
        self.invcov = inverse_covariance
        self.psd = np.array([self.invcov[i][i] for i in range(self.invcov.shape[1])]) * 4. * self.df 
        if inverse_covariance.shape[0] != self.waveform_fiducial.shape[1]:
            raise ValueError('the shape of CSD and waveform mismatch.')


        self.verbose = verbose
        if self.verbose > 0:
            print(self.num_params, 'parameters in total:', self.param_names)
            print(self.num_analyze_params, 'analyzed parameters:', self.analyze_param_names)
            test_params = []
            for k, v in self.analyze_param_step_dict.items():
                if v < 0:
                    test_params.append(k)
            print('parameters whose steps should be tested:', test_params)

    def get_derivative(self, param_name, param_shift):
        if not (param_name in self.analyze_param_names):
            raise ValueError(param_name + " is not in analyze parameters.")
      
        param_dict_shifted = self.param_dict.copy()
        param_dict_shifted[param_name] += param_shift
        waveform_shifted = self.waveform_generator(param_dict_shifted, self.frequency)
        # shape: (Nchannel, Nfreqs)
        return (waveform_shifted - self.waveform_fiducial) / param_shift
    
    def test_step(self, param_name, init_shift, factor=2., threshold=1e-4, iteration_threshold=50):
        """
        shift = shift / factor for each iteration
        iteration ends when relative difference of Fisher element is less than the threshold
        """
        if not (param_name in self.analyze_param_names):
            raise ValueError(param_name + " is not in analyze parameters.")
        
        shift = init_shift
        derivative1 = self.get_derivative(param_name=param_name, param_shift=shift)
        Fisher_element1 = FrequencyDomainCovarianceSNR(data_channels=derivative1, inv_cov=self.invcov)
        error_record = [1. / np.sqrt(Fisher_element1), ]

        num_iterate = 0
        rel_deff = 1.
        while(rel_deff > threshold):
            num_iterate += 1
            if num_iterate >= iteration_threshold:
                raise ValueError("Not converged after 50 iterations.")
            if (num_iterate % 10 == 0):
                print('Iteration No.', num_iterate)
            shift = shift / factor
            derivative2 = self.get_derivative(param_name=param_name, param_shift=shift)
            Fisher_element2 = FrequencyDomainCovarianceSNR(data_channels=derivative2, inv_cov=self.invcov)
            error_record.append(1. / np.sqrt(Fisher_element2))
            print('error =', error_record[-1])
            
            if num_iterate == 1:
                rel_deff = np.sqrt(np.var([error_record[-2], error_record[-1]])) / np.abs(error_record[-1])
            else:
                rel_deff = np.sqrt(np.var([error_record[-3], error_record[-2], error_record[-1]])) / np.abs(error_record[-1])
            print('shift =', shift, 'relative diff =', rel_deff)
    
        self.analyze_param_step_dict[param_name] = shift
        print('shift of parameter', param_name, 'is', shift)

    def auto_test_step(self, factor=2., threshold=1e-4, iteration_threshold=50):
        """ 
        shift = shift / factor for each iteration
        """
        for k, v in self.analyze_param_step_dict.items():
            if v < 0:
                print('\n ========= testing parameter', k, '==========')
                self.test_step(param_name=k, init_shift=np.abs(v), factor=factor, threshold=threshold, iteration_threshold=iteration_threshold)
        print('all parameters tested.')

    def calculate_Fisher(self):
        """
        the index of Fisher matrix is in coincidence with that of analyze parameter
        """
        for k, v in self.analyze_param_step_dict.items():
            if v <= 0:
                raise ValueError("steps should be positive, run auto_test_step first.")
            
        self.param_derivatives = {}
        for k, v in self.analyze_param_step_dict.items():
            self.param_derivatives[k] = self.get_derivative(param_name=k, param_shift=v)

        for i1 in range(self.num_analyze_params):
            name1 = self.analyze_param_names[i1]
            for i2 in range(i1, self.num_analyze_params):
                name2 = self.analyze_param_names[i2]
                self.Fisher[i1][i2] = FrequencyDomainCovarianceInnerProduct(
                    data_channels1=self.param_derivatives[name1], 
                    data_channels2=self.param_derivatives[name2], 
                    inv_cov=self.invcov
                    )
        self.Fisher += self.Fisher.T - np.diag(self.Fisher.diagonal())

    def calculate_errors(self):
        """  
            calculate_Fisher must be called before 
        """
        self.param_errors = {}
        Covariance = np.linalg.inv(self.Fisher)
        errors = np.sqrt(np.diagonal(Covariance))
        for i in range(self.num_analyze_params):
            name = self.analyze_param_names[i]
            self.param_errors[name] = errors[i]
        self.errors = errors
        self.CovMatrix = Covariance
        if self.verbose > 0:
            print('errors:')
            print(self.param_errors)

    def Cutler_Vallisneri_bias(self, true_waveform):
        """   
            calculate_errors must be called before 
        """
        residual_waveform = self.waveform_fiducial - true_waveform
        bias_vector = np.zeros(self.num_analyze_params)
        for i, v in enumerate(self.param_derivatives.values()):
            bias_vector[i] = FrequencyDomainCovarianceInnerProduct(data_channels1=v, data_channels2=residual_waveform, inv_cov=self.invcov)
        return -np.matmul(self.CovMatrix, bias_vector)
        

        

    

    