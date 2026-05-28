import numpy as np 
from Triangle.FFTTools import *
from Triangle_BBH.Utils import *
from Triangle.Noise import * 


class Fisher():
    def __init__(self, waveform_generator, param_dict, analyze_param_step_dict, 
                 frequency, psd=None, verbose=0):
        """Fisher matrix analysis with diagonal (uncorrelated) noise.

        Parameters
        ----------
        waveform_generator : callable
            Function of signature ``(param_dict, frequency) -> waveform``
            returning TDI channels of shape (Nchannel, Nfreq).
        param_dict : dict
            Dictionary of all waveform parameters.
        analyze_param_step_dict : dict
            Subset of ``param_dict`` keys with finite-difference step sizes.
            Set a step to a negative value to trigger automatic tuning via
            ``auto_test_step()``; its absolute value is used as the initial step.
        frequency : ndarray
            1D frequency array of shape (Nfreq,).  Assumed uniformly spaced.
        psd : ndarray, optional
            One-sided power spectral density for each channel,
            shape (Nchannel, Nfreq).
        verbose : int, optional
            Verbosity level.  0 = silent, > 0 = progress output.  Default 0.

        Notes
        -----
        Fisher matrix indices follow the order of ``analyze_param_step_dict``.
        """
        
        self.waveform_generator = waveform_generator

        # parameter storage
        self.param_dict = param_dict.copy()
        self.analyze_param_step_dict = analyze_param_step_dict.copy()
        self.num_params = len(param_dict.keys())
        self.num_analyze_params = len(analyze_param_step_dict.keys())
        self.param_names = list(param_dict.keys())
        self.analyze_param_names = list(analyze_param_step_dict.keys())

        # Fisher matrix and fiducial waveform
        self.Fisher = np.zeros((self.num_analyze_params, self.num_analyze_params))
        self.waveform_fiducial = self.waveform_generator(param_dict, frequency)
        self.num_channels = len(self.waveform_fiducial)
        
        # noise PSD and frequency grid
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
        """Compute the waveform derivative w.r.t. a parameter using
        second-order central finite difference.

        Parameters
        ----------
        param_name : str
            Name of the parameter to differentiate.
        param_shift : float
            Step size for the finite difference.

        Returns
        -------
        ndarray
            Derivative waveform of shape (Nchannel, Nfreq).
        """
        if not (param_name in self.analyze_param_names):
            raise ValueError(param_name + " is not in analyze parameters.")
      
        param_dict_plus = self.param_dict.copy()
        param_dict_plus[param_name] += param_shift
        waveform_plus = self.waveform_generator(param_dict_plus, self.frequency)
        
        param_dict_minus = self.param_dict.copy()
        param_dict_minus[param_name] -= param_shift
        waveform_minus = self.waveform_generator(param_dict_minus, self.frequency)
        # shape: (Nchannel, Nfreq)
        return (waveform_plus - waveform_minus) / (2. * param_shift)
    
    def test_step(self, param_name, init_shift, factor=2., threshold=1e-4, iteration_threshold=50):
        """Tune the finite-difference step size for a single parameter.

        The step is iteratively reduced by ``factor`` until the 1-sigma
        error (computed from the diagonal Fisher element) stabilizes,
        i.e., the relative fluctuation over the last few iterations
        falls below ``threshold``.

        Parameters
        ----------
        param_name : str
            Name of the parameter to tune.
        init_shift : float
            Initial step size (absolute value).
        factor : float, optional
            Reduction factor per iteration.  Default 2.
        threshold : float, optional
            Convergence threshold on relative error fluctuation.
            Default 1e-4.
        iteration_threshold : int, optional
            Maximum number of iterations.  Default 50.
        """
        if not (param_name in self.analyze_param_names):
            raise ValueError(param_name + " is not in analyze parameters.")
        
        shift = init_shift
        derivative1 = self.get_derivative(param_name=param_name, param_shift=shift)
        Fisher_element1 = 0
        for i in range(self.num_channels):
            Fisher_element1 += FrequencyDomainInnerProduct(h1=derivative1[i], h2=derivative1[i], psd=self.PSD[i], df=self.df)
        error_record = [1. / np.sqrt(Fisher_element1), ]
        if self.verbose > 0:
            print('initial 1-sigma error = {:.3e}'.format(error_record[0]))

        num_iterate = 0
        rel_deff = 1.
        while(rel_deff > threshold):
            num_iterate += 1
            if num_iterate >= iteration_threshold:
                raise ValueError("Not converged after {} iterations.".format(iteration_threshold))
            if (num_iterate % 10 == 0):
                if self.verbose > 0:
                    print('Iteration No.', num_iterate)
            shift = shift / factor
            derivative2 = self.get_derivative(param_name=param_name, param_shift=shift)
            Fisher_element2 = 0
            for i in range(self.num_channels):
                Fisher_element2 += FrequencyDomainInnerProduct(h1=derivative2[i], h2=derivative2[i], psd=self.PSD[i], df=self.df)
            error_record.append(1. / np.sqrt(Fisher_element2))
            if self.verbose > 0:
                print('error = {:.3e}'.format(error_record[-1]))
            
            if num_iterate == 1:
                rel_deff = np.sqrt(np.var([error_record[-2], error_record[-1]])) / np.abs(error_record[-1])
            else:
                rel_deff = np.sqrt(np.var([error_record[-3], error_record[-2], error_record[-1]])) / np.abs(error_record[-1])
            if self.verbose > 0:
                print('shift = {:.3e}, rel diff = {:.3e}'.format(shift, rel_deff))
    
        self.analyze_param_step_dict[param_name] = shift
        if self.verbose > 0:
            print('shift of parameter', param_name, 'is', shift)

    def auto_test_step(self, factor=2., threshold=1e-4, iteration_threshold=50):
        """Automatically tune step sizes for all parameters with negative steps.

        Calls ``test_step()`` on each parameter whose value in
        ``analyze_param_step_dict`` is negative.

        Parameters
        ----------
        factor : float, optional
            Reduction factor per iteration.  Default 2.
        threshold : float, optional
            Convergence threshold.  Default 1e-4.
        iteration_threshold : int, optional
            Maximum iterations per parameter.  Default 50.
        """
        for k, v in self.analyze_param_step_dict.items():
            if v < 0:
                if self.verbose > 0:
                    print('\n ========= testing parameter', k, '==========')
                self.test_step(param_name=k, init_shift=np.abs(v), factor=factor, threshold=threshold, iteration_threshold=iteration_threshold)
        if self.verbose > 0:
            print('all parameters tested.')

    def calculate_Fisher(self):
        """Compute the Fisher information matrix.

        Uses finite-difference derivatives evaluated at the fiducial
        parameter values.  Matrix indices follow ``analyze_param_names``
        ordering.  Must be called after all step sizes are set to positive
        values (manually or via ``auto_test_step()``).
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
                    self.Fisher[i1][i2] += FrequencyDomainInnerProduct(
                        h1=self.param_derivatives[name1][i3], 
                        h2=self.param_derivatives[name2][i3],
                        psd=self.PSD[i3],
                        df=self.df
                        )
        self.Fisher += self.Fisher.T - np.diag(self.Fisher.diagonal())
        if self.verbose > 0:
            print('index of Fisher:', self.analyze_param_names)
            print('Fisher matrix computed.')


    def calculate_errors(self):
        """Invert the Fisher matrix to obtain 1-sigma parameter uncertainties.

        Computes the covariance matrix via ``np.linalg.inv`` and extracts
        marginal errors.  Sets ``self.CovMatrix`` and ``self.errors``.
        Must be called after ``calculate_Fisher()``.
        """
        self.param_errors = {}
        cond = np.linalg.cond(self.Fisher)
        if self.verbose > 0:
            print('Fisher matrix condition number: {:.1e}'.format(cond))
        if cond > 1e15:
            print('WARNING: Fisher matrix condition number ({:.1e}) exceeds threshold 1e15, results may be unreliable.'.format(cond))
        Covariance = np.linalg.inv(self.Fisher)
        errors = np.sqrt(np.diagonal(Covariance))
        for i in range(self.num_analyze_params):
            name = self.analyze_param_names[i]
            self.param_errors[name] = errors[i]
        self.errors = errors
        self.CovMatrix = Covariance
        if self.verbose > 0:
            print('1-sigma parameter uncertainties:')
            for name in self.analyze_param_names:
                print('  {:<25s} {:.3e}'.format(name, self.param_errors[name]))




class MultiChannelFisher():
    def __init__(self, waveform_generator, param_dict, analyze_param_step_dict, frequency, inverse_covariance, verbose=0):
        """Fisher matrix analysis with correlated noise (full inverse covariance).

        Parameters
        ----------
        waveform_generator : callable
            Function of signature ``(param_dict, frequency) -> waveform``
            returning TDI channels of shape (Nchannel, Nfreq).
        param_dict : dict
            Dictionary of all waveform parameters.
        analyze_param_step_dict : dict
            Subset of ``param_dict`` keys with finite-difference step sizes.
            Set a step to a negative value to trigger automatic tuning via
            ``auto_test_step()``; its absolute value is used as the initial step.
        frequency : ndarray
            1D frequency array of shape (Nfreq,).
        inverse_covariance : ndarray
            Inverse noise covariance matrix of shape
            (Nfreq, Nchannel, Nchannel), where
            Cov_IJ = CSD_IJ / (4 * df) and CSD_IJ = 2 <I J^*> / T.
        verbose : int, optional
            Verbosity level.  0 = silent, > 0 = progress output.  Default 0.

        Notes
        -----
        Fisher matrix indices follow the order of ``analyze_param_step_dict``.
        """
        
        self.waveform_generator = waveform_generator

        # parameter storage
        self.param_dict = param_dict.copy()
        self.analyze_param_step_dict = analyze_param_step_dict.copy()
        self.num_params = len(param_dict.keys())
        self.num_analyze_params = len(analyze_param_step_dict.keys())
        self.param_names = list(param_dict.keys())
        self.analyze_param_names = list(analyze_param_step_dict.keys())

        # Fisher matrix and fiducial waveform
        self.Fisher = np.zeros((self.num_analyze_params, self.num_analyze_params))
        self.waveform_fiducial = self.waveform_generator(param_dict, frequency)
        self.num_channels = len(self.waveform_fiducial)
        
        # noise covariance and frequency grid
        self.frequency = frequency
        self.invcov = inverse_covariance
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
        """Compute the waveform derivative w.r.t. a parameter using
        second-order central finite difference.

        Parameters
        ----------
        param_name : str
            Name of the parameter to differentiate.
        param_shift : float
            Step size for the finite difference.

        Returns
        -------
        ndarray
            Derivative waveform of shape (Nchannel, Nfreq).
        """
        if not (param_name in self.analyze_param_names):
            raise ValueError(param_name + " is not in analyze parameters.")
      
        param_dict_plus = self.param_dict.copy()
        param_dict_plus[param_name] += param_shift
        waveform_plus = self.waveform_generator(param_dict_plus, self.frequency)
        
        param_dict_minus = self.param_dict.copy()
        param_dict_minus[param_name] -= param_shift
        waveform_minus = self.waveform_generator(param_dict_minus, self.frequency)
        # shape: (Nchannel, Nfreq)
        return (waveform_plus - waveform_minus) / (2. * param_shift)
    
    def test_step(self, param_name, init_shift, factor=2., threshold=1e-4, iteration_threshold=50):
        """Tune the finite-difference step size for a single parameter.

        The step is iteratively reduced by ``factor`` until the 1-sigma
        error (computed from the diagonal Fisher element) stabilizes,
        i.e., the relative fluctuation over the last few iterations
        falls below ``threshold``.

        Parameters
        ----------
        param_name : str
            Name of the parameter to tune.
        init_shift : float
            Initial step size (absolute value).
        factor : float, optional
            Reduction factor per iteration.  Default 2.
        threshold : float, optional
            Convergence threshold on relative error fluctuation.
            Default 1e-4.
        iteration_threshold : int, optional
            Maximum number of iterations.  Default 50.
        """
        if not (param_name in self.analyze_param_names):
            raise ValueError(param_name + " is not in analyze parameters.")
        
        shift = init_shift
        derivative1 = self.get_derivative(param_name=param_name, param_shift=shift)
        Fisher_element1 = FrequencyDomainCovarianceSNR(data_channels=derivative1, inv_cov=self.invcov)
        error_record = [1. / np.sqrt(Fisher_element1), ]
        if self.verbose > 0:
            print('initial 1-sigma error = {:.3e}'.format(error_record[0]))

        num_iterate = 0
        rel_deff = 1.
        while(rel_deff > threshold):
            num_iterate += 1
            if num_iterate >= iteration_threshold:
                raise ValueError("Not converged after {} iterations.".format(iteration_threshold))
            if (num_iterate % 10 == 0):
                if self.verbose > 0:
                    print('Iteration No.', num_iterate)
            shift = shift / factor
            derivative2 = self.get_derivative(param_name=param_name, param_shift=shift)
            Fisher_element2 = FrequencyDomainCovarianceSNR(data_channels=derivative2, inv_cov=self.invcov)
            error_record.append(1. / np.sqrt(Fisher_element2))
            if self.verbose > 0:
                print('error = {:.3e}'.format(error_record[-1]))
            
            if num_iterate == 1:
                rel_deff = np.sqrt(np.var([error_record[-2], error_record[-1]])) / np.abs(error_record[-1])
            else:
                rel_deff = np.sqrt(np.var([error_record[-3], error_record[-2], error_record[-1]])) / np.abs(error_record[-1])
            if self.verbose > 0:
                print('shift = {:.3e}, rel diff = {:.3e}'.format(shift, rel_deff))
    
        self.analyze_param_step_dict[param_name] = shift
        if self.verbose > 0:
            print('shift of parameter', param_name, 'is', shift)

    def auto_test_step(self, factor=2., threshold=1e-4, iteration_threshold=50):
        """Automatically tune step sizes for all parameters with negative steps.

        Calls ``test_step()`` on each parameter whose value in
        ``analyze_param_step_dict`` is negative.

        Parameters
        ----------
        factor : float, optional
            Reduction factor per iteration.  Default 2.
        threshold : float, optional
            Convergence threshold.  Default 1e-4.
        iteration_threshold : int, optional
            Maximum iterations per parameter.  Default 50.
        """
        for k, v in self.analyze_param_step_dict.items():
            if v < 0:
                if self.verbose > 0:
                    print('\n ========= testing parameter', k, '==========')
                self.test_step(param_name=k, init_shift=np.abs(v), factor=factor, threshold=threshold, iteration_threshold=iteration_threshold)
        if self.verbose > 0:
            print('all parameters tested.')

    def calculate_Fisher(self):
        """Compute the Fisher information matrix.

        Uses finite-difference derivatives evaluated at the fiducial
        parameter values.  Matrix indices follow ``analyze_param_names``
        ordering.  Must be called after all step sizes are set to positive
        values (manually or via ``auto_test_step()``).
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
                self.Fisher[i1][i2] = np.real(FrequencyDomainCovarianceInnerProduct(
                    data_channels1=self.param_derivatives[name1], 
                    data_channels2=self.param_derivatives[name2], 
                    inv_cov=self.invcov
                    ))
        self.Fisher += self.Fisher.T - np.diag(self.Fisher.diagonal())
        if self.verbose > 0:
            print('index of Fisher:', self.analyze_param_names)
            print('Fisher matrix computed.')

    def calculate_errors(self):
        """Invert the Fisher matrix to obtain 1-sigma parameter uncertainties.

        Computes the covariance matrix via ``np.linalg.inv`` and extracts
        marginal errors.  Sets ``self.CovMatrix`` and ``self.errors``.
        Must be called after ``calculate_Fisher()``.
        """
        self.param_errors = {}
        cond = np.linalg.cond(self.Fisher)
        if self.verbose > 0:
            print('Fisher matrix condition number: {:.1e}'.format(cond))
        if cond > 1e15:
            print('WARNING: Fisher matrix condition number ({:.1e}) exceeds threshold 1e15, results may be unreliable.'.format(cond))
        Covariance = np.linalg.inv(self.Fisher)
        errors = np.sqrt(np.diagonal(Covariance))
        for i in range(self.num_analyze_params):
            name = self.analyze_param_names[i]
            self.param_errors[name] = errors[i]
        self.errors = errors
        self.CovMatrix = Covariance
        if self.verbose > 0:
            print('1-sigma parameter uncertainties:')
            for name in self.analyze_param_names:
                print('  {:<25s} {:.3e}'.format(name, self.param_errors[name]))

    def Cutler_Vallisneri_bias(self, true_waveform):
        """Compute the Cutler & Vallisneri (2007) waveform systematic bias.

        Estimates the parameter bias caused by waveform modelling errors,
        given a ``true_waveform`` that differs from the fiducial template.

        Parameters
        ----------
        true_waveform : ndarray
            The "true" waveform of shape (Nchannel, Nfreq).

        Returns
        -------
        ndarray
            Bias vector of length ``num_analyze_params``.

        Notes
        -----
        ``calculate_errors()`` must be called before this method.
        """
        residual_waveform = self.waveform_fiducial - true_waveform
        bias_vector = np.zeros(self.num_analyze_params)
        for i, v in enumerate(self.param_derivatives.values()):
            bias_vector[i] = FrequencyDomainCovarianceInnerProduct(data_channels1=v, data_channels2=residual_waveform, inv_cov=self.invcov)
        bias = -np.matmul(self.CovMatrix, bias_vector)
        if self.verbose > 0:
            print('Cutler-Vallisneri bias vector:')
            for i, name in enumerate(self.analyze_param_names):
                print('  {:<25s} {:.3e}'.format(name, bias[i]))
        return bias
        

        

    

    