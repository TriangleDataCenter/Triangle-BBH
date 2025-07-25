import numpy as np
import scipy.interpolate as interp 
try:
    import cupy as xp
    import cupyx.scipy.interpolate as xinterp
    # print("has cupy")
except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    import scipy.interpolate as xinterp  
    # print("no cupy")

from scipy.interpolate import interp1d, CubicSpline
from Triangle.Constants import * 



def SpinWeightedSphericalHarmonic(theta, phi, l, m, s=-2):
    # Taken from arXiv:0709.0093v3 eq. (II.7), (II.8) and LALSimulation for the s=-2 case and up to l=4
    """  
    The results are in consistency with LDC Radler manual.
    theta, phi: spherical coordinates of the source frame (i.e. iota0 and phi0 in the LDC manual). 
    This function allows vectorized inputs and outputs.
    """
    
    if s != -2:
        raise ValueError('The only spin-weight implemented for the moment is s = -2.')
        
    if (2 == l):
        if (-2 == m):
            res = np.sqrt( 5.0 / ( 64.0 * PI ) ) * ( 1.0 - np.cos( theta ))*( 1.0 - np.cos( theta ))
        elif (-1 == m):
            res = np.sqrt( 5.0 / ( 16.0 * PI ) ) * np.sin( theta )*( 1.0 - np.cos( theta ))
        elif (0 == m):
            res = np.sqrt( 15.0 / ( 32.0 * PI ) ) * np.sin( theta )*np.sin( theta )
        elif (1 == m):
            res = np.sqrt( 5.0 / ( 16.0 * PI ) ) * np.sin( theta )*( 1.0 + np.cos( theta ))
        elif (2 == m):
            res = np.sqrt( 5.0 / ( 64.0 * PI ) ) * ( 1.0 + np.cos( theta ))*( 1.0 + np.cos( theta ))
        else:
            raise ValueError('Invalid m for l = 2.')
            
    elif (3 == l):
        if (-3 == m):
            res = np.sqrt(21.0/(TWOPI))*np.cos(theta*0.5)*((np.sin(theta*0.5))**(5.))
        elif (-2 == m):
            res = np.sqrt(7.0/(4.0*PI))*(2.0 + 3.0*np.cos(theta))*((np.sin(theta*0.5))**(4.0))
        elif (-1 == m):
            res = np.sqrt(35.0/(TWOPI))*(np.sin(theta) + 4.0*np.sin(2.0*theta) - 3.0*np.sin(3.0*theta))/32.0
        elif (0 == m):
            res = (np.sqrt(105.0/(TWOPI))*np.cos(theta)*(np.sin(theta)*np.sin(theta)))*0.25
        elif (1 == m):
            res = -np.sqrt(35.0/(TWOPI))*(np.sin(theta) - 4.0*np.sin(2.0*theta) - 3.0*np.sin(3.0*theta))/32.0
        elif (2 == m):
            res = np.sqrt(7.0/PI)*((np.cos(theta*0.5))**(4.0))*(-2.0 + 3.0*np.cos(theta))*0.5
        elif (3 == m):
            res = -np.sqrt(21.0/(TWOPI))*((np.cos(theta/2.0))**(5.0))*np.sin(theta*0.5)
        else:
            raise ValueError('Invalid m for l = 3.')
            
    elif (4 == l):
        if (-4 == m):
            res = 3.0*np.sqrt(7.0/PI)*(np.cos(theta*0.5)*np.cos(theta*0.5))*((np.sin(theta*0.5))**6.0)
        elif (-3 == m):
            res = 3.0*np.sqrt(7.0/(TWOPI))*np.cos(theta*0.5)*(1.0 + 2.0*np.cos(theta))*((np.sin(theta*0.5))**5.0)
        elif (-2 == m):
            res = (3.0*(9.0 + 14.0*np.cos(theta) + 7.0*np.cos(2.0*theta))*((np.sin(theta/2.0))**4.0))/(4.0*np.sqrt(PI))
        elif (-1 == m):
            res = (3.0*(3.0*np.sin(theta) + 2.0*np.sin(2.0*theta) + 7.0*np.sin(3.0*theta) - 7.0*np.sin(4.0*theta)))/(32.0*np.sqrt(TWOPI))
        elif (0 == m):
            res = (3.0*np.sqrt(5.0/(TWOPI))*(5.0 + 7.0*np.cos(2.0*theta))*(np.sin(theta)*np.sin(theta)))/16.
        elif (1 == m):
            res = (3.0*(3.0*np.sin(theta) - 2.0*np.sin(2.0*theta) + 7.0*np.sin(3.0*theta) + 7.0*np.sin(4.0*theta)))/(32.0*np.sqrt(TWOPI))
        elif (2 == m):
            res = (3.0*((np.cos(theta*0.5))**4.0)*(9.0 - 14.0*np.cos(theta) + 7.0*np.cos(2.0*theta)))/(4.0*np.sqrt(PI))
        elif (3 == m):
            res = -3.0*np.sqrt(7.0/(TWOPI))*((np.cos(theta*0.5))**5.0)*(-1.0 + 2.0*np.cos(theta))*np.sin(theta*0.5)
        elif (4 == m):
            res = 3.0*np.sqrt(7.0/PI)*((np.cos(theta*0.5))**6.0)*(np.sin(theta*0.5)*np.sin(theta*0.5))
        else:
            raise ValueError('Invalid m for l = 4.')
            
    else:
        raise ValueError('Multipoles with l > 4 not implemented yet.')
    
    return res*np.exp(1j*m*phi)


class FDTDIResponseGenerator():
    """  
        The rule of Fourier transform is the same as S. Marsat, i.e. the ``unusual'' convention,
        aiming to ensure h_lm(f) neq 0 when m > 0 and f > 0.
        Accordingly, h_lm(f) = A_lm(f) * exp[-i Psi_lm(f)].
        The waveform class outputs Phi_lm(f) = -Psi_lm(f), thus h_lm(f) = A_lm(f) exp[i Phi_lm(f)]
        The conjugate will be taken at last to convert to the usual convension.
        
        Parameters should be given in the form of dictionary, with the keys denoting the names of parameters, 
        including:
        "chirp_mass" [MSUN], "mass_ratio" [1], "spin_1z" [1], "spin_2z" [1], 
        "coalescence_time" [DAY] (at the center of detector constellation), "coalescence_phase" [rad],
        "luminosity_distance" [MPC], "inclination" [rad], 
        "longitude" [rad], "latitude" [rad], "psi" [rad]
        
        modes = [(2, 2)] for PhenomD and [(2, 2), (2, 1), (3, 2), (3, 3), (4, 3), (4, 4)] for PhenomHM
    """
    ep_0 = np.array(
        [[1, 0, 0], 
         [0, -1, 0], 
         [0, 0, 0]]
        )
    ec_0 = np.array(
        [[0, 1, 0], 
         [1, 0, 0], 
         [0, 0, 0]]
        )
    def __init__(self, orbit_class, waveform_generator):
        self.orbit = orbit_class
        self.PositionFunctions = self.orbit.Positionfunctions()
        self.ArmVectorFunctions = self.orbit.ArmVectorfunctions()
        self.LTTFunctions = self.orbit.LTTfunctions()   

        self.POS_time_int = orbit_class.POS_time_int
        self.POS_data_int = orbit_class.POS_data_int
        self.ARM_time_int = orbit_class.ARM_time_int
        self.ARM_data_int = orbit_class.ARM_data_int
        self.LTT_time_int = orbit_class.LTT_time_int
        self.LTT_data_int = orbit_class.LTT_data_int  
        
        self.POS0_time_int = self.POS_time_int["1"] # 1, 2, 3 are actually the same, so we use the time of 1 
        self.POS0_data_int = (self.POS_data_int["1"] + self.POS_data_int["2"] + self.POS_data_int["3"]) / 3. # (N_orbit_time, 3)

        self.waveform = waveform_generator
        
        
    def WaveVector(self, parameters):
        """ 
            returns:
            k: wave vector of GW
        """
        l, b = parameters['longitude'], parameters['latitude']
        # shape of k: (Nevents, 3)
        k = -np.array([np.cos(l) * np.cos(b), np.sin(l) * np.cos(b), np.sin(b)])
        return np.transpose(k)
    
    def PolarBasis(self, parameters):
        """ 
            returns:
            ep, ec: polarization bases of GW in the source frame 
        """
        l, b, p = parameters['longitude'], parameters['latitude'], parameters['psi']
        Nevents = l.shape[0]
        
        # shape of O and OT: (Nevents, 3, 3)
        O = np.zeros((3, 3, Nevents))
        O[0][0] = np.sin(l) * np.cos(p) - np.cos(l) * np.sin(b) * np.sin(p)
        O[0][1] = -np.sin(l) * np.sin(p) - np.cos(l) * np.sin(b) * np.cos(p)
        O[0][2] = -np.cos(l) * np.cos(b)
        O[1][0] = -np.cos(l) * np.cos(p) - np.sin(l) * np.sin(b) * np.sin(p)
        O[1][1] = np.cos(l) * np.sin(p) - np.sin(l) * np.sin(b) * np.cos(p)
        O[1][2] = -np.sin(l) * np.cos(b)
        O[2][0] = np.cos(b) * np.sin(p)
        O[2][1] = np.cos(b) * np.cos(p)
        O[2][2] = -np.sin(b)
        O = np.transpose(O, (2, 0, 1))
        OT = np.transpose(O, (0, 2, 1))
        
        # shapes of e+ and ex: (Nevents, 3, 3)        
        ep = np.matmul(np.matmul(O, self.ep_0), OT)
        ec = np.matmul(np.matmul(O, self.ec_0), OT)
        return ep, ec
    
    def PolarBasis_lm(self, parameters, modes=[(2, 2), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3)]):
        """ 
            Polarizations basis tensor Plm for each (l, m) mode stored in a dictionary with the keys being modes.
            The shape of each Plm item is (Nevents, 3, 3).
            Plm = P^+_lm + P^x_lm 
            P^A_lm = K^A_lm * e^A, where A \in {+, x} should be summed over
            K^+_lm = 1/2 * [Ylm + (-1)^l Yl-m^*]
            K^x_lm = i/2 * [Ylm - (-1)^l Yl-m^*]
            the expressions of K are consistent in LDC Radler manual and S. Marsat
        """
        iota0, phi0 = parameters['inclination'], parameters['coalescence_phase']
        
        ep, ec = self.PolarBasis(parameters=parameters)
        
        # Plm 
        Plm = {}
        for l, m in modes:
            # array of shape (Nevents)
            Ylm = SpinWeightedSphericalHarmonic(theta=iota0, phi=phi0, l=l, m=m)
            Yl_m_star = np.conjugate(SpinWeightedSphericalHarmonic(theta=iota0, phi=phi0, l=l, m=-m)) * (-1.) ** l
            Kplm = 0.5 * (Ylm + Yl_m_star)
            Kclm = 0.5j * (Ylm - Yl_m_star)
            
            Plm[(l, m)] = \
                np.expand_dims(np.expand_dims(Kplm, axis=-1), axis=-1) * ep \
                + np.expand_dims(np.expand_dims(Kclm, axis=-1), axis=-1) * ec
        return Plm 

    def TransferFunction(self, t, f, k, Plm, TDIGeneration='2nd', tmin=None, tmax=None):
        """  
            t and f should have the same shape (Nevents, Nfreqs). 
            k is the wave vector of GW with shape (Nevents, 3)
            Plm is the polar basis of a specific (l, m) mode with shape (Nevents, 3, 3).
            
            arraies which should be calculated for each `ij' arm:
            dij : (Nevents, Nfreqs) in [s]
            nij : (Nevents, Nfreqs, 3)
            knij : (Nevents, Nfreqs)
            pi : (Nevents, Nfreqs, 3) in [s]
            pj : (Nevents, Nfreqs, 3) in [s]
            prefactorij : (Nevents, Nfreqs)
            sincfactorij : (Nevents, Nfreqs)
            expfactorij : (Nevents, Nfreqs)
            Fij : (Nevents, Nfreqs) = nij \bigocross nij : Plm
            Gij : (Nevents, Nfreqs) i.e. G^lm_slr in S. Marsat
            X/Y/Z plus/minus factor : (Nevents, Nfreqs)
            GTDI : (Nevnets, Nfreqs) i.e. transfer functions for TDI channels X, Y, Z
            these values all depend on lm implicitly due to the t-f relations, thus should be calculated per mode
            the data will first be calculated at the frequency grids and  stored in dictionaries.
            
            Transfer functions before tmin [DAY] are set to 0.
        """
        (Nevents, Nfreqs) = t.shape
        t_flatten = t.flatten()

        # positions 
        # p1 = self.PositionFunctions['1'](t)
        # p2 = self.PositionFunctions['2'](t)
        # p3 = self.PositionFunctions['3'](t)
        p1 = np.array([
            np.interp(x=t_flatten, xp=self.POS_time_int["1"], fp=self.POS_data_int["1"][:, 0]),
            np.interp(x=t_flatten, xp=self.POS_time_int["1"], fp=self.POS_data_int["1"][:, 1]),
            np.interp(x=t_flatten, xp=self.POS_time_int["1"], fp=self.POS_data_int["1"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        p2 = np.array([
            np.interp(x=t_flatten, xp=self.POS_time_int["2"], fp=self.POS_data_int["2"][:, 0]),
            np.interp(x=t_flatten, xp=self.POS_time_int["2"], fp=self.POS_data_int["2"][:, 1]),
            np.interp(x=t_flatten, xp=self.POS_time_int["2"], fp=self.POS_data_int["2"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        p3 = np.array([
            np.interp(x=t_flatten, xp=self.POS_time_int["3"], fp=self.POS_data_int["3"][:, 0]),
            np.interp(x=t_flatten, xp=self.POS_time_int["3"], fp=self.POS_data_int["3"][:, 1]),
            np.interp(x=t_flatten, xp=self.POS_time_int["3"], fp=self.POS_data_int["3"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        
        # arm lengths 
        # d12 = self.LTTFunctions['12'](t)
        # d13 = self.LTTFunctions['13'](t)
        # d23 = self.LTTFunctions['23'](t)
        # d21 = self.LTTFunctions['21'](t)
        # d31 = self.LTTFunctions['31'](t)
        # d32 = self.LTTFunctions['32'](t)
        d12 = np.interp(x=t_flatten, xp=self.LTT_time_int["12"], fp=self.LTT_data_int["12"]).reshape((Nevents, Nfreqs))
        d13 = np.interp(x=t_flatten, xp=self.LTT_time_int["13"], fp=self.LTT_data_int["13"]).reshape((Nevents, Nfreqs))
        d23 = np.interp(x=t_flatten, xp=self.LTT_time_int["23"], fp=self.LTT_data_int["23"]).reshape((Nevents, Nfreqs))
        d21 = np.interp(x=t_flatten, xp=self.LTT_time_int["21"], fp=self.LTT_data_int["21"]).reshape((Nevents, Nfreqs))
        d31 = np.interp(x=t_flatten, xp=self.LTT_time_int["31"], fp=self.LTT_data_int["31"]).reshape((Nevents, Nfreqs))
        d32 = np.interp(x=t_flatten, xp=self.LTT_time_int["32"], fp=self.LTT_data_int["32"]).reshape((Nevents, Nfreqs))
        
        
        # arm directions 
        # n12 = self.ArmVectorFunctions['12'](t)
        # n13 = self.ArmVectorFunctions['13'](t)
        # n23 = self.ArmVectorFunctions['23'](t)
        # n21 = self.ArmVectorFunctions['21'](t)
        # n31 = self.ArmVectorFunctions['31'](t)
        # n32 = self.ArmVectorFunctions['32'](t)
        n12 = np.array([
            np.interp(x=t_flatten, xp=self.ARM_time_int["12"], fp=self.ARM_data_int["12"][:, 0]),
            np.interp(x=t_flatten, xp=self.ARM_time_int["12"], fp=self.ARM_data_int["12"][:, 1]),
            np.interp(x=t_flatten, xp=self.ARM_time_int["12"], fp=self.ARM_data_int["12"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        n13 = np.array([
            np.interp(x=t_flatten, xp=self.ARM_time_int["13"], fp=self.ARM_data_int["13"][:, 0]),
            np.interp(x=t_flatten, xp=self.ARM_time_int["13"], fp=self.ARM_data_int["13"][:, 1]),
            np.interp(x=t_flatten, xp=self.ARM_time_int["13"], fp=self.ARM_data_int["13"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        n23 = np.array([
            np.interp(x=t_flatten, xp=self.ARM_time_int["23"], fp=self.ARM_data_int["23"][:, 0]),
            np.interp(x=t_flatten, xp=self.ARM_time_int["23"], fp=self.ARM_data_int["23"][:, 1]),
            np.interp(x=t_flatten, xp=self.ARM_time_int["23"], fp=self.ARM_data_int["23"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        n21 = np.array([
            np.interp(x=t_flatten, xp=self.ARM_time_int["21"], fp=self.ARM_data_int["21"][:, 0]),
            np.interp(x=t_flatten, xp=self.ARM_time_int["21"], fp=self.ARM_data_int["21"][:, 1]),
            np.interp(x=t_flatten, xp=self.ARM_time_int["21"], fp=self.ARM_data_int["21"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        n31 = np.array([
            np.interp(x=t_flatten, xp=self.ARM_time_int["31"], fp=self.ARM_data_int["31"][:, 0]),
            np.interp(x=t_flatten, xp=self.ARM_time_int["31"], fp=self.ARM_data_int["31"][:, 1]),
            np.interp(x=t_flatten, xp=self.ARM_time_int["31"], fp=self.ARM_data_int["31"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        n32 = np.array([
            np.interp(x=t_flatten, xp=self.ARM_time_int["32"], fp=self.ARM_data_int["32"][:, 0]),
            np.interp(x=t_flatten, xp=self.ARM_time_int["32"], fp=self.ARM_data_int["32"][:, 1]),
            np.interp(x=t_flatten, xp=self.ARM_time_int["32"], fp=self.ARM_data_int["32"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        
        # k \dot n 
        ke = np.expand_dims(k, axis=1)
        kn12 =  np.sum(ke * n12, axis=2)
        kn13 =  np.sum(ke * n13, axis=2)
        kn23 =  np.sum(ke * n23, axis=2)
        kn21 =  np.sum(ke * n21, axis=2)
        kn31 =  np.sum(ke * n31, axis=2)
        kn32 =  np.sum(ke * n32, axis=2)
        
        # k \dot (p_receive + p_send)
        kp1p2 = np.sum(ke * (p1 + p2), axis=2)
        kp2p3 = np.sum(ke * (p2 + p3), axis=2)
        kp3p1 = np.sum(ke * (p3 + p1), axis=2)
        
        # prefactor = i*PI*f*d
        prefactor12 = 1.j * PI * f * d12
        prefactor13 = 1.j * PI * f * d13
        prefactor23 = 1.j * PI * f * d23
        prefactor21 = 1.j * PI * f * d21
        prefactor31 = 1.j * PI * f * d31
        prefactor32 = 1.j * PI * f * d32
        
        # sincfactor = sinc(PI*f*d*(1-kn)) = np.sinc(f*d*(1-kn))
        sincfactor12 = np.sinc(f * d12 * (1. - kn12))
        sincfactor13 = np.sinc(f * d13 * (1. - kn13))
        sincfactor23 = np.sinc(f * d23 * (1. - kn23))
        sincfactor21 = np.sinc(f * d21 * (1. - kn21))
        sincfactor31 = np.sinc(f * d31 * (1. - kn31))
        sincfactor32 = np.sinc(f * d32 * (1. - kn32))
        
        # expfactor = exp(i*PI*f*(d+k(p_receive+p_send)))
        expfactor12 = np.exp(1.j * PI * f * (d12 + kp1p2))
        expfactor13 = np.exp(1.j * PI * f * (d13 + kp3p1))
        expfactor23 = np.exp(1.j * PI * f * (d23 + kp2p3))
        expfactor21 = np.exp(1.j * PI * f * (d21 + kp1p2))
        expfactor31 = np.exp(1.j * PI * f * (d31 + kp3p1))
        expfactor32 = np.exp(1.j * PI * f * (d32 + kp2p3))
        
        # antenna pattern functions Fij
        # matmul(n12, Plm) is of shape (Nevents, Nfreqs, 3)
        F12 = np.sum(np.matmul(n12, Plm) * n12, axis=2)
        F13 = np.sum(np.matmul(n13, Plm) * n13, axis=2)
        F23 = np.sum(np.matmul(n23, Plm) * n23, axis=2)
        F21 = np.sum(np.matmul(n21, Plm) * n21, axis=2)
        F31 = np.sum(np.matmul(n31, Plm) * n31, axis=2)
        F32 = np.sum(np.matmul(n32, Plm) * n32, axis=2)
        
        # combine into single-arm transferfunction  Gij
        Gij = {}
        Gij['12'] = prefactor12 * sincfactor12 * expfactor12 * F12
        Gij['13'] = prefactor13 * sincfactor13 * expfactor13 * F13
        Gij['23'] = prefactor23 * sincfactor23 * expfactor23 * F23
        Gij['21'] = prefactor21 * sincfactor21 * expfactor21 * F21
        Gij['31'] = prefactor31 * sincfactor31 * expfactor31 * F31
        Gij['32'] = prefactor32 * sincfactor32 * expfactor32 * F32
        
        # delay factors 
        D12 = np.exp(1.j * TWOPI * f * d12) 
        D13 = np.exp(1.j * TWOPI * f * d13) 
        D23 = np.exp(1.j * TWOPI * f * d23) 
        D21 = np.exp(1.j * TWOPI * f * d21) 
        D31 = np.exp(1.j * TWOPI * f * d31) 
        D32 = np.exp(1.j * TWOPI * f * d32) 
        
        # combine into TDI transferfunction GTDI
        if TDIGeneration == '1st':
            X1plusfactor = (1. - D12 * D21) 
            X1minusfactor = (1. - D13 * D31) 
            Y1plusfactor = (1. - D23 * D32) 
            Y1minusfactor = (1. - D21 * D12) 
            Z1plusfactor = (1. - D31 * D13) 
            Z1minusfactor = (1. - D32 * D23) 
            GTDI = {} 
            GTDI['X'] = X1plusfactor * (Gij['13'] + Gij['31'] * D13) - X1minusfactor * (Gij['12'] + Gij['21'] * D12)
            GTDI['Y'] = Y1plusfactor * (Gij['21'] + Gij['12'] * D21) - Y1minusfactor * (Gij['23'] + Gij['32'] * D23)
            GTDI['Z'] = Z1plusfactor * (Gij['32'] + Gij['23'] * D32) - Z1minusfactor * (Gij['31'] + Gij['13'] * D31)
            
        elif TDIGeneration == '2nd':
            D121 = D12 * D21
            D131 = D13 * D31 
            D12131_1 = 1. - D121 * D131 # 1 - D12131
            X2plusfactor = (1. - D121) * D12131_1
            X2minusfactor = (1. - D131) * D12131_1
            D232 = D23 * D32 
            D212 = D21 * D12 
            D23212_1 = 1. - D232 * D212 
            Y2plusfactor = (1. - D232) * D23212_1
            Y2minusfactor = (1. - D212) * D23212_1
            D313 = D31 * D13 
            D323 = D32 * D23 
            D31323_1 = 1. - D313 * D323
            Z2plusfactor = (1. - D313) * D31323_1
            Z2minusfactor = (1. - D323) * D31323_1
            GTDI = {} 
            GTDI['X'] = X2plusfactor * (Gij['13'] + Gij['31'] * D13) - X2minusfactor * (Gij['12'] + Gij['21'] * D12)
            GTDI['Y'] = Y2plusfactor * (Gij['21'] + Gij['12'] * D21) - Y2minusfactor * (Gij['23'] + Gij['32'] * D23)
            GTDI['Z'] = Z2plusfactor * (Gij['32'] + Gij['23'] * D32) - Z2minusfactor * (Gij['31'] + Gij['13'] * D31)
            
        if tmin != None:
            zero_inds = np.where(t < tmin * DAY)
            for k in Gij.keys():
                Gij[k][zero_inds] = 0.
            for k in GTDI.keys():
                GTDI[k][zero_inds] = 0.
                
        if tmax != None:
            zero_inds = np.where(t > tmax * DAY)
            for k in Gij.keys():
                Gij[k][zero_inds] = 0.
            for k in GTDI.keys():
                GTDI[k][zero_inds] = 0.
        
        return Gij, GTDI
    
    def SSBToConstellationDelay(self, k, parameters):
        """  
            returns the delay of coalescence time at constellation center w.r.t SSB 
            tc = tc^SSB + tc^delay --> tc^SSB = tc - tc^delay
        """
        tc_SI = parameters['coalescence_time'] * DAY # (Nevent)
        p0 = np.transpose(np.array([
            np.interp(x=tc_SI, xp=self.POS0_time_int, fp=self.POS0_data_int[:, 0]),
            np.interp(x=tc_SI, xp=self.POS0_time_int, fp=self.POS0_data_int[:, 1]),
            np.interp(x=tc_SI, xp=self.POS0_time_int, fp=self.POS0_data_int[:, 2]),
        ])) # (3, Nevent) -> (Nevent, 3), in second unit 
        return np.sum(k * p0, axis=1) # (Nevent), in second unit 
    
    def Response(
        self, 
        parameters, 
        freqs, 
        fmin=1e-5, # used to calculate freq grid
        fmax=1e-1, # used to calculate freq grid
        Nfreqs=1024, # used to calculate freq grid
        modes=[(2, 2), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3)], 
        tmin=None, # minimum time of orbit in day 
        tmax=None, # maximum time of orbit in day
        tc_at_constellation=False, # whether tc is the coalescence time at constellation center (True) or SSB (False)
        TDIGeneration='2nd',
        optimal_combination=True,
        interpolation_method='cubic',
        output_by_mode=False, 
        ):
        """   
            Calculate 1st/2nd-generation TDI response in the frequency domain.
            Work flow:
            PolarBasis_lm, WaveVector --> frequency grids, Waveform, time grids
            --> TDI TransferFunction --> Interpolation
        """
        
        # convert scalar parameters to arraies 
        parameter_dict = parameters.copy()
        for k, v in parameters.items():
            parameter_dict[k] = np.atleast_1d(v)
        
        Nevents = parameter_dict['chirp_mass'].shape[0]
        
        # calculate polarization tensors (mode-dependent) and wavevectors (mode-independent)
        Plm = self.PolarBasis_lm(parameters=parameter_dict, modes=modes) 
        k = self.WaveVector(parameters=parameter_dict) 
        
        # convert the coalescence time from constellaton center to SSB
        if tc_at_constellation:
            tc_delay = self.SSBToConstellationDelay(k, parameter_dict) # (Nevent)
            parameter_dict['coalescence_time'] += -tc_delay / DAY # (Nevent)
            # print("delayed tc =", tc_delay)
            # print("converted tc =", parameter_dict["coalescence_time"])
        
        # calculate frequency grids (mode-independent), waveforms and time grids (mode-dependent)
        # by default all the modes will be calculated 
        if interpolation_method == None:
            fgrids, amps, phas, tgrids = self.waveform(parameters=parameter_dict, Nfreqs=Nfreqs, fmin=fmin, fmax=fmax, freqs=np.tile(freqs, (Nevents, 1)))
        else:
            fgrids, amps, phas, tgrids = self.waveform(parameters=parameter_dict, Nfreqs=Nfreqs, fmin=fmin, fmax=fmax, freqs=None)

        # calculate transfer function at the frequency and time grids and then interpolate (Nevents, Nfreqs) --> (Nevents, Nfreqs_out)
        Nfreqs_out = freqs.shape[-1]
        fill_value=0.
        # fill_value="extrapolate"
        if output_by_mode:
            Nmode = len(modes)
            X = np.zeros((Nmode, Nevents, Nfreqs_out), dtype=np.complex128)
            Y = np.zeros((Nmode, Nevents, Nfreqs_out), dtype=np.complex128)
            Z = np.zeros((Nmode, Nevents, Nfreqs_out), dtype=np.complex128)
            if interpolation_method == None: 
                for imode, mode in enumerate(modes):
                    _, GTDI = self.TransferFunction(t=tgrids[mode], f=fgrids, k=k, Plm=Plm[mode], TDIGeneration=TDIGeneration, tmin=tmin, tmax=tmax)
                    Xamp_int = GTDI['X'] * amps[mode]
                    Yamp_int = GTDI['Y'] * amps[mode]
                    Zamp_int = GTDI['Z'] * amps[mode]
                    phase_int = phas[mode]
                    X[imode] = Xamp_int * np.exp(1.j * phase_int)
                    Y[imode] = Yamp_int * np.exp(1.j * phase_int)
                    Z[imode] = Zamp_int * np.exp(1.j * phase_int)       
            else:
                for imode, mode in enumerate(modes):
                    _, GTDI = self.TransferFunction(t=tgrids[mode], f=fgrids, k=k, Plm=Plm[mode], TDIGeneration=TDIGeneration, tmin=tmin, tmax=tmax)
                    Xamp_int = GTDI['X'] * amps[mode]
                    Yamp_int = GTDI['Y'] * amps[mode]
                    Zamp_int = GTDI['Z'] * amps[mode]
                    phase_int = phas[mode]
                    Xamp_func = interp1d(x=fgrids[0], y=Xamp_int, kind=interpolation_method, axis=1, bounds_error=False, fill_value=fill_value)
                    Yamp_func = interp1d(x=fgrids[0], y=Yamp_int, kind=interpolation_method, axis=1, bounds_error=False, fill_value=fill_value)
                    Zamp_func = interp1d(x=fgrids[0], y=Zamp_int, kind=interpolation_method, axis=1, bounds_error=False, fill_value=fill_value)
                    phase_func = interp1d(x=fgrids[0], y=phase_int, kind=interpolation_method, axis=1, bounds_error=False, fill_value=fill_value)
                    Xamp = Xamp_func(freqs)
                    Yamp = Yamp_func(freqs)
                    Zamp = Zamp_func(freqs)
                    phase = phase_func(freqs)
                    X[imode] = Xamp * np.exp(1.j * phase)
                    Y[imode] = Yamp * np.exp(1.j * phase)
                    Z[imode] = Zamp * np.exp(1.j * phase)       
        else:
            X = np.zeros((Nevents, Nfreqs_out), dtype=np.complex128)
            Y = np.zeros((Nevents, Nfreqs_out), dtype=np.complex128)
            Z = np.zeros((Nevents, Nfreqs_out), dtype=np.complex128)
            if interpolation_method == None: 
                for mode in modes:
                    _, GTDI = self.TransferFunction(t=tgrids[mode], f=fgrids, k=k, Plm=Plm[mode], TDIGeneration=TDIGeneration, tmin=tmin, tmax=tmax)
                    Xamp_int = GTDI['X'] * amps[mode]
                    Yamp_int = GTDI['Y'] * amps[mode]
                    Zamp_int = GTDI['Z'] * amps[mode]
                    phase_int = phas[mode]
                    X += Xamp_int * np.exp(1.j * phase_int)
                    Y += Yamp_int * np.exp(1.j * phase_int)
                    Z += Zamp_int * np.exp(1.j * phase_int)
            else:
                for mode in modes:
                    _, GTDI = self.TransferFunction(t=tgrids[mode], f=fgrids, k=k, Plm=Plm[mode], TDIGeneration=TDIGeneration, tmin=tmin, tmax=tmax)
                    Xamp_int = GTDI['X'] * amps[mode]
                    Yamp_int = GTDI['Y'] * amps[mode]
                    Zamp_int = GTDI['Z'] * amps[mode]
                    phase_int = phas[mode]
                    Xamp_func = interp1d(x=fgrids[0], y=Xamp_int, kind=interpolation_method, axis=1, bounds_error=False, fill_value=fill_value)
                    Yamp_func = interp1d(x=fgrids[0], y=Yamp_int, kind=interpolation_method, axis=1, bounds_error=False, fill_value=fill_value)
                    Zamp_func = interp1d(x=fgrids[0], y=Zamp_int, kind=interpolation_method, axis=1, bounds_error=False, fill_value=fill_value)
                    phase_func = interp1d(x=fgrids[0], y=phase_int, kind=interpolation_method, axis=1, bounds_error=False, fill_value=fill_value)
                    Xamp = Xamp_func(freqs)
                    Yamp = Yamp_func(freqs)
                    Zamp = Zamp_func(freqs)
                    phase = phase_func(freqs)
                    X += Xamp * np.exp(1.j * phase)
                    Y += Yamp * np.exp(1.j * phase)
                    Z += Zamp * np.exp(1.j * phase)
        if Nevents == 1: 
            X = np.conjugate(X[0])
            Y = np.conjugate(Y[0])
            Z = np.conjugate(Z[0])
        else:
            X = np.conjugate(X)
            Y = np.conjugate(Y)
            Z = np.conjugate(Z)
        
        if optimal_combination:
            A, E, T = self.AETfromXYZ(X, Y, Z)
            results = np.array([A, E, T]) 
        else:
            results = np.array([X, Y, Z])
        
        results[np.abs(results)<1e-25]=0.
        return results 
            
    def AETfromXYZ(self, X, Y, Z):
        A = (Z - X) / np.sqrt(2.)
        E = (X - 2. * Y + Z) / np.sqrt(6.)
        T = (X + Y + Z) / np.sqrt(3.)
        return A, E, T

    def XYZfromAET(self, A, E, T):
        X = (-np.sqrt(3.) * A + E + np.sqrt(2.) * T) / np.sqrt(6.)
        Y = (T - np.sqrt(2.) * E) / np.sqrt(3.)
        Z = (np.sqrt(3.) * A + E + np.sqrt(2.) * T) / np.sqrt(6.)
        return X, Y, Z





class BBHxFDTDIResponseGenerator():
    """  
        The rule of Fourier transform is the same as S. Marsat, which is an ``unusual'' convention,
        aiming to ensure h_lm(f) neq 0 when m > 0 and f > 0.
        Accordingly, h_lm(f) = A_lm(f) * exp[-i Psi_lm(f)].
        The waveform class outputs Phi_lm(f) = -Psi_lm(f), thus h_lm(f) = A_lm(f) exp[i Phi_lm(f)]
        While, the complex conjugate of results will be taken at last to convert to the usual FT convension.
        
        Parameters should be given in the form of dictionary, with the keys denoting the names of parameters, 
        including:
        "chirp_mass" [MSUN], "mass_ratio" [1], "spin_1z" [1], "spin_2z" [1], 
        "coalescence_time" [DAY] (at the center of detector constellation), "coalescence_phase" [rad],
        "luminosity_distance" [MPC], "inclination" [rad], 
        "longitude" [rad], "latitude" [rad], "psi" [rad]
        
        modes = [(2, 2)] for PhenomD and [(2, 2), (2, 1), (3, 2), (3, 3), (4, 3), (4, 4)] for PhenomHM
    """
    ep_0 = np.array(
        [[1, 0, 0], 
         [0, -1, 0], 
         [0, 0, 0]]
        )
    ec_0 = np.array(
        [[0, 1, 0], 
         [1, 0, 0], 
         [0, 0, 0]]
        )
    def __init__(self, orbit_class, waveform_generator, use_gpu):
        self.orbit = orbit_class
        self.PositionFunctions = self.orbit.Positionfunctions()
        self.ArmVectorFunctions = self.orbit.ArmVectorfunctions()
        self.LTTFunctions = self.orbit.LTTfunctions()     
            
        self.waveform = waveform_generator
        
        self.use_gpu = use_gpu
        if use_gpu: 
            self.xp = xp 
            self.xinterp = xinterp
        else: 
            self.xp = np 
            self.xinterp = interp
            
        self.POS_time_int = orbit_class.POS_time_int
        self.POS_data_int = orbit_class.POS_data_int
        self.ARM_time_int = orbit_class.ARM_time_int
        self.ARM_data_int = orbit_class.ARM_data_int
        self.LTT_time_int = orbit_class.LTT_time_int
        self.LTT_data_int = orbit_class.LTT_data_int   
        
        # convert to xp array 
        for data_dict in [self.POS_time_int, self.POS_data_int, self.ARM_time_int, self.ARM_data_int, self.LTT_time_int, self.LTT_data_int]:
            for k, v in data_dict.items():
                data_dict[k] = self.xp.array(v)
                
        # calculate the sparse data of R0
        self.POS0_time_int = self.POS_time_int["1"] # 1, 2, 3 are actually the same, so we use the time of 1 
        self.POS0_data_int = (self.POS_data_int["1"] + self.POS_data_int["2"] + self.POS_data_int["3"]) / 3. # (N_orbit_time, 3)

        if use_gpu: 
            self.ep_0 = self.xp.array(self.ep_0)
            self.ec_0 = self.xp.array(self.ec_0)
        
    def SpinWeightedSphericalHarmonic(self, theta, phi, l, m, s=-2):
        # Taken from arXiv:0709.0093v3 eq. (II.7), (II.8) and LALSimulation for the s=-2 case and up to l=4
        """  
            The results are in consistency with LDC Radler manual.
            theta, phi: spherical coordinates of the source frame (i.e. iota0 and phi0 in the LDC manual). 
            This function allows vectorized inputs and outputs.
        """
        
        if s != -2:
            raise ValueError('The only spin-weight implemented for the moment is s = -2.')
            
        if (2 == l):
            if (-2 == m):
                res = self.xp.sqrt( 5.0 / ( 64.0 * PI ) ) * ( 1.0 - self.xp.cos( theta ))*( 1.0 - self.xp.cos( theta ))
            elif (-1 == m):
                res = self.xp.sqrt( 5.0 / ( 16.0 * PI ) ) * self.xp.sin( theta )*( 1.0 - self.xp.cos( theta ))
            elif (0 == m):
                res = self.xp.sqrt( 15.0 / ( 32.0 * PI ) ) * self.xp.sin( theta )*self.xp.sin( theta )
            elif (1 == m):
                res = self.xp.sqrt( 5.0 / ( 16.0 * PI ) ) * self.xp.sin( theta )*( 1.0 + self.xp.cos( theta ))
            elif (2 == m):
                res = self.xp.sqrt( 5.0 / ( 64.0 * PI ) ) * ( 1.0 + self.xp.cos( theta ))*( 1.0 + self.xp.cos( theta ))
            else:
                raise ValueError('Invalid m for l = 2.')
                
        elif (3 == l):
            if (-3 == m):
                res = self.xp.sqrt(21.0/(TWOPI))*self.xp.cos(theta*0.5)*((self.xp.sin(theta*0.5))**(5.))
            elif (-2 == m):
                res = self.xp.sqrt(7.0/(4.0*PI))*(2.0 + 3.0*self.xp.cos(theta))*((self.xp.sin(theta*0.5))**(4.0))
            elif (-1 == m):
                res = self.xp.sqrt(35.0/(TWOPI))*(self.xp.sin(theta) + 4.0*self.xp.sin(2.0*theta) - 3.0*self.xp.sin(3.0*theta))/32.0
            elif (0 == m):
                res = (self.xp.sqrt(105.0/(TWOPI))*self.xp.cos(theta)*(self.xp.sin(theta)*self.xp.sin(theta)))*0.25
            elif (1 == m):
                res = -self.xp.sqrt(35.0/(TWOPI))*(self.xp.sin(theta) - 4.0*self.xp.sin(2.0*theta) - 3.0*self.xp.sin(3.0*theta))/32.0
            elif (2 == m):
                res = self.xp.sqrt(7.0/PI)*((self.xp.cos(theta*0.5))**(4.0))*(-2.0 + 3.0*self.xp.cos(theta))*0.5
            elif (3 == m):
                res = -self.xp.sqrt(21.0/(TWOPI))*((self.xp.cos(theta/2.0))**(5.0))*self.xp.sin(theta*0.5)
            else:
                raise ValueError('Invalid m for l = 3.')
                
        elif (4 == l):
            if (-4 == m):
                res = 3.0*self.xp.sqrt(7.0/PI)*(self.xp.cos(theta*0.5)*self.xp.cos(theta*0.5))*((self.xp.sin(theta*0.5))**6.0)
            elif (-3 == m):
                res = 3.0*self.xp.sqrt(7.0/(TWOPI))*self.xp.cos(theta*0.5)*(1.0 + 2.0*self.xp.cos(theta))*((self.xp.sin(theta*0.5))**5.0)
            elif (-2 == m):
                res = (3.0*(9.0 + 14.0*self.xp.cos(theta) + 7.0*self.xp.cos(2.0*theta))*((self.xp.sin(theta/2.0))**4.0))/(4.0*self.xp.sqrt(PI))
            elif (-1 == m):
                res = (3.0*(3.0*self.xp.sin(theta) + 2.0*self.xp.sin(2.0*theta) + 7.0*self.xp.sin(3.0*theta) - 7.0*self.xp.sin(4.0*theta)))/(32.0*self.xp.sqrt(TWOPI))
            elif (0 == m):
                res = (3.0*self.xp.sqrt(5.0/(TWOPI))*(5.0 + 7.0*self.xp.cos(2.0*theta))*(self.xp.sin(theta)*self.xp.sin(theta)))/16.
            elif (1 == m):
                res = (3.0*(3.0*self.xp.sin(theta) - 2.0*self.xp.sin(2.0*theta) + 7.0*self.xp.sin(3.0*theta) + 7.0*self.xp.sin(4.0*theta)))/(32.0*self.xp.sqrt(TWOPI))
            elif (2 == m):
                res = (3.0*((self.xp.cos(theta*0.5))**4.0)*(9.0 - 14.0*self.xp.cos(theta) + 7.0*self.xp.cos(2.0*theta)))/(4.0*self.xp.sqrt(PI))
            elif (3 == m):
                res = -3.0*self.xp.sqrt(7.0/(TWOPI))*((self.xp.cos(theta*0.5))**5.0)*(-1.0 + 2.0*self.xp.cos(theta))*self.xp.sin(theta*0.5)
            elif (4 == m):
                res = 3.0*self.xp.sqrt(7.0/PI)*((self.xp.cos(theta*0.5))**6.0)*(self.xp.sin(theta*0.5)*self.xp.sin(theta*0.5))
            else:
                raise ValueError('Invalid m for l = 4.')
                
        else:
            raise ValueError('Multipoles with l > 4 not implemented yet.')
        
        return res*self.xp.exp(1j*m*phi)
        
        
    def WaveVector(self, parameters):
        """ 
            returns:
            k: wave vector of GW
        """
        l, b = parameters['longitude'], parameters['latitude']
        # shape of k: (Nevents, 3)
        k = -self.xp.array([self.xp.cos(l) * self.xp.cos(b), self.xp.sin(l) * self.xp.cos(b), self.xp.sin(b)])
        return self.xp.transpose(k)
    
    def PolarBasis(self, parameters):
        """ 
            returns:
            ep, ec: polarization bases of GW in the source frame 
        """
        l, b, p = parameters['longitude'], parameters['latitude'], parameters['psi']
        Nevents = l.shape[0]
        
        # shape of O and OT: (Nevents, 3, 3)
        O = self.xp.zeros((3, 3, Nevents))
        O[0][0] = self.xp.sin(l) * self.xp.cos(p) - self.xp.cos(l) * self.xp.sin(b) * self.xp.sin(p)
        O[0][1] = -self.xp.sin(l) * self.xp.sin(p) - self.xp.cos(l) * self.xp.sin(b) * self.xp.cos(p)
        O[0][2] = -self.xp.cos(l) * self.xp.cos(b)
        O[1][0] = -self.xp.cos(l) * self.xp.cos(p) - self.xp.sin(l) * self.xp.sin(b) * self.xp.sin(p)
        O[1][1] = self.xp.cos(l) * self.xp.sin(p) - self.xp.sin(l) * self.xp.sin(b) * self.xp.cos(p)
        O[1][2] = -self.xp.sin(l) * self.xp.cos(b)
        O[2][0] = self.xp.cos(b) * self.xp.sin(p)
        O[2][1] = self.xp.cos(b) * self.xp.cos(p)
        O[2][2] = -self.xp.sin(b)
        O = self.xp.transpose(O, (2, 0, 1))
        OT = self.xp.transpose(O, (0, 2, 1))
        
        # shapes of e+ and ex: (Nevents, 3, 3)        
        ep = self.xp.matmul(self.xp.matmul(O, self.ep_0), OT)
        ec = self.xp.matmul(self.xp.matmul(O, self.ec_0), OT)
        return ep, ec
    
    def PolarBasis_lm(self, parameters, modes=[(2, 2), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3)]):
        """ 
            Polarizations basis tensor Plm for each (l, m) mode stored in a dictionary with the keys being modes.
            The shape of each Plm item is (Nevents, 3, 3).
            Plm = P^+_lm + P^x_lm 
            P^A_lm = K^A_lm * e^A, where A \in {+, x} should be summed over
            K^+_lm = 1/2 * [Ylm + (-1)^l Yl-m^*]
            K^x_lm = i/2 * [Ylm - (-1)^l Yl-m^*]
            the expressions of K are consistent in LDC Radler manual and S. Marsat
        """
        iota0, phi0 = parameters['inclination'], parameters['coalescence_phase']
        
        ep, ec = self.PolarBasis(parameters=parameters)
        
        # Plm 
        Plm = {}
        for l, m in modes:
            # array of shape (Nevents)
            Ylm = self.SpinWeightedSphericalHarmonic(theta=iota0, phi=phi0, l=l, m=m)
            Yl_m_star = self.xp.conjugate(self.SpinWeightedSphericalHarmonic(theta=iota0, phi=phi0, l=l, m=-m)) * (-1.) ** l
            Kplm = 0.5 * (Ylm + Yl_m_star)
            Kclm = 0.5j * (Ylm - Yl_m_star)
            
            Plm[(l, m)] = \
                self.xp.expand_dims(self.xp.expand_dims(Kplm, axis=-1), axis=-1) * ep \
                + self.xp.expand_dims(self.xp.expand_dims(Kclm, axis=-1), axis=-1) * ec
        return Plm 
    

    def TransferFunction(self, t, f, k, Plm, TDIGeneration='2nd', tmin=None, tmax=None):
        """  
            t and f should have the same shape (Nevents, Nfreqs). 
            k is the wave vector of GW with shape (Nevents, 3)
            Plm is the polar basis of a specific (l, m) mode with shape (Nevents, 3, 3).
            
            arraies which should be calculated for each `ij' arm:
            dij : (Nevents, Nfreqs) in second unit 
            nij : (Nevents, Nfreqs, 3)
            knij : (Nevents, Nfreqs)
            pi : (Nevents, Nfreqs, 3) in second unit 
            pj : (Nevents, Nfreqs, 3)
            prefactorij : (Nevents, Nfreqs)
            sincfactorij : (Nevents, Nfreqs)
            expfactorij : (Nevents, Nfreqs)
            Fij : (Nevents, Nfreqs) = nij \bigocross nij : Plm
            Gij : (Nevents, Nfreqs) i.e. G^lm_slr in S. Marsat
            X/Y/Z plus/minus factor : (Nevents, Nfreqs)
            GTDI : (Nevnets, Nfreqs) i.e. transfer functions for TDI channels X, Y, Z
            these values all depend on lm implicitly due to the t-f relations, thus should be calculated per mode
            the data will first be calculated at the frequency grids and  stored in dictionaries.
            
            Transfer functions before tmin [DAY] are set to 0.
        """
        (Nevents, Nfreqs) = t.shape
        t_flatten = t.flatten()
        
        # positions 
        # p1 = self.PositionFunctions['1'](t)
        # p2 = self.PositionFunctions['2'](t)
        # p3 = self.PositionFunctions['3'](t)
        p1 = self.xp.array([
            self.xp.interp(x=t_flatten, xp=self.POS_time_int["1"], fp=self.POS_data_int["1"][:, 0]),
            self.xp.interp(x=t_flatten, xp=self.POS_time_int["1"], fp=self.POS_data_int["1"][:, 1]),
            self.xp.interp(x=t_flatten, xp=self.POS_time_int["1"], fp=self.POS_data_int["1"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        p2 = self.xp.array([
            self.xp.interp(x=t_flatten, xp=self.POS_time_int["2"], fp=self.POS_data_int["2"][:, 0]),
            self.xp.interp(x=t_flatten, xp=self.POS_time_int["2"], fp=self.POS_data_int["2"][:, 1]),
            self.xp.interp(x=t_flatten, xp=self.POS_time_int["2"], fp=self.POS_data_int["2"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        p3 = self.xp.array([
            self.xp.interp(x=t_flatten, xp=self.POS_time_int["3"], fp=self.POS_data_int["3"][:, 0]),
            self.xp.interp(x=t_flatten, xp=self.POS_time_int["3"], fp=self.POS_data_int["3"][:, 1]),
            self.xp.interp(x=t_flatten, xp=self.POS_time_int["3"], fp=self.POS_data_int["3"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        
        # arm lengths 
        # d12 = self.LTTFunctions['12'](t)
        # d13 = self.LTTFunctions['13'](t)
        # d23 = self.LTTFunctions['23'](t)
        # d21 = self.LTTFunctions['21'](t)
        # d31 = self.LTTFunctions['31'](t)
        # d32 = self.LTTFunctions['32'](t)
        d12 = self.xp.interp(x=t_flatten, xp=self.LTT_time_int["12"], fp=self.LTT_data_int["12"]).reshape((Nevents, Nfreqs))
        d13 = self.xp.interp(x=t_flatten, xp=self.LTT_time_int["13"], fp=self.LTT_data_int["13"]).reshape((Nevents, Nfreqs))
        d23 = self.xp.interp(x=t_flatten, xp=self.LTT_time_int["23"], fp=self.LTT_data_int["23"]).reshape((Nevents, Nfreqs))
        d21 = self.xp.interp(x=t_flatten, xp=self.LTT_time_int["21"], fp=self.LTT_data_int["21"]).reshape((Nevents, Nfreqs))
        d31 = self.xp.interp(x=t_flatten, xp=self.LTT_time_int["31"], fp=self.LTT_data_int["31"]).reshape((Nevents, Nfreqs))
        d32 = self.xp.interp(x=t_flatten, xp=self.LTT_time_int["32"], fp=self.LTT_data_int["32"]).reshape((Nevents, Nfreqs))
        
        # arm directions 
        # n12 = self.ArmVectorFunctions['12'](t)
        # n13 = self.ArmVectorFunctions['13'](t)
        # n23 = self.ArmVectorFunctions['23'](t)
        # n21 = self.ArmVectorFunctions['21'](t)
        # n31 = self.ArmVectorFunctions['31'](t)
        # n32 = self.ArmVectorFunctions['32'](t)
        n12 = self.xp.array([
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["12"], fp=self.ARM_data_int["12"][:, 0]),
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["12"], fp=self.ARM_data_int["12"][:, 1]),
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["12"], fp=self.ARM_data_int["12"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        n13 = self.xp.array([
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["13"], fp=self.ARM_data_int["13"][:, 0]),
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["13"], fp=self.ARM_data_int["13"][:, 1]),
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["13"], fp=self.ARM_data_int["13"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        n23 = self.xp.array([
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["23"], fp=self.ARM_data_int["23"][:, 0]),
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["23"], fp=self.ARM_data_int["23"][:, 1]),
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["23"], fp=self.ARM_data_int["23"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        n21 = self.xp.array([
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["21"], fp=self.ARM_data_int["21"][:, 0]),
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["21"], fp=self.ARM_data_int["21"][:, 1]),
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["21"], fp=self.ARM_data_int["21"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        n31 = self.xp.array([
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["31"], fp=self.ARM_data_int["31"][:, 0]),
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["31"], fp=self.ARM_data_int["31"][:, 1]),
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["31"], fp=self.ARM_data_int["31"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        n32 = self.xp.array([
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["32"], fp=self.ARM_data_int["32"][:, 0]),
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["32"], fp=self.ARM_data_int["32"][:, 1]),
            self.xp.interp(x=t_flatten, xp=self.ARM_time_int["32"], fp=self.ARM_data_int["32"][:, 2]),
        ]).transpose().reshape((Nevents, Nfreqs, 3))
        
        # k \dot n 
        ke = self.xp.expand_dims(k, axis=1)
        kn12 =  self.xp.sum(ke * n12, axis=2)
        kn13 =  self.xp.sum(ke * n13, axis=2)
        kn23 =  self.xp.sum(ke * n23, axis=2)
        kn21 =  self.xp.sum(ke * n21, axis=2)
        kn31 =  self.xp.sum(ke * n31, axis=2)
        kn32 =  self.xp.sum(ke * n32, axis=2)
        
        # k \dot (p_receive + p_send)
        kp1p2 = self.xp.sum(ke * (p1 + p2), axis=2)
        kp2p3 = self.xp.sum(ke * (p2 + p3), axis=2)
        kp3p1 = self.xp.sum(ke * (p3 + p1), axis=2)
        
        # prefactor = i*PI*f*d
        prefactor12 = 1.j * PI * f * d12
        prefactor13 = 1.j * PI * f * d13
        prefactor23 = 1.j * PI * f * d23
        prefactor21 = 1.j * PI * f * d21
        prefactor31 = 1.j * PI * f * d31
        prefactor32 = 1.j * PI * f * d32
        
        # sincfactor = sinc(PI*f*d*(1-kn)) = self.xp.sinc(f*d*(1-kn))
        sincfactor12 = self.xp.sinc(f * d12 * (1. - kn12))
        sincfactor13 = self.xp.sinc(f * d13 * (1. - kn13))
        sincfactor23 = self.xp.sinc(f * d23 * (1. - kn23))
        sincfactor21 = self.xp.sinc(f * d21 * (1. - kn21))
        sincfactor31 = self.xp.sinc(f * d31 * (1. - kn31))
        sincfactor32 = self.xp.sinc(f * d32 * (1. - kn32))
        
        # expfactor = exp(i*PI*f*(d+k(p_receive+p_send)))
        expfactor12 = self.xp.exp(1.j * PI * f * (d12 + kp1p2))
        expfactor13 = self.xp.exp(1.j * PI * f * (d13 + kp3p1))
        expfactor23 = self.xp.exp(1.j * PI * f * (d23 + kp2p3))
        expfactor21 = self.xp.exp(1.j * PI * f * (d21 + kp1p2))
        expfactor31 = self.xp.exp(1.j * PI * f * (d31 + kp3p1))
        expfactor32 = self.xp.exp(1.j * PI * f * (d32 + kp2p3))
        
        # antenna pattern functions Fij
        # matmul(n12, Plm) is of shape (Nevents, Nfreqs, 3)
        F12 = self.xp.sum(self.xp.matmul(n12, Plm) * n12, axis=2)
        F13 = self.xp.sum(self.xp.matmul(n13, Plm) * n13, axis=2)
        F23 = self.xp.sum(self.xp.matmul(n23, Plm) * n23, axis=2)
        F21 = self.xp.sum(self.xp.matmul(n21, Plm) * n21, axis=2)
        F31 = self.xp.sum(self.xp.matmul(n31, Plm) * n31, axis=2)
        F32 = self.xp.sum(self.xp.matmul(n32, Plm) * n32, axis=2)
        
        # combine into single-arm transferfunction  Gij
        Gij = {}
        Gij['12'] = prefactor12 * sincfactor12 * expfactor12 * F12
        Gij['13'] = prefactor13 * sincfactor13 * expfactor13 * F13
        Gij['23'] = prefactor23 * sincfactor23 * expfactor23 * F23
        Gij['21'] = prefactor21 * sincfactor21 * expfactor21 * F21
        Gij['31'] = prefactor31 * sincfactor31 * expfactor31 * F31
        Gij['32'] = prefactor32 * sincfactor32 * expfactor32 * F32
        
        # delay factors 
        D12 = self.xp.exp(1.j * TWOPI * f * d12) 
        D13 = self.xp.exp(1.j * TWOPI * f * d13) 
        D23 = self.xp.exp(1.j * TWOPI * f * d23) 
        D21 = self.xp.exp(1.j * TWOPI * f * d21) 
        D31 = self.xp.exp(1.j * TWOPI * f * d31) 
        D32 = self.xp.exp(1.j * TWOPI * f * d32) 
        
        # combine into TDI transferfunction GTDI
        if TDIGeneration == '1st':
            X1plusfactor = (1. - D12 * D21) 
            X1minusfactor = (1. - D13 * D31) 
            Y1plusfactor = (1. - D23 * D32) 
            Y1minusfactor = (1. - D21 * D12) 
            Z1plusfactor = (1. - D31 * D13) 
            Z1minusfactor = (1. - D32 * D23) 
            GTDI = {} 
            GTDI['X'] = X1plusfactor * (Gij['13'] + Gij['31'] * D13) - X1minusfactor * (Gij['12'] + Gij['21'] * D12)
            GTDI['Y'] = Y1plusfactor * (Gij['21'] + Gij['12'] * D21) - Y1minusfactor * (Gij['23'] + Gij['32'] * D23)
            GTDI['Z'] = Z1plusfactor * (Gij['32'] + Gij['23'] * D32) - Z1minusfactor * (Gij['31'] + Gij['13'] * D31)
            
        elif TDIGeneration == '2nd':
            D121 = D12 * D21
            D131 = D13 * D31 
            D12131_1 = 1. - D121 * D131 # 1 - D12131
            X2plusfactor = (1. - D121) * D12131_1
            X2minusfactor = (1. - D131) * D12131_1
            D232 = D23 * D32 
            D212 = D21 * D12 
            D23212_1 = 1. - D232 * D212 
            Y2plusfactor = (1. - D232) * D23212_1
            Y2minusfactor = (1. - D212) * D23212_1
            D313 = D31 * D13 
            D323 = D32 * D23 
            D31323_1 = 1. - D313 * D323
            Z2plusfactor = (1. - D313) * D31323_1
            Z2minusfactor = (1. - D323) * D31323_1
            GTDI = {} 
            GTDI['X'] = X2plusfactor * (Gij['13'] + Gij['31'] * D13) - X2minusfactor * (Gij['12'] + Gij['21'] * D12)
            GTDI['Y'] = Y2plusfactor * (Gij['21'] + Gij['12'] * D21) - Y2minusfactor * (Gij['23'] + Gij['32'] * D23)
            GTDI['Z'] = Z2plusfactor * (Gij['32'] + Gij['23'] * D32) - Z2minusfactor * (Gij['31'] + Gij['13'] * D31)
            
        if tmin != None:
            zero_inds = self.xp.where(t < tmin * DAY)
            for k in Gij.keys():
                Gij[k][zero_inds] = 0.
            for k in GTDI.keys():
                GTDI[k][zero_inds] = 0.
                
        if tmax != None:
            zero_inds = self.xp.where(t > tmax * DAY)
            for k in Gij.keys():
                Gij[k][zero_inds] = 0.
            for k in GTDI.keys():
                GTDI[k][zero_inds] = 0.
        
        return Gij, GTDI
    
    
    def SSBToConstellationDelay(self, k, parameters):
        """  
            returns the delay of coalescence time at constellation center w.r.t SSB 
            tc = tc^SSB + tc^delay --> tc^SSB = tc - tc^delay
        """
        tc_SI = parameters['coalescence_time'] * DAY # (Nevent)
        p0 = self.xp.transpose(self.xp.array([
            self.xp.interp(x=tc_SI, xp=self.POS0_time_int, fp=self.POS0_data_int[:, 0]),
            self.xp.interp(x=tc_SI, xp=self.POS0_time_int, fp=self.POS0_data_int[:, 1]),
            self.xp.interp(x=tc_SI, xp=self.POS0_time_int, fp=self.POS0_data_int[:, 2]),
        ])) # (3, Nevent) -> (Nevent, 3), in second unit 
        return self.xp.sum(k * p0, axis=1) # (Nevent), in second unit 
    
    def Response(
        self, 
        parameters, 
        freqs, 
        modes=[(2, 2), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3)], 
        tmin=None, 
        tmax=None, 
        tc_at_constellation=False, 
        TDIGeneration='2nd',
        optimal_combination=True,
        output_by_mode=False, 
        interpolation=False, 
        interpolate_points=1024, 
        ):
        """   
            Calculate 1st/2nd-generation TDI response in the frequency domain.
            Work flow:
            frequency grids, Waveform, time grids 
            --> PolarBasis_lm, WaveVector 
            --> TDI TransferFunction 
            --> TDI response
            Args: 
                parameters: dictionary of parameters, each item can be either a float or a numpy array 
                freqs: frequencies where the responses are calculated, xp array of shape (Nfreqs) or (Nevents, Nfreqs), if interpolation=True, freqs should be of shape (Nfreqs)
                modes: list of the modes calculated 
                tmin, tmax: minimum / maximum time of waveform in [day] 
                tc_at_constellation: whether tc is the coalescence time at constellation center (True) or SSB (False)
                TDIGeneration: "1st" or "2nd", standing for the 1st or 2nd generation Michelson channels, respectively 
                optimal_combination: True for AET, False for XYZ
                output_by_mode: whether to return the waveform summed over modes, or return modes separately 
                interpolation: whether to calculate on orignal frequency grid or interpolate from a sparse grid 
                interpolate_points: number of frequencies in the sparse grid 
        """
        
        # calculate frequency grids (mode-independent), waveforms and time grids (mode-dependent)
        if interpolation: 
            freqs_sparse = self.xp.logspace(self.xp.log10(freqs[0]) - 1e-1, self.xp.log10(freqs[-1]) + 1e-1, interpolate_points) # (1024,)
            fgrids, amps, phas, tgrids = self.waveform(parameters=parameters, freqs=freqs_sparse)
        else: 
            fgrids, amps, phas, tgrids = self.waveform(parameters=parameters, freqs=freqs)
        
        # convert scalar parameters to arraies 
        parameter_dict = parameters.copy()
        for k, v in parameters.items():
            parameter_dict[k] = self.xp.atleast_1d(v)
        
        Nevents = parameter_dict['chirp_mass'].shape[0]
        
        # calculate polarization tensors (mode-dependent) and wavevectors (mode-independent)
        Plm = self.PolarBasis_lm(parameters=parameter_dict, modes=modes) 
        k = self.WaveVector(parameters=parameter_dict) 
        
        # convert the coalescence time from constellaton center to SSB
        if tc_at_constellation:
            tc_delay = self.SSBToConstellationDelay(k, parameter_dict) # (Nevent)
            parameter_dict['coalescence_time'] += -tc_delay / DAY 

        # calculate transfer function at the frequency and time grids and then interpolate (Nevents, Nfreqs) --> (Nevents, Nfreqs_out)
        Nfreqs_out = freqs.shape[-1]
        if output_by_mode:
            Nmode = len(modes)
            X = self.xp.zeros((Nmode, Nevents, Nfreqs_out), dtype=self.xp.complex128)
            Y = self.xp.zeros((Nmode, Nevents, Nfreqs_out), dtype=self.xp.complex128)
            Z = self.xp.zeros((Nmode, Nevents, Nfreqs_out), dtype=self.xp.complex128)
            if interpolation: 
                for imode, mode in enumerate(modes):
                    _, GTDI = self.TransferFunction(t=tgrids[mode], f=fgrids, k=k, Plm=Plm[mode], TDIGeneration=TDIGeneration, tmin=tmin, tmax=tmax)
                    Xamp_int = GTDI['X'] * amps[mode]
                    Yamp_int = GTDI['Y'] * amps[mode]
                    Zamp_int = GTDI['Z'] * amps[mode]
                    phase_int = phas[mode]
                    Xamp_func = self.xinterp.Akima1DInterpolator(x=fgrids[0], y=Xamp_int, axis=1) 
                    Yamp_func = self.xinterp.Akima1DInterpolator(x=fgrids[0], y=Yamp_int, axis=1)
                    Zamp_func = self.xinterp.Akima1DInterpolator(x=fgrids[0], y=Zamp_int, axis=1)
                    phase_func = self.xinterp.Akima1DInterpolator(x=fgrids[0], y=phase_int, axis=1)
                    Xamp = Xamp_func(freqs)
                    Yamp = Yamp_func(freqs)
                    Zamp = Zamp_func(freqs)
                    phase = phase_func(freqs)
                    X[imode] = Xamp * self.xp.exp(1.j * phase)
                    Y[imode] = Yamp * self.xp.exp(1.j * phase)
                    Z[imode] = Zamp * self.xp.exp(1.j * phase) 

            else: 
                for imode, mode in enumerate(modes):
                    _, GTDI = self.TransferFunction(t=tgrids[mode], f=fgrids, k=k, Plm=Plm[mode], TDIGeneration=TDIGeneration, tmin=tmin, tmax=tmax)
                    Xamp_int = GTDI['X'] * amps[mode]
                    Yamp_int = GTDI['Y'] * amps[mode]
                    Zamp_int = GTDI['Z'] * amps[mode]
                    phase_int = phas[mode]
                    X[imode] = Xamp_int * self.xp.exp(1.j * phase_int)
                    Y[imode] = Yamp_int * self.xp.exp(1.j * phase_int)
                    Z[imode] = Zamp_int * self.xp.exp(1.j * phase_int) 
        else:
            X = self.xp.zeros((Nevents, Nfreqs_out), dtype=self.xp.complex128)
            Y = self.xp.zeros((Nevents, Nfreqs_out), dtype=self.xp.complex128)
            Z = self.xp.zeros((Nevents, Nfreqs_out), dtype=self.xp.complex128)
            if interpolation: 
                for mode in modes:
                    _, GTDI = self.TransferFunction(t=tgrids[mode], f=fgrids, k=k, Plm=Plm[mode], TDIGeneration=TDIGeneration, tmin=tmin, tmax=tmax)
                    Xamp_int = GTDI['X'] * amps[mode]
                    Yamp_int = GTDI['Y'] * amps[mode]
                    Zamp_int = GTDI['Z'] * amps[mode]
                    phase_int = phas[mode]

                    # Xamp_func = self.xinterp.Akima1DInterpolator(x=fgrids[0], y=Xamp_int, axis=1)
                    # Yamp_func = self.xinterp.Akima1DInterpolator(x=fgrids[0], y=Yamp_int, axis=1)
                    # Zamp_func = self.xinterp.Akima1DInterpolator(x=fgrids[0], y=Zamp_int, axis=1)
                    # phase_func = self.xinterp.Akima1DInterpolator(x=fgrids[0], y=phase_int, axis=1)
                    # Xamp = Xamp_func(freqs)
                    # Yamp = Yamp_func(freqs)
                    # Zamp = Zamp_func(freqs)
                    # phase = phase_func(freqs)

                    Xamp_func_re = self.xinterp.Akima1DInterpolator(x=fgrids[0], y=self.xp.real(Xamp_int), axis=1)
                    Yamp_func_re = self.xinterp.Akima1DInterpolator(x=fgrids[0], y=self.xp.real(Yamp_int), axis=1)
                    Zamp_func_re = self.xinterp.Akima1DInterpolator(x=fgrids[0], y=self.xp.real(Zamp_int), axis=1)
                    Xamp_func_im = self.xinterp.Akima1DInterpolator(x=fgrids[0], y=self.xp.imag(Xamp_int), axis=1)
                    Yamp_func_im = self.xinterp.Akima1DInterpolator(x=fgrids[0], y=self.xp.imag(Yamp_int), axis=1)
                    Zamp_func_im = self.xinterp.Akima1DInterpolator(x=fgrids[0], y=self.xp.imag(Zamp_int), axis=1)
                    phase_func = self.xinterp.Akima1DInterpolator(x=fgrids[0], y=phase_int, axis=1)
                    Xamp = Xamp_func_re(freqs) + Xamp_func_im(freqs) * 1.j 
                    Yamp = Yamp_func_re(freqs) + Yamp_func_im(freqs) * 1.j 
                    Zamp = Zamp_func_re(freqs) + Zamp_func_im(freqs) * 1.j 
                    phase = phase_func(freqs)

                    X += Xamp * self.xp.exp(1.j * phase)
                    Y += Yamp * self.xp.exp(1.j * phase)
                    Z += Zamp * self.xp.exp(1.j * phase)
            else: 
                for mode in modes:
                    _, GTDI = self.TransferFunction(t=tgrids[mode], f=fgrids, k=k, Plm=Plm[mode], TDIGeneration=TDIGeneration, tmin=tmin, tmax=tmax)
                    Xamp_int = GTDI['X'] * amps[mode]
                    Yamp_int = GTDI['Y'] * amps[mode]
                    Zamp_int = GTDI['Z'] * amps[mode]
                    phase_int = phas[mode]
                    X += Xamp_int * self.xp.exp(1.j * phase_int)
                    Y += Yamp_int * self.xp.exp(1.j * phase_int)
                    Z += Zamp_int * self.xp.exp(1.j * phase_int)
        if Nevents == 1:
            X = self.xp.conjugate(X[0])
            Y = self.xp.conjugate(Y[0])
            Z = self.xp.conjugate(Z[0])
        else:
            X = self.xp.conjugate(X)
            Y = self.xp.conjugate(Y)
            Z = self.xp.conjugate(Z)
        
        if optimal_combination:
            A, E, T = self.AETfromXYZ(X, Y, Z)
            results = self.xp.array([A, E, T]) 
        else:
            results = self.xp.array([X, Y, Z])
        
        results[self.xp.abs(results)<1e-23]=0.
        return results 
        
    def AETfromXYZ(self, X, Y, Z):
        A = (Z - X) / self.xp.sqrt(2.)
        E = (X - 2. * Y + Z) / self.xp.sqrt(6.)
        T = (X + Y + Z) / self.xp.sqrt(3.)
        return A, E, T

    def XYZfromAET(self, A, E, T):
        X = (-self.xp.sqrt(3.) * A + E + self.xp.sqrt(2.) * T) / self.xp.sqrt(6.)
        Y = (T - self.xp.sqrt(2.) * E) / self.xp.sqrt(3.)
        Z = (self.xp.sqrt(3.) * A + E + self.xp.sqrt(2.) * T) / self.xp.sqrt(6.)
        return X, Y, Z
            
            
        
        
        
        