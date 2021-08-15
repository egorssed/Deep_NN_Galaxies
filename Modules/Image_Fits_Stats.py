import numpy as np
from scipy.optimize import curve_fit
import copy

#See Lackner & Gunn 2012 for details on Sersic profile
def Sersic_profile(M,I,HLR,n,q,x0,y0,phi):
        (x, y) = M
        R=np.sqrt(np.power((x-x0)*np.cos(phi)+(y-y0)*np.sin(phi),2)+
                np.power((y-y0)*np.cos(phi)-(x-x0)*np.sin(phi),2)/np.power(q,2))

        #correct coefficient from Ciotti 1991
        k=2*n-0.324
        S=I*np.exp(-k*(np.power(R/HLR,1/n)-1))
        return S

#See S. H. Suyu et al 2014 ApJL 788 L35
def Chameleon_profile(M,I,Wt,Wratio,q,x0,y0,phi):
  (x, y) = M
  Wc=Wt*Wratio
  Rsq=np.power((x-x0)*np.cos(phi)+(y-y0)*np.sin(phi),2)+np.power((y-y0)*np.cos(phi)-(x-x0)*np.sin(phi),2)/np.power(q,2)

  def Isothermal_profile(w):
    Softening=np.power(2*w/(1+q),2)
    return np.power(Rsq+Softening,-0.5)

  return I/(1+q)*(Isothermal_profile(Wc)-Isothermal_profile(Wt))

# Combination Chameleon+Sersic
def Chameleon_Sersic(M,I1,Wt1,Wratio1,q1,phi1,I2,HLR,n,q2,phi2,x0,y0):
    return Chameleon_profile(M,I1,Wt1,Wratio1,q1,x0,y0,phi1)+Sersic_profile(M,I2,HLR,n,q2,x0,y0,phi2)

# Combination Sersic+Sersic
def Double_Sersic(M,I1,HLR1,n1,q1,phi1,I12ratio,HLR2,n2,q2,phi2,x0,y0):
    I2=I1*I12ratio
    return Sersic_profile(M,I1,HLR1,n1,q1,x0,y0,phi1)+Sersic_profile(M,I2,HLR2,n2,q2,x0,y0,phi2)

#Exposure time for COSMOS is 2028 (https://iopscience.iop.org/article/10.1086/520086/pdf)
def fit_image(image,profile_type='Sersic',noise_std=None,exposure_time=None):
    x = np.linspace(0, image.shape[1], image.shape[1])
    y = np.linspace(0, image.shape[0], image.shape[0])
    X, Y = np.meshgrid(x, y)
    # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    xdata = np.vstack((X.ravel(), Y.ravel()))

    if profile_type=='Sersic':
        func=Sersic_profile
        n_initial=1
        k=2*n_initial-0.324
        initial_guess=np.array([image.max()*np.exp(-k),5,n_initial,0.64,
                            image.shape[1]/2,image.shape[0]/2,0])
        lower_bounds=np.array([image.max()*1e-5,0,0.5,0.01,0,0,-np.pi/2])
        upper_bounds=np.array([image.max(),np.min(image.shape)/2,6,1,
                           image.shape[1],image.shape[0],np.pi/2])

    elif profile_type=='Chameleon':
        func=Chameleon_profile
        initial_guess=np.array([image.max(),5,0.5,0.5,image.shape[1]/2,image.shape[0]/2,0])
        lower_bounds=np.array([image.max()*1e-5,0,0,0.01,0,0,-np.pi/2])
        upper_bounds=np.array([np.inf,np.inf,1,1,image.shape[1],image.shape[0],np.pi/2])

    elif profile_type=='Chameleon_Sersic':
        func=Chameleon_Sersic
        n_initial=1
        k=2*n_initial-0.324
        initial_guess=np.array([image.max()*20,2,0.1,0.64,0,
                             image.max()*np.exp(-k),5,n_initial,0.64,0,
                             image.shape[1]/2,image.shape[0]/2])
        lower_bounds=np.array([image.max()*1e-5,0,0,0.01,-np.pi/2,
                           0,0,0.5,0.01,-np.pi/2,
                           0,0])
        upper_bounds=np.array([np.inf,np.inf,1,1,np.pi/2,
                           image.max(),np.min(image.shape)/2,6,1,np.pi/2,
                           image.shape[1],image.shape[0]])
    elif profile_type=='Double_Sersic':
        func=Double_Sersic
        n_initial=np.array([4,1])
        k=2*n_initial-0.324
        Iratio_init=np.exp(-k[0]+k[1])
        initial_guess=np.array([image.max()*np.exp(-k[0]),2,n_initial[0],0.64,0,
                             Iratio_init,5,n_initial[1],0.64,0,
                             image.shape[1]/2,image.shape[0]/2])
        lower_bounds=np.array([image.max()*1e-5,0,0.5,0.01,-np.pi/2,
                           0,0,0.5,0.01,-np.pi/2,
                           0,0])
        upper_bounds=np.array([image.max(),np.min(image.shape)/2,6,1,np.pi/2,
                           1,np.min(image.shape)/2,6,1,np.pi/2,
                           image.shape[1],image.shape[0]])

    else:
        print('Wrong profile type')
        return False

    #assume that exposure time is 1
    #Poisson_errors=np.sqrt(np.abs(image))
    #0.0244 is a median MAD extracted relative gaussian noise sigma of the COSMOS dataset
    #See "Preformance test" notebook
    #Gaussian_errors=np.sqrt(image.max()*0.0244)
    sigma=np.sqrt(np.abs(image)+image.max()*1e-5)
    if exposure_time and noise_std:
        Poisson_errors=copy.deepcopy(image/exposure_time)
        Poisson_errors[Poisson_errors<0]=0
        Gaussian_errors=noise_std
        sigma=np.sqrt(np.power(Poisson_errors,2)+np.power(Gaussian_errors,2))

    popt, pcov = curve_fit(f=func, xdata=xdata,
                               ydata=image.ravel(),
                               p0=initial_guess,
                               sigma=sigma.ravel(),
                               bounds=(lower_bounds,upper_bounds))
    return popt,np.sqrt(np.diag(pcov))

def chi_sq(image_true,image_pred,exposure,noise):
    Poisson_variance=image_true/exposure
    Poisson_variance[Poisson_variance<0]=0
    Gaussian_variance=np.power(noise,2)

    Variance=Poisson_variance*np.ones_like(image_true)+Gaussian_variance*np.ones_like(image_true)

    #Chi_square in form of image_true
    return np.power(image_true-image_pred,2)/Variance



def Radial_profile(image,q=1,x0=None,y0=None,phi=0):
    if x0 is None or y0 is None:
        x0=image.shape[1]/2
        y0=image.shape[0]/2


    R_max=np.min(image.shape)//2
    radial_profile=np.zeros(R_max)
    counter=np.zeros(R_max)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            R=np.sqrt(np.power((x-x0)*np.cos(phi)+(y-y0)*np.sin(phi),2)+
                np.power((y-y0)*np.cos(phi)-(x-x0)*np.sin(phi),2)/np.power(q,2))
            if(R<R_max):
                radial_profile[int(R)]+=image[y,x]
                counter[int(R)]+=1

    mask_nan=(counter==0)
    counter[mask_nan]=np.nan
    radial_profile=radial_profile/counter
    #radial_profile[mask_nan]=0

    #Radial profile is average Flux in a ring of radius R
    return radial_profile

