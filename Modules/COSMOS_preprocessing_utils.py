import numpy as np
import pandas as pd

import galsim
from coord import radians

import Image_Fits_Stats

import pywt
from scipy.stats import norm

#Galsim arguments
target_size=64
galaxy_type='real'
psf_type='real'

cat = galsim.COSMOSCatalog(sample='23.5')
#Pixel scale in arcsec (the same for all the galaxies)
_,_,_,pixel_scale,_=cat.getRealParams(0)


#Obtain galsim.GSObject for a given galaxy
#with noise,ellipticity and rotation angle
def get_COSMOS_galaxy(index):
    #Making galaxy
    gal=cat.makeGalaxy(index,gal_type=galaxy_type)

    #Get the PSF object
    psf = gal.original_psf
    #Perform convolution with PSF to get the original HST image
    gal=galsim.Convolve(gal, psf)

    return gal

#Rotate GSObject
def rotate_galaxy(gal,angle):

    # gal.rotate turns anticlockwise, therefore we need to pass -angle
    rotation_angle=galsim.Angle(-angle*radians)

    rotated_gal=gal.rotate(rotation_angle)

    return rotated_gal


#Our goal is to transform the galaxy such that galaxy_size->target_size
#However, we only use downscaling, so if initial galaxy_size<target_size
#we leave the image as it is
def get_downscale_factor(image_size,galaxy_size,target_size):

    if target_size<galaxy_size:
        #We want to downscale image,so galaxy_size->target_size

        #The scale may seem strange but Galsim usually works in terms Galsim_size=Image_size-2
        Galsim_scale=pixel_scale*(galaxy_size-2)/(target_size-2)
        Scaling_factor=target_size/galaxy_size

        return Galsim_scale,Scaling_factor


    elif target_size<=image_size:
        #Here image_size≥target_size≥galaxy_size
        #We don't want to upscale anything which is needed for
        #It literally means that no downscaling is needed
        #hence return the original values

        return pixel_scale,1

    else:
        #The original image is smaller than the desired one.
        #We don't want to upscale anything or use padding
        #So this galaxy doesn't suit our needs

        return False,False


#Using elliptical representation of galaxy extract galaxy-free regions
#That we can use for noise estimation
def get_Background_image(image,R_cut,Ellipticity,Angle):

    (y_size,x_size)=image.shape
    R_cut=np.minimum(R_cut,np.minimum(y_size,x_size)//2)

    #oblique parallelepiped inscribed in a vertical parallelepiped
    galaxy_x_size=R_cut*np.cos(Angle)+Ellipticity*R_cut*np.abs(np.sin(Angle))
    galaxy_y_size=R_cut*np.abs(np.sin(Angle))+Ellipticity*R_cut*np.cos(Angle)

    #Offsets from image borders that cover noise regions
    Bkg_x_offset=np.floor(x_size//2-galaxy_x_size).astype(int)
    Bkg_y_offset=np.floor(y_size//2-galaxy_y_size).astype(int)

    #Understand whether we should use horizontally aligned background
    #Or vertically aligned
    Horizontal_lines_area=2*Bkg_y_offset*x_size
    Vertical_lines_area=2*Bkg_x_offset*y_size

    if Vertical_lines_area>Horizontal_lines_area:
        Bkg_image=np.zeros((y_size,2*Bkg_x_offset))

        #Left line
        Bkg_image[:,:Bkg_x_offset]=image[:,:Bkg_x_offset]
        #Right line
        Bkg_image[:,-Bkg_x_offset:]=image[:,-Bkg_x_offset:]


    elif Horizontal_lines_area!=0:
        Bkg_image=np.zeros((2*Bkg_y_offset,x_size))

        #Upper line
        Bkg_image[:Bkg_y_offset,:]=image[:Bkg_y_offset,:]
        #Lower line
        Bkg_image[-Bkg_y_offset:,:]=image[-Bkg_y_offset:,:]


    else:
        return False,0

    return True,Bkg_image

'''
skimage.restoration.estimate_sigma
[1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation 
by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       :DOI:`10.1093/biomet/81.3.425`
'''
def estimate_sigma(Background_image):

    coeffs = pywt.dwtn(Background_image, wavelet='db2')
    detail_coeffs = coeffs['d' * Background_image.ndim]

    # Consider regions with detail coefficients exactly zero to be masked out
    detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]

    # 75th quantile of the underlying, symmetric noise distribution
    denom = norm.ppf(0.75)
    return np.median(np.abs(detail_coeffs)) / denom

def create_Parametric_description(index):

    #Create description of galaxy features
    parameters=pd.Series(cat.getParametricRecord(index))[['IDENT', 'mag_auto',  'zphot']]
    parameters['COSMOS_index']=index
    #Extract Galaxy fits, namely Sersic and Double Sersic (bulge+disk)
    parameters['COSMOS_use_bulgefit']= cat.getParametricRecord(index)['use_bulgefit']

    #Drop Boxiness from Sersic
    COSMOS_Sersic=np.delete(pd.Series(cat.getParametricRecord(index))['sersicfit'],4)
    COSMOS_Sersic=pd.Series(data=COSMOS_Sersic,index=['COSMOS_Sersic_I','COSMOS_Sersic_HLR','COSMOS_Sersic_n','COSMOS_Sersic_q',
                                                      'COSMOS_Sersic_x0','COSMOS_Sersic_y0','COSMOS_Sersic_phi'])

    #Drop Boxiness from Bulge and Disk
    COSMOS_Bulge_Disk=np.delete(pd.Series(cat.getParametricRecord(index))['bulgefit'],[4,12])
    COSMOS_Disk=pd.Series(data=COSMOS_Bulge_Disk[:7],
                            index=['COSMOS_Disk_I','COSMOS_Disk_HLR','COSMOS_Disk_n','COSMOS_Disk_q',
                'COSMOS_Disk_x0','COSMOS_Disk_y0','COSMOS_Disk_phi'])
    COSMOS_Bulge=pd.Series(data=COSMOS_Bulge_Disk[7:],
                            index=['COSMOS_Bulge_I','COSMOS_Bulge_HLR','COSMOS_Bulge_n','COSMOS_Bulge_q',
                'COSMOS_Bulge_x0','COSMOS_Bulge_y0','COSMOS_Bulge_phi'])

    #Get Galaxy's noise
    _,_,_,_,COSMOS_noise_variance=cat.getRealParams(index)
    parameters['COSMOS_noise']=np.sqrt(COSMOS_noise_variance)

    parameters=parameters.append(COSMOS_Sersic)
    parameters=parameters.append(COSMOS_Bulge)
    parameters=parameters.append(COSMOS_Disk)

    return parameters

def create_Galaxy(index,Radial_profile_threshold=0.05):

    parameters=create_Parametric_description(index)

    #Get GSObject and noise
    gal=get_COSMOS_galaxy(index)
    original_image=gal.drawImage(use_true_center=True, method='auto').array

    try:
        #Extracting the parameters to build elliptical coordinates
        Sersic_fit,_=Image_Fits_Stats.fit_image(original_image,noise_std=parameters['COSMOS_noise'],exposure_time=2028)
        Ellipticity,x0,y0,Angle=Sersic_fit[-4:]
        Radial_profile=Image_Fits_Stats.Radial_profile(original_image,Ellipticity,x0,y0,Angle)

        #Getting border of galaxy from radial profile
        Radial_profile[-1]=np.nan_to_num(Radial_profile[-1])
        Radial_profile=Radial_profile-Radial_profile[-1]
        Radial_profile/=np.nanmax(Radial_profile)
        R_cut=np.where((Radial_profile<Radial_profile_threshold))[0][0]
    except:
        print('Unsuccesfull original image processing')

        #Return False for Success
        return False,0,0

    parameters['Original_R_cut']=R_cut

    #Our target is to bring 2*R_cut->target_size, which is 64
    Galsim_scale,Scaling_factor=get_downscale_factor(original_image.shape[1],2*R_cut,target_size)

    #If original image is smaller than the desired one
    if np.logical_not(Galsim_scale):
        return False

    #Account for scale change
    R_cut=R_cut*Scaling_factor

    image_scaled=gal.drawImage(scale=Galsim_scale,use_true_center=True, method='auto').array

    #Extract galaxy-free regions merged together to form an image of background
    Bkg_extraction_success,Background_image=get_Background_image(image_scaled,R_cut,Ellipticity,Angle)
    #Estimate noise from background image
    if Bkg_extraction_success:
        noise_sigma=estimate_sigma(Background_image)
        noise_median=np.median(Background_image)
    else:
        noise_sigma=parameters['COSMOS_noise']
        noise_median=0


    rotated_gal=rotate_galaxy(gal,Angle)
    image_rotated=rotated_gal.drawImage(scale=Galsim_scale,use_true_center=True, method='auto').array

    #Perform the crop
    target_radius=target_size//2
    x0=image_scaled.shape[1]//2
    y0=image_scaled.shape[0]//2
    image_64x64=image_rotated[y0-target_radius:y0+target_radius,x0-target_radius:x0+target_radius]

    Additional_parameters=pd.Series(data=[original_image.shape[1],original_image.shape[0],Background_image.shape[1],Background_image.shape[0],
                                          Scaling_factor,R_cut,noise_median,noise_sigma,Ellipticity,Angle,image_64x64.max(),image_64x64.min()],
                                    index=['Original_x_size','Original_y_size','Background_x_size','Background_y_size',
                                           'Scaling_factor','R_cut','Noise_median','Noise_sigma','Ellipticity','Angle','max_I','min_I'])
    parameters=parameters.append(Additional_parameters)

    #return True for Success
    return True,image_64x64,parameters

def merge_files(Folder,Name_Images,Name_Labels,split_array):
  images=np.load(Folder+Name_Images+'_{}_{}.npy'.format(split_array[0][0],split_array[0][1]))
  labels=pd.read_csv(Folder+Name_Labels+'_{}_{}.csv'.format(split_array[0][0],split_array[0][1]),index_col=0)
  for i,split in enumerate(split_array[1:]):
    start,stop=split
    images_t=np.load(Folder+Name_Images+'_{}_{}.npy'.format(start,stop))
    labels_t=pd.read_csv(Folder+Name_Labels+'_{}_{}.csv'.format(start,stop),index_col=0)

    images=np.append(images,images_t,axis=0)
    labels=labels.append(labels_t)

  return images,labels
