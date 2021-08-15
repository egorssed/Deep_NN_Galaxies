import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import Image_Fits_Stats

def Show_stats(images,df,column,min_value,max_value,dimensions=2,invert_colors=False,cut_negatives=False,log_hist=False,cmap='viridis'):
    #Convenient function to observe impact of feature on image
    #For a given limited feature shows histogram of this feature, best and worst galaxy images
    fig=plt.figure(figsize=(20,5))
    plt.subplot(131)

    #Extract galaxies according to feature limits
    df_filt=df[(df[column].values>=min_value) & (df[column].values<=max_value)]
    min_mask=df_filt[column]==df_filt[column].min()
    max_mask=df_filt[column]==df_filt[column].max()
    plt.hist(df_filt[column],bins=100)
    if log_hist:
      plt.xscale('log')
      plt.yscale('log')
    plt.title(column+' histogram')

    #Number of galaxies left for given limits
    print(df_filt.shape[0], 'Galaxies')
    print((column + ' [{:.2f},{:.2f}]').format(min_value,max_value))

    if dimensions==2:
      #Images of best and worst galaxies in terms of the given feature
      i=(df_filt[min_mask].index)[0]
      image=images[i]
      plt.subplot(132)
      plt.title(column+'={:.2f}'.format(df_filt[column].min()))
      if invert_colors:
        plt.imshow(1-image,cmap=cmap)
      else:
        plt.imshow(image,cmap=cmap)
      plt.axis('off')

      i=(df_filt[max_mask].index)[0]
      image=images[i]
      plt.subplot(133)
      plt.title(column+'={:.2f}'.format(df_filt[column].max()))
      if invert_colors:
        plt.imshow(1-image,cmap=cmap)
      else:
        plt.imshow(image,cmap=cmap)
      plt.axis('off')

      plt.tight_layout()
    elif dimensions==3:
      X = np.arange(-32, 32, 1)
      Y = np.arange(-32, 32, 1)
      X, Y = np.meshgrid(X, Y)

      i=(df_filt[min_mask].index)[0]
      image=copy.deepcopy(images[i])
      if cut_negatives:
          #noise=df.iloc[i]['Noise_mean']
          #image-=noise
          image[image<0]=0
      ax = fig.add_subplot(132, projection='3d')
      ax.plot_surface(X, Y, image/image.max(), cmap=cm.coolwarm,
                          linewidth=0, antialiased=False)
      ax.set_title(column+'={:.2f}'.format(df_filt[column].min()))
      ax.set_xticks([])
      ax.set_yticks([])

      i=(df_filt[max_mask].index)[0]
      image=copy.deepcopy(images[i])
      if cut_negatives:
          #noise=df.iloc[i]['Noise_mean']
          #image-=noise
          image[image<0]=0
      ax = fig.add_subplot(133, projection='3d')
      ax.plot_surface(X, Y, image/image.max(), cmap=cm.coolwarm,
                          linewidth=0, antialiased=False)
      ax.set_title(column+'={:.2f}'.format(df_filt[column].max()))
      ax.set_xticks([])
      ax.set_yticks([])

    else:
      print('wrong dimensions')

def Show_100(images,df,invert_colors=False,cut_negatives=False,cmap='viridis'):
    #Show images of 100 first galaxies in the dataframe
    #It is useful if dataframe is sorted in a proper way
    plt.figure(figsize=(20,20))
    gal_to_see=images[df.index[:100].astype(int)]
    for i in range(len(gal_to_see)):
        x=i//10
        y=np.mod(i,10)
        ax = plt.subplot2grid((10,10), (x,y))
        image=copy.deepcopy(gal_to_see[i])
        if cut_negatives:
          #noise=df.iloc[i]['Noise_mean']
          #image-=noise
          image[image<0]=0
        if invert_colors:
          ax.imshow(image,cmap=cmap)
        else:
          ax.imshow(image,cmap=cmap)
        #,cmap='gray_r'
        ax.axis('off')
    plt.show()

def Show_filtered(images,df_fit,Column,ascending=True,threshold=0,Index_from=0,invert_colors=False,cut_negatives=False,cmap='viridis'):
  df_filt=df_fit.sort_values(by=[Column],ascending=ascending)
  if ascending:
    df_filt=df_filt[df_filt[Column]>=threshold].iloc[Index_from:]
  else:
    if threshold==0:
      threshold=df_filt[Column].max()
    df_filt=df_filt[df_filt[Column]<=threshold].iloc[Index_from:]
  print(Column,df_filt.iloc[0][Column])
  print('Number of galaxies',len(df_filt))
  Show_100(images,df_filt,invert_colors,cut_negatives,cmap)
  return df_filt

from matplotlib import cm

def present_image(image,Sersic_fit):
  X = np.arange(0, image.shape[0], 1)
  Y = np.arange(0, image.shape[1], 1)
  X, Y = np.meshgrid(X, Y)

  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  surf = ax.plot_surface(X, Y, image, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
  ax.set_xticks([])
  ax.set_yticks([])

  fig_fit, ax_fit = plt.subplots(subplot_kw={"projection": "3d"})
  xdata=np.vstack((X.ravel(), Y.ravel()))
  surf_fit = ax_fit.plot_surface(X, Y, Image_Fits_Stats.Sersic_profile(xdata,*Sersic_fit).reshape((64,64)), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
  ax_fit.set_xticks([])
  ax_fit.set_yticks([])

  plt.figure(3)
  plt.imshow(image)
  plt.show()


