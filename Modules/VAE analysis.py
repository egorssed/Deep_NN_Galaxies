import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import keras.backend as K
import tensorflow as tf
import pandas as pd
from sklearn.decomposition import PCA


image_size=64
x = np.linspace(0, 64, 64)
y = np.linspace(0, 64, 64)
X, Y = np.meshgrid(x, y)
xdata = np.vstack((X.ravel(), Y.ravel()))

#Images from one arg array are plotted in a row
#N args- N rows
def plot_galaxies(*args,dimensions='2d',colorbar=False):
    #(num_of_rows,num_of_cols,im_shape[0],im_shape[1])
    args = [x.squeeze() for x in args]
    #num of cols
    n = min([x.shape[0] for x in args])
    I_max=np.max(args)
    print('Maximal brightness',I_max)

    if dimensions=='2d':
      fig=plt.figure(figsize=(2*n, 2*len(args)))
    else:
      fig=plt.figure(figsize=(5*n, 5*len(args)))

    for row in range(len(args)):
        for col in range(n):
            if dimensions=='2d':
                ax = fig.add_subplot(len(args),n,row*n+col+1)
                ax.imshow(args[row][col].squeeze(),cmap='Greys_r',vmax=I_max)
            else:
                ax = fig.add_subplot(len(args),n,row*n+col+1, projection='3d')
                ax.plot_surface(X, Y, args[row][col].squeeze(), cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
                ax.set_zlim(0,I_max)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.show()

#Calculate gradient of the reconstruction loss with respect to predicted image
#reconstruction loss=f(image_true,image_pred,labels)
#Where labels is with shape (len(image_true),2) for original amplitude and gaussian noise
def calculate_gradient(image_true,image_pred,reconstruction_loss,labels=None):
    image_true=tf.constant(image_true)
    image_pred=tf.constant(image_pred)
    if labels is None:
        labels=np.ones((len(image_true),2))
    labels=tf.constant(labels)
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g:
      g.watch(image_pred)
      loss=reconstruction_loss(image_true,image_pred,labels)
    return K.eval(g.gradient(loss,image_pred))

#Present comparison of "imgs" (32,64,64,1) to the results of their reconstruction with the VAE "model"
#One may also see residuals, residuals for reconstruction_loss_function
#Or gradients of reconstruction_loss_function with respect to decoded images
#Dimensions are same as in plot_galaxies
def present_reconstruction(model,images,number_of_images_to_consider=8,reconstruction_loss_function=None,labels=None,dimensions='2d',resid=False, grads=False,abs_grads=True):
    #Images selection
    images_for_reconst=images
    decoded_to_reconstruct=model.predict(images_for_reconst, batch_size=len(images))
    plot_galaxies(images_for_reconst[:number_of_images_to_consider],decoded_to_reconstruct[:number_of_images_to_consider],dimensions=dimensions)
    if grads:
        if (reconstruction_loss_function is not None) and (labels is not None):
          #Gradients in form of (len(images),64,64,1)
          gradients=calculate_gradient(images_for_reconst,decoded_to_reconstruct.astype('double'),labels.astype('double'),reconstruction_loss_function)
          if abs_grads:
            #Normalize gradients to one and the same dynamical range for better visualization
            norm_gradients=np.abs(gradients)
            norm_gradients=np.array([norm_gradients[i]/norm_gradients[i].flatten().max() for i in range(len(norm_gradients))])
          else:
            #Normalize gradients to one and the same dynamical range for better visualization
            norm_gradients=np.array([gradients[i]-gradients[i].flatten().min() for i in range(len(gradients))])
            norm_gradients=np.array([norm_gradients[i]/norm_gradients[i].flatten().max() for i in range(len(norm_gradients))])
        plot_galaxies(norm_gradients[:number_of_images_to_consider],dimensions=dimensions,colorbar=True)
    if resid:
      if (reconstruction_loss_function is None) or (labels is None):
          #Simple residuals
          residuals=decoded_to_reconstruct-images_for_reconst
      else:
          #Loss as it is seen by reconstruction_loss_function in a form of (len(images),64,64,1)
          residuals=K.eval(reconstruction_loss_function(images_for_reconst,decoded_to_reconstruct.astype('double'),labels.astype('double')))
      plot_galaxies(residuals[:number_of_images_to_consider],dimensions=dimensions,colorbar=True)

    #Return either of the values of interest
    if grads:
      return gradients
    elif resid:
      return residuals

#Read learning curve in a from of arrays of dictionaries {'loss': Train_loss,'val_loss': Validation_loss}
def learning_curve(filename,start_epoch=0,stop_epoch=1000,log_scale=True):
  logs_file=open(filename)
  lines=logs_file.readlines()
  logs_file.close()


  loss=np.array([])
  val_loss=np.array([])
  #Parse the array of dicts
  #[{'loss': Train_loss,'val_loss': Validation_loss},...]
  for line in lines:
    note=ast.literal_eval(line)
    loss=np.append(loss,[note['loss']])
    val_loss=np.append(val_loss,[note['val_loss']])

  start_index=start_epoch
  stop_index=np.minimum(len(loss),stop_epoch+1)
  #Plot Train_loss and Validation_loss
  plt.plot(np.arange(start_index,stop_index),loss[start_index:stop_index],label='Train')
  plt.plot(np.arange(start_index,stop_index),val_loss[start_index:stop_index],label='Validation')
  if log_scale:
    plt.yscale('log')
  plt.ylabel('Loss')
  plt.xlabel('epoch number')
  plt.title('Learning curve')
  plt.legend()
  plt.show()

#Show KL_divergence for each of the latent variables in sorted order
def DKL_per_variable(z_means,z_log_vars):
    DKL=-0.5 * K.mean(1 + z_log_vars - K.square(z_means) - K.exp(z_log_vars), axis=0)
    sns.barplot(x=np.linspace(0,64,64),y=np.sort(DKL)[::-1])
    plt.xticks([])
    plt.xlabel('Latent variable')
    plt.ylabel('nats')
    plt.title(r'KL_div')
    plt.show()
    return K.eval(DKL)

#KL_divergence Learning curve for five most significant latent features
def DKL_learning_curve(filename,start_epoch=0,stop_epoch=1000):
  text_file = open(filename, "r")
  lines = text_file.read().replace('\n','').replace('[','').split(']')
  nums=np.array([np.array([float(x) for x in line.split()]) for line in lines[:-1]])
  text_file.close()

  start_index=start_epoch//10
  stop_index=np.minimum(len(nums),stop_epoch//10-1)
  print(start_index,stop_index,len(nums))

  significant_vars=nums[-1].argsort()[-5:][::-1]
  for i in range(5):
    plt.plot(np.arange(start_index,stop_index),nums[start_index:stop_index,significant_vars[i]],label=i)
  plt.ylabel('nats')
  plt.xlabel('Checkpoint number')
  plt.title('DKL significant features Learning curve')
  plt.legend()
  plt.show()

  plt.plot(np.arange(start_index,stop_index),nums[start_index:stop_index].sum(axis=1))
  plt.ylabel('nats')
  plt.xlabel('Checkpoint number')
  plt.title('DKL Capacity Learning curve')
  plt.show()

#Show characteristics of the latent space and balance between reconstruction and regularization loss
def Show_latent_distr(models,x_test_labeled,reconst_loss_func=tf.keras.losses.binary_crossentropy,regul_loss_func=None):
  x_test=x_test_labeled[:,:-1,:,:]
  #Target Sersic parameters
  labels=x_test_labeled[:,-1,:,0]
  decoded=models['vae'].predict(x_test)

  fig,ax=plt.subplots(2,2,figsize=(20,5))

  z_mean=K.eval(models['z_meaner'](x_test))
  z_log_var=K.eval(models['z_log_varer'](x_test))
  ratio=np.std(z_mean,axis=0)/np.mean(np.exp(z_log_var/2),axis=0)
  #Latent space SNR
  sns.barplot(ax=ax[0,0],x=np.linspace(0,64,64),y=np.sort(ratio)[::-1])
  ax[0,0].hlines(1,0,63,label=r'$\mu_{std}=\sigma_{mean}$')
  ax[0,0].legend()
  ax[0,0].set_xticks([])
  ax[0,0].set_xlabel('Latent variable')
  ax[0,0].set_ylabel('Ratio')
  ax[0,0].set_title('Latent SNR')

  #Latent space DKL
  DKL=0.5 * K.mean(K.exp(z_log_var)+K.square(z_mean) - 1 - z_log_var, axis=0)
  sns.barplot(ax=ax[0,1],x=np.linspace(0,64,64),y=np.sort(DKL)[::-1])
  ax[0,1].set_xticks([])
  ax[0,1].set_xlabel('Latent variable')
  ax[0,1].set_ylabel('nats')
  ax[0,1].set_title(r'KL_div')

  #reconstruction quality
  Log_loss=K.mean(reconst_loss_func(x_test,decoded,labels),axis=[1,2])

  df=pd.DataFrame()
  df['Reconstruction loss']=Log_loss
  sns.barplot(ax=ax[1,0],data=df[['Reconstruction loss']])
  if regul_loss_func is not None:
    df['Regularization loss']=regul_loss_func(z_mean,z_log_var)/image_size/image_size
    sns.barplot(ax=ax[1,1],data=df[['Regularization loss']])

  plt.tight_layout()
  plt.show()

#Latent SNR
def latent_relation_of_variances(ratio,sorted=False):
    if sorted:
      sns.barplot(x=np.linspace(0,64,64),y=np.sort(ratio)[::-1])
    else:
      sns.barplot(x=np.linspace(0,64,64),y=ratio)
    #g.xaxis.set_visible(False)
    plt.hlines(1,0,63,label=r'$\mu_{std}=\sigma_{mean}$')
    plt.legend()
    plt.xticks([])
    plt.xlabel('Latent variable')
    plt.ylabel('Ratio')
    plt.title(r'$\mu_{std}/\sigma_{mean}$')
    plt.show()

#Present distribution of all 64 latent features for either mean or variance
def latent_distribution(latent_variable):
  fig, axs = plt.subplots(nrows=8, ncols=8, figsize=(20, 20))
  axs_flat = axs.flatten()
  for i in range(latent_variable.shape[1]):
    axs_flat[i].hist(latent_variable[:,i], bins=50)
    axs_flat[i].set_title('Z {}'.format(i))
    axs_flat[i].get_yaxis().set_visible(False)
  fig.tight_layout()
  plt.show()


#Vary each of latent variables and consider its impact on the image
def galaxy_properties_from_latent_variables(models,z_means,number_of_z_to_consider=64,dimensions='2d',sort_array=None):
  latent_average=z_means.mean(axis=0)
  if sort_array is None:
    variables_to_consider=np.linspace(0,number_of_z_to_consider-1,number_of_z_to_consider,dtype=int)
  else:
    variables_to_consider=np.argsort(sort_array)[::-1][:number_of_z_to_consider]
  variances_to_consider=z_means.std(axis=0)[variables_to_consider]
  z_to_consider=np.zeros((number_of_z_to_consider,7,64))
  #i - number of variable to change
  for i in range(number_of_z_to_consider):
    #j - number of sigmas to add to it
    for j in range(7):
      #Assign everything to be like average galaxy
      z_to_consider[i,j,:]=latent_average
      #Vary one of the variables to get mu+-(0,1,2,3)*sigma
      z_to_consider[i,j,variables_to_consider[i]]+=(j-3)*variances_to_consider[i]
  images_to_consider=models['decoder'].predict(z_to_consider.reshape((number_of_z_to_consider*7,64)))
  #plot_digits(images_to_consider[:7],images_to_consider[7:14])
  for i in range(number_of_z_to_consider):
    plot_galaxies(images_to_consider[i*7:(i+1)*7],dimensions=dimensions)

#Vary each of PCA components of the transformed latent space and consider its impact on the image
def galaxy_properties_from_PCA(models,z_means,n_components=4,dimensions='2d'):
  pca = PCA(n_components=n_components)
  pca.fit(z_means)
  Principal_components=pca.transform(z_means)
  Principal_means=Principal_components.mean(axis=0)
  Principal_stds=Principal_components.std(axis=0)

  PC=np.zeros((n_components,7,n_components))
  #i - number of variable to change
  for i in range(n_components):
    #j - number of sigmas to add to it
    for j in range(7):
      #Assign everything to be like average galaxy
        PC[i,j,:]=Principal_means
        PC[i,j,i]+=(j-3)*Principal_stds[i]

  z_to_consider=pca.inverse_transform(PC)
  print('Explained variance ratio',pca.explained_variance_ratio_)
  images_to_consider=models['decoder'].predict(z_to_consider.reshape((n_components*7,64)))
  #plot_digits(images_to_consider[:7],images_to_consider[7:14])
  for i in range(n_components):
    plot_galaxies(images_to_consider[i*7:(i+1)*7],dimensions=dimensions)

# Pick random values of latent variables and reconstruct images from them
def sample_random_images(models,z_means):
  sample_z=np.random.normal(z_means.mean(axis=0),z_means.std(axis=0),size=(32,64))
  sample_image=models['decoder'].predict(sample_z)
  plot_galaxies(sample_image[:10],sample_image[10:20],sample_image[20:30])

# Pick random values of PCA transformed latent space components and reconstruct images from them
def sample_random_images_PCA(models,z_means,n_components=64,dimensions='2d'):
  pca = PCA(n_components=n_components)
  pca.fit(z_means)
  Principal_components=pca.transform(z_means)
  Principal_means=Principal_components.mean(axis=0)
  Principal_stds=Principal_components.std(axis=0)

  sample_PC=np.random.normal(Principal_means,Principal_stds,size=(32,64))

  sample_z=pca.inverse_transform(sample_PC)
  sample_image=models['decoder'].predict(sample_z)
  plot_galaxies(sample_image[:10],sample_image[10:20],sample_image[20:30])

#Estimate statistics of error of maximal intensity reconstruction,
# reconstruction_loss_function and SSIM metrics
def numerical_reconstruction_quality(images_labeled,decoded,reconstruction_loss_function):
        images=images_labeled[:,:-1,:,:]
        #Amplitudes and gaussian noises
        labels=images_labeled[:,-1,:,0]


        fig,ax=plt.subplots(1,3,figsize=(10,5))
        maxima=K.eval(decoded).max(axis=(1,2,3))
        #Relative absolute error for images_true normalized to 1.
        maxima_RAE=np.abs(maxima-1)
        ax[0].hist(maxima,bins=20)
        ax[0].set_title('Distribution of reconstructed maximum')

        reconstruction_loss=K.eval(K.mean(reconstruction_loss_function(images,decoded,labels),axis=[1,2]))
        reconstruction_loss=np.sort(reconstruction_loss)[:int(len(reconstruction_loss)*0.99)]
        ax[1].hist(reconstruction_loss,bins=40)
        ax[1].set_title('reconstruction loss')

        SSIM=tf.image.ssim(images.astype('double'), decoded.astype('double'),max_val=1,filter_size=8).numpy()
        ax[2].hist(SSIM,bins=20)
        ax[2].set_title('SSIM')
        plt.show()

        print('Max values RAE {:.3f} ± {:.3f}, median {:.3f}'.format(maxima_RAE.mean(),maxima_RAE.std(),np.median(maxima_RAE)))
        print('reconstruction loss values {:.6f} ± {:.6f}, median {:.6f}'.format(reconstruction_loss.mean(),reconstruction_loss.std(),np.median(reconstruction_loss)))
        print('SSIM values {:.3f} ± {:.3f}, median {:.3f}'.format(SSIM.mean(),SSIM.std(),np.median(SSIM)))
