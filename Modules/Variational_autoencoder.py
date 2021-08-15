import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization,Flatten,Reshape,Lambda,Conv2D,Conv2DTranspose,LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras import initializers
from tensorflow.python.ops import math_ops
import keras.backend as K

latent_dim=64
image_size=64
batch_size=32

start_lr=1e-6
Adam_beta_1=0.5
Adam_beta_2=0.999
#Clip gradients to avoid explosion
Adam_clipvalue=1
#clip values
epsilon_=1e-3
Sigmoid_temperature=1
MAE_SSIM_alpha=0.5

#He initialization for Relu activated layers
Heinitializer = initializers.HeNormal()

#Xavier intialization for differentiable functions activated layers
Xavierinitializer=initializers.GlorotNormal()

#Zero initializer for latent space to start exactly from prior N(0,I)
Zero_initializer=initializers.Constant(value=0)

#Extract values below zero, as they for sure are noise
#Normalize galaxy intensity to 1
def normalize_image(images):
    #Normalize image to [0,1]
    for i in range(len(images)):
        images[i][images[i]<0]=0
        images[i]=images[i]/images[i].max()
    return images

#Transfrom an array (N) to matrix (N,64,64,1)
#It is needed for construction weighted losses like chi_sq from MSE
def get_matrix_shape(array,matrix):
  if array.shape[0]==matrix.shape[0]:
    result=K.reshape(tf.repeat(array,tf.size(matrix[0])),(matrix.shape))
    return tf.cast(result,matrix.dtype)
  else:
    return False

def temp_sigmoid(x):
    return K.pow((1+K.exp(-x/Sigmoid_temperature)),-1)

def crop_logit(x):
  p=tf.clip_by_value(x, epsilon_, 1. - epsilon_)
  return - K.log(1. / p - 1.)


def Loss_dumped_sign_Sigmoid(x):
    #gradient is (1-exp(-x))/x/(1-x)
    Exponent_coef=get_matrix_shape(tf.math.exp(1.)*tf.ones(len(x)),x)
    return tf.math.special.expint(1-x)/Exponent_coef-tf.math.special.expint(-x)+K.log(x/(1-x))

def gauss_kernel(l=64, sig=8.):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = tf.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = tf.meshgrid(ax, ax)

    kernel = tf.math.exp(-0.5 * (tf.square(xx) + tf.square(yy)) / tf.square(sig))

    return kernel / K.sum(kernel)


Gaussian_kernel=gauss_kernel(sig=8.)

#Encoder
def encoder_function(input_img):
    x = Conv2D(filters=64, kernel_size=4, strides=2,padding='same',use_bias=False,kernel_initializer=Heinitializer)(input_img)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=128, kernel_size=4, strides=2,padding='same',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=256, kernel_size=4, strides=2,padding='same',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=512, kernel_size=4, strides=2,padding='same',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=4096, kernel_size=4, strides=1,padding='valid',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)

    #Predict mean of standard distribution and logarithm of variance
    #Initialize weights as zeros to start exactly from the prior distribution N(0,I)
    z_mean = Dense(latent_dim,kernel_initializer=Zero_initializer,bias_initializer=Zero_initializer)(x)
    z_log_var = Dense(latent_dim, kernel_initializer=Zero_initializer,bias_initializer=Zero_initializer)(x)
    return z_mean,z_log_var

#Getter for decoder function that mirrors the encoder
#It is needed since we use different output activations
def get_decoder(activation='No activation'):

  def decoder_function(z):

    x = Reshape(target_shape=(1, 1, 64))(z)

    x = Conv2DTranspose(filters=512, kernel_size=4, strides=1,padding='valid',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(filters=256, kernel_size=4, strides=2,padding='same',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(filters=128, kernel_size=4, strides=2,padding='same',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(filters=64, kernel_size=4, strides=2,padding='same',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    if activation=='No activation':
        decoded = Conv2DTranspose(filters=1, kernel_size=4, strides=2,padding='same',use_bias=False
                                ,kernel_initializer=Xavierinitializer)(x)
    elif activation=='temp_sigmoid':
        decoded = Conv2DTranspose(filters=1, kernel_size=4, strides=2,padding='same',use_bias=False,
                              activation=temp_sigmoid,kernel_initializer=Xavierinitializer)(x)
    else:
        decoded = Conv2DTranspose(filters=1, kernel_size=4, strides=2,padding='same',use_bias=False,
                              activation=activation,kernel_initializer=Xavierinitializer)(x)

    return decoded

  return decoder_function

#Getter for reconstruction loss function
#By default this function expects y_true,y_pred,labels
#With labels as (N,2) for amplitudes and gaussian noises of the images
def get_reconstruction_loss(loss_type='mae'):

  if loss_type=='chi_sq':
    #chi_sq=(data-model)^2/(Poisson_variance+Gaussian_variance)
    #Poisson_variance=image/exposure_time
    #Gaussian_variance=Gaussian_noise^2 in a from of image (N,64,64,1)

    def reconstruction_loss_function(y_true,y_pred,labels):
      #Set the true values for Chi squared
      Amplitude_true=labels[:,0]
      Gaussian_noise=labels[:,1]

      y_true = tf.convert_to_tensor(y_true)
      y_pred = tf.convert_to_tensor(y_pred)

      #(data-model)^2
      reconstruction_loss=K.square(y_true-y_pred)

      #Calculate Poisson and Gaussian noise
      #COSMOS dataset exposure time is 2028 sec
      Poisson_noise_variance=tf.abs(y_true/2028)
      Gaussian_noise_scaled=get_matrix_shape(Gaussian_noise,y_true)
      Gaussian_noise_variance=tf.math.pow(Gaussian_noise_scaled,2)

      #Calculate weights=1/noise_variance
      noise_variance=Poisson_noise_variance+Gaussian_noise_variance
      weights=tf.math.pow(noise_variance,-1)
      weights= math_ops.cast(weights, y_pred.dtype)

      #chi_sq=(data-model)^2*weights
      chi_sq=reconstruction_loss*weights
      #Reduce last dimension to return images (N,64,64)
      return K.sum(chi_sq,axis=-1)


  elif loss_type=='mse':
    #MSE=(data-model)^2

    def reconstruction_loss_function(y_true,y_pred,labels):
      y_true = tf.convert_to_tensor(y_true)
      y_pred = tf.convert_to_tensor(y_pred)

      #(data-model)^2
      reconstruction_loss=K.square(y_true-y_pred)

      #Reduce last dimension to return images (N,64,64)
      return K.sum(reconstruction_loss,axis=-1)

  elif loss_type=='mae':
    #MAE=abs(data-model)

    def reconstruction_loss_function(y_true,y_pred,labels):
      y_true = tf.convert_to_tensor(y_true)
      y_pred = tf.convert_to_tensor(y_pred)

      #(data-model)^2
      reconstruction_loss=K.abs(y_true-y_pred)

      #Reduce last dimension to return images (N,64,64)
      return K.sum(reconstruction_loss,axis=-1)

  elif loss_type=='mae_sigmoid_sign_grads':
    #MAE=abs(data-model)
    #If model comes from sigmoid we ensure that gradients
    #With respect to logit before output activation are sign(y-x)

    def reconstruction_loss_function(y_true,y_pred,labels):

        y_true = tf.convert_to_tensor(tf.clip_by_value(y_true, epsilon_, 1. - epsilon_))
        y_pred = tf.convert_to_tensor(tf.clip_by_value(y_pred, epsilon_, 1. - epsilon_))

        target_logit= - K.log(1. / y_true - 1.)
        pred_logit= - K.log(1. / y_pred - 1.)

        #(data-model)^2
        coefficient=get_matrix_shape(Sigmoid_temperature*tf.ones(len(y_true)),y_true)
        reconstruction_loss=K.abs(target_logit-pred_logit)*coefficient

        #Reduce last dimension to return images (N,64,64)
        return K.sum(reconstruction_loss,axis=-1)

  elif loss_type=='mae_sigmoid_sign_grads_dumped':
    #MAE=abs(data-model)
    #If model comes from sigmoid we ensure that gradients
    #With respect to logit before output activation are sign(y-x)*(1-exp(-x))

    def reconstruction_loss_function(y_true,y_pred,labels):
        y_true = tf.convert_to_tensor(tf.clip_by_value(y_true, epsilon_, 1. - epsilon_))
        y_pred = tf.convert_to_tensor(tf.clip_by_value(y_pred, epsilon_, 1. - epsilon_))


        #(data-model)^2
        Temperature_coef=get_matrix_shape(Sigmoid_temperature*tf.ones(len(y_true)),y_true)
        reconstruction_loss=K.abs(Loss_dumped_sign_Sigmoid(y_true)-Loss_dumped_sign_Sigmoid(y_pred))*Temperature_coef
        #Reduce last dimension to return images (N,64,64)
        return K.sum(reconstruction_loss,axis=-1)

  elif loss_type=='binary_crossentropy':
    #BCE is proportional to sum(y_true*log(y_pred)+(1-y_true)*log(1-y_pred))

    def reconstruction_loss_function(y_true,y_pred,labels):
      Temperature_coef=get_matrix_shape(Sigmoid_temperature*tf.ones(len(y_true)),y_true)

      #Reduce last dimension to return images (N,64,64)
      return K.sum(K.binary_crossentropy(y_true,y_pred)*Temperature_coef,axis=-1)

  elif loss_type=='SSIM':

    def reconstruction_loss_function(y_true,y_pred,labels):

      #Reduce last dimension to return images (N,64,64)
      return 1-tf.image.ssim(y_true,y_pred,max_val=1,filter_size=8)

  elif loss_type=='MAE_SSIM':

    def reconstruction_loss_function(y_true,y_pred,labels):

      y_true = tf.convert_to_tensor(y_true)
      y_pred = tf.convert_to_tensor(y_pred)


      kernel=K.reshape(tf.ones((len(y_true),64,64))*Gaussian_kernel,(len(y_true),64,64,1))
      MAE_loss=K.sum(K.abs(y_true-y_pred)*kernel,axis=[1,2,3])
      SSIM_loss=1-tf.image.ssim(y_true,y_pred,max_val=1,filter_size=8)

      reconstruction_loss=MAE_SSIM_alpha*SSIM_loss+(1-MAE_SSIM_alpha)*MAE_loss

      #Reduce last dimension to return images (N,64,64)
      return K.sum(reconstruction_loss,axis=-1)



  else:
    print('Wrong loss type')

  return reconstruction_loss_function

#Getter for regularization function
def get_regularization_loss(beta_vae=None,lambda_Flow=None,Capacity=None,gamma=None):

  #We either make constant pressure on the latent space with "beta_vae"
  if beta_vae is not None:
      def regularization_loss_function(mean,logvar,beta_vae=1e-2):
          KL_loss=0.5 * K.sum(K.exp(logvar)+K.square(mean)-1-logvar, axis=-1)
          return beta_vae*KL_loss


  #Or constraint amount of information leaking into regularization loss  (Lanusse et al. 2020)
  elif lambda_Flow is not None:
      def regularization_loss_function(mean,logvar,lambda_Flow=300):
          KL_loss=0.5 * K.sum(K.exp(logvar)+K.square(mean)-1-logvar, axis=-1)
          #Strange it is more sensible to constrain gradients, not the values
          return K.minimum(lambda_Flow,KL_loss)

  #Or control its latent space' capacit with "Capacity" and "gamma"
  elif (gamma is not None) and (Capacity is not None):
      def regularization_loss_function(mean,logvar,Capacity=300):
          KL_loss=0.5 * K.sum(K.exp(logvar)+K.square(mean)-1-logvar, axis=-1)
          #Strange it is more sensible to constrain gradients, not the values
          return gamma*tf.abs(KL_loss-Capacity)

  else:
    print("No parameters for regularization")

  return regularization_loss_function

#Getter for overall model loss
def get_model_loss(models,loss_type='mae',beta_vae=None,
                   lambda_Flow=None,Capacity=None,gamma=None,variable_regularization_parameter=False):

  reconstruction_loss_function=get_reconstruction_loss(loss_type)
  regularization_loss_function=get_regularization_loss(beta_vae,lambda_Flow,Capacity,gamma)

  #Leaves a possibility to vary the regularization parameter during training
  if (variable_regularization_parameter==True):
    global Regularization_parameter


  #Beta,Lambda or Capacity
  Regularization_parameter=0
  if beta_vae is not None:
      Regularization_parameter=beta_vae
  elif lambda_Flow is not None:
      Regularization_parameter=lambda_Flow
  elif (gamma is not None) and (Capacity is not None):
      Regularization_parameter=Capacity


  def model_loss_function(target_x_labeled,x_decoded):

    #Target image (batch_size,64,64,1)
    x_true=target_x_labeled[:,:-1,:,:]
    #Amplitude and Gaussian noise (batch_size,2)
    labels=target_x_labeled[:,-1,:,0]

    #KL divergence regularization quality
    mean = models['z_meaner'](x_true)
    logvar=models['z_log_varer'](x_true)
    #Array (N)
    regularization_loss=regularization_loss_function(mean,logvar,Regularization_parameter)

    #reconstruction quality
    reconstruction_loss=reconstruction_loss_function(x_true,x_decoded,labels)
    if (loss_type!='SSIM') and (loss_type!='MAE_SSIM'):
        reconstruction_loss=K.sum(reconstruction_loss,axis=[1,2])

    #Sum reconstruction fidelity and informational cost of deviation from prior latent space distribution
    VAE_Loss=(reconstruction_loss+regularization_loss)/image_size/image_size
    return VAE_Loss

  return model_loss_function,reconstruction_loss_function,regularization_loss_function


def create_vae(activation='softplus',loss_type='mae',beta_vae=None,
               lambda_Flow=None,Capacity=None,gamma=None,variable_regularization_parameter=False):

    #Reparametrization trick
    def reparameterize(args):
      mean,logvar=args
      eps = tf.random.normal(shape=mean.shape)
      return eps * tf.exp(logvar/2) + mean

    #Get decoder
    decoder_function=get_decoder(activation)

    #VAE
    models = {}

    #Encoder
    input_img = Input(batch_shape=(batch_size, image_size, image_size, 1))
    #Predict latent mean and variance
    z_mean, z_log_var=encoder_function(input_img)
    #Carry out reparametrization trick
    l=Lambda(reparameterize, output_shape=(latent_dim,))([z_mean, z_log_var])

    models["encoder"]  = Model(input_img, l, name='Encoder')
    models["z_meaner"] = Model(input_img, z_mean, name='Enc_z_mean')
    models["z_log_varer"] = Model(input_img, z_log_var, name='Enc_z_log_var')

    #Decoder
    z = Input(shape=(latent_dim, ))
    decoded=decoder_function(z)

    #Decode latent features
    models["decoder"] = Model(z, decoded, name='Decoder')

    #Decode(Encode(input_images))
    models["vae"]     = Model(input_img, models["decoder"](models["encoder"](input_img)), name="VAE")


    #Get all the Losses
    temp_reconstruction_loss_function=get_reconstruction_loss(loss_type)
    if (loss_type!='SSIM') and (loss_type!='MAE_SSIM'):
        def reconstruction_loss_function(y_true,y_pred,labels):
            reconstruction_loss=temp_reconstruction_loss_function(y_true,y_pred,labels)
            return K.sum(reconstruction_loss,axis=[1,2])
    else:
        reconstruction_loss_function=temp_reconstruction_loss_function


    regularization_loss_function=get_regularization_loss(beta_vae,lambda_Flow,Capacity,gamma)

    #Leaves a possibility to vary the regularization parameter during training
    if (variable_regularization_parameter==True):
      global Regularization_parameter


    #Beta,Lambda or Capacity
    Regularization_parameter=0
    if beta_vae is not None:
      Regularization_parameter=beta_vae
    elif lambda_Flow is not None:
      Regularization_parameter=lambda_Flow
    elif (gamma is not None) and (Capacity is not None):
      Regularization_parameter=Capacity


    def model_loss_function(target_x_labeled,x_decoded):

      #Target image (batch_size,64,64,1)
      x_true=target_x_labeled[:,:-1,:,:]
      #Amplitude and Gaussian noise (batch_size,2)
      labels=target_x_labeled[:,-1,:,0]

      #KL divergence regularization quality
      mean = models['z_meaner'](x_true)
      logvar=models['z_log_varer'](x_true)
      #Array (N)
      regularization_loss=regularization_loss_function(mean,logvar,Regularization_parameter)

      #reconstruction quality
      reconstruction_loss=reconstruction_loss_function(x_true,x_decoded,labels)
      #Bring it to array (N)
      #reconstruction_loss=K.sum(reconstruction_loss,axis=[1,2])

      #Sum reconstruction fidelity and informational cost of deviation from prior latent space distribution
      VAE_Loss=(reconstruction_loss+regularization_loss)/image_size/image_size
      return VAE_Loss



    return models,model_loss_function,reconstruction_loss_function,regularization_loss_function

#Getter for the entire VAE
def get_VAE(activation='softplus',loss_type='mae',beta_vae=None,
               lambda_Flow=None,Capacity=None,gamma=None,variable_regularization_parameter=False,clip_grads=True):

  with tf.device('/device:GPU:0'):
    models,model_loss_function,reconstruction_loss_function,regularization_loss_function =\
        create_vae(activation,loss_type,beta_vae,lambda_Flow,Capacity,gamma,variable_regularization_parameter)

    if clip_grads:
        models["vae"].compile(optimizer=Adam(learning_rate=start_lr,
                                         beta_1=Adam_beta_1, beta_2=Adam_beta_2,clipvalue=Adam_clipvalue),
                                         loss=model_loss_function)
    else:
        models["vae"].compile(optimizer=Adam(learning_rate=start_lr,
                                             beta_1=Adam_beta_1, beta_2=Adam_beta_2),loss=model_loss_function)

    return models,model_loss_function,reconstruction_loss_function,regularization_loss_function
