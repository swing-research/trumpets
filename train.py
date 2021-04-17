import os
import cv2
import shutil
import numpy as np
from time import time

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp

from utils import *
from models import generator, latent_generator, posterior

import operators as inv_probs
from logdetJ import wrapper_logdet

tfb = tfp.bijectors
tfd = tfp.distributions

FLAGS, unparsed = flags()

num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
dataset = FLAGS.dataset
lr = FLAGS.lr
gpu_num = FLAGS.gpu_num
learntop = bool(FLAGS.learntop)
remove_all = bool(FLAGS.remove_all)
desc = FLAGS.desc
ml_threshold = FLAGS.ml_threshold
model_depth = FLAGS.model_depth
latent_depth = FLAGS.latent_depth
inv_conv_activation = FLAGS.inv_conv_activation
T = FLAGS.T

run_train = FLAGS.train
run_inv = FLAGS.inv
inv_prob = FLAGS.inv_prob
snr = FLAGS.snr
train_posterior = FLAGS.posterior

calc_logdet = FLAGS.calc_logdet


c = 1 if dataset == 'mnist' or dataset == 'chest' else 3
f = 2 if dataset == 'chest' else 1

        
        

all_experiments = 'experiment_results/'
if os.path.exists(all_experiments) == False:

    os.mkdir(all_experiments)

# experiment path
exp_path = all_experiments + 'Final_' + \
    dataset + '_' + 'model_depth_%d' % (model_depth,) + '_' + 'latent_depth_%d'% (latent_depth,) + '_learntop_%d' \
        % (int(learntop)) + '_' + desc


if os.path.exists(exp_path) == True and remove_all == True:
    shutil.rmtree(exp_path)

if os.path.exists(exp_path) == False:
    os.mkdir(exp_path)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)



class Prior(layers.Layer):
    """Defines the low dimensional distribution as Guassian"""
    def __init__(self, **kwargs):
        super(Prior, self).__init__()
        self.mu = tf.Variable(tf.zeros(4*f *4*f *4*c),
                              dtype=tf.float32, trainable=learntop)
        self.logsigma = tf.Variable(tf.ones(4*f *4*f *4*c)*np.log(1.0),
                                    dtype=tf.float32, trainable=learntop)

        self.prior = tfd.MultivariateNormalDiag(
            self.mu, tf.math.exp(self.logsigma))



def latent_space_interplotion(model, x1, x2, latent=True , sample_number = 16):
    """Creates a grid of images from x1 to x2"""
    if not latent:
        """if latent then x1 and x2 are treated to be latent codes"""
        z1, _ = model(x1, reverse=True)
        z2, _ = model(x2, reverse=True)
    else:
        z1 = x1
        z2 = x2

    # create a grid of latent codes
    
    a = tf.cast(tf.reshape(tf.linspace(0, 1, sample_number), (sample_number, 1)), tf.float32)
    z = z1 + a * (z2 - z1)
    xhat = model(z, reverse= True)[0]

    return xhat.numpy()


def train(num_epochs,
          batch_size,
          dataset,
          lr,
          exp_path,):


    # Print the experiment setup:
    print('Experiment setup:')
    print('---> num_epochs: {}'.format(num_epochs))
    print('---> batch_size: {}'.format(batch_size))
    print('---> dataset: {}'.format(dataset))
    print('---> Learning rate: {}'.format(lr))
    print('---> experiment path: {}'.format(exp_path))
    
    if os.path.exists(os.path.join(exp_path, 'logs')):
        shutil.rmtree(os.path.join(exp_path, 'logs'))

    MSE_train_log_dir = os.path.join(exp_path, 'logs', 'MSE_train')
    MSE_train_summary_writer = tf.summary.create_file_writer(MSE_train_log_dir)
    MSE_train_loss_metric = tf.keras.metrics.Mean(
        'MSE_train_loss', dtype=tf.float32)

    MSE_test_log_dir = os.path.join(exp_path, 'logs', 'MSE_test')
    MSE_test_summary_writer = tf.summary.create_file_writer(MSE_test_log_dir)
    MSE_test_loss_metric = tf.keras.metrics.Mean('MSE_test_loss', dtype=tf.float32)

    ML_log_dir = os.path.join(exp_path, 'logs', 'ML')
    ML_summary_writer = tf.summary.create_file_writer(ML_log_dir)
    ML_loss_metric = tf.keras.metrics.Mean('ML_loss', dtype=tf.float32)
    
    pz_log_dir = os.path.join(exp_path, 'logs', 'pz')
    pz_summary_writer = tf.summary.create_file_writer(pz_log_dir)
    pz_metric = tf.keras.metrics.Mean(
        'pz', dtype=tf.float32)
    
    jacobian_log_dir = os.path.join(exp_path, 'logs', 'jacobian')
    jacobian_summary_writer = tf.summary.create_file_writer(jacobian_log_dir)
    jacobian_metric = tf.keras.metrics.Mean(
        'jacobian', dtype=tf.float32)
    
    

    training_images, testing_images = Dataset_preprocessing(image_type=True, dataset=dataset)
    training_images = tf.convert_to_tensor(training_images, tf.float32)
    testing_images = tf.convert_to_tensor(testing_images, tf.float32)
    print('Dataset is loaded: training and test dataset shape: {} {}'.
          format(np.shape(training_images), np.shape(testing_images)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr , clipnorm=1) # Optimizer of injective network
    f_optimizer = tf.keras.optimizers.Adam(learning_rate=lr) # Optimizer of bijective network

    pz = Prior()
    
    time_vector = np.zeros([num_epochs,1]) # time per epoch
    model = generator(dataset=dataset ,
                      revnet_depth = model_depth ,
                      activation = inv_conv_activation) # Injective network
    latent_model = latent_generator(dataset=dataset ,
                                    revnet_depth = latent_depth) # Bijective network

     # call generator once to set weights (Data dependent initialization)
    dummy_x = training_images[0:1000]
    dummy_z, _ = model(dummy_x, reverse=False)
    dummy_l_z , _ = latent_model(dummy_z, reverse=False)
    

    ckpt = tf.train.Checkpoint(pz = pz , model=model,optimizer=optimizer,
        latent_model=latent_model,f_optimizer=f_optimizer)
    manager = tf.train.CheckpointManager(
        ckpt, os.path.join(exp_path, 'checkpoints'), max_to_keep=5)

    ckpt.restore(manager.latest_checkpoint)
    
    @tf.function
    def train_step_mse(sample):
        """MSE training of the injective network"""

        bs = tf.shape(sample)[0]
        with tf.GradientTape() as tape:
            
            MSE = tf.keras.losses.MeanSquaredError()
            MSE_z = tf.keras.losses.MeanSquaredError()
            
            z , _ = model(sample, reverse= False)
            
            recon = model(z , reverse = True)[0]
            recon_z = model(recon , reverse = False)[0]
            
            mse_loss = MSE(sample , recon)
            mse_z = MSE_z(z , recon_z) # Added for stability
            loss = mse_loss + mse_z
            
            variables= tape.watched_variables()
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

        return loss
    
    
    
    @tf.function
    def train_step_ml(sample):
        """ML training of the bijective network"""

        bs = tf.shape(sample)[0]
        with tf.GradientTape() as tape:
            latent_sample, obj = latent_model(sample, reverse=False)
            p = -tf.reduce_mean(pz.prior.log_prob(latent_sample))
            j = -tf.reduce_mean(obj) # Log-det of Jacobian
            loss =  p + j
            variables = tape.watched_variables()
            grads = tape.gradient(loss, variables)
            f_optimizer.apply_gradients(zip(grads, variables))

        return loss , p , j

   

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    
    Ntrain = len(training_images.numpy())
    
    if run_train:
        for epoch in range(num_epochs):
            
            epoch_start = time()
            training_images = tf.random.shuffle(training_images)
            testing_images = tf.random.shuffle(testing_images)
            num_iters = Ntrain//batch_size

            for i in range(num_iters):
                if epoch < ml_threshold: 
                    # MSE traiing of the injective network for ml-threshold epochs
                    t = np.random.randint(0, num_iters)
                    x = training_images[t*batch_size:(t+1)*batch_size]
                    mse_loss = train_step_mse(x)
                    
                    ml_loss = 0
                    p = 0
                    j = 0
                    
                    
                else:
                    
                    # ML training of the bijective network after ml threshold epochs
                    if ml_threshold == 0:
                        mse_loss = 0
                    t = np.random.randint(0, num_iters)
                    x = training_images[t*batch_size:(t+1)*batch_size]
                    z_batch, _ = model(x, reverse= False)
                    ml_loss , p , j = train_step_ml(z_batch)
                    
                if epoch == 0 and i == 0:
                    
                    # Just for the first iteration of the first epoch
                    # to calculate the number of trainable parametrs
                    with tf.GradientTape() as tape:
                        
                        z_batch1 , _ = model(x, reverse= False)
                        variables_model = tape.watched_variables()
                    
                    with tf.GradientTape() as tape:
                        
                        _, _ = latent_model(z_batch1, reverse=False)
                        variables_latent_model = tape.watched_variables()
                        
                    parameters_model = np.sum([np.prod(v.get_shape().as_list()) 
                        for v in variables_model])
                    parameters_latent_model = np.sum([np.prod(v.get_shape().as_list()) 
                        for v in variables_latent_model])
                    
                    print('Number of trainable_parameters of model: {}'.format(parameters_model))
                    print('Number of trainable_parameters of latent model: {}'.format(parameters_latent_model))
                    print('Total number of trainable_parameters: {}'.format(parameters_model + parameters_latent_model))
                
            
            
            MSE_train_loss_metric.update_state(mse_loss)
            ML_loss_metric.update_state(ml_loss)
            pz_metric.update_state(p)
            jacobian_metric.update_state(j)
            
            sample_number = 25 # Number of samples to show
        
            z_hat_test = model(testing_images[:sample_number], reverse= False)[0] 
            # Low dimensinal representation of testing images
            x_hat_test = model(z_hat_test , reverse = True)[0]
            # Reconstrcted testing images
            
            test_mse = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(tf.square(testing_images[:sample_number] - x_hat_test) ,
                                                       axis = [1,2,3]))/tf.math.sqrt(tf.reduce_sum(tf.square(testing_images[:sample_number]) , axis = [1,2,3])))
            # MSE of reconstrcution test samples
            MSE_test_loss_metric.update_state(test_mse)


            image_size = 64 if dataset == 'celeba' or dataset == 'imagenet' or dataset == 'church' or dataset == 'rheo' else 32 
            image_size = 128 if dataset == 'chest' else image_size

            c = 3 if dataset=='celeba' or dataset=='imagenet' or dataset=='church' or dataset =='rheo' else 1
            
            
            x_generated = model(model(training_images[:sample_number],
                                      reverse= False)[0] , reverse = True)[0].numpy()
            # Reconstructed training images
            
            x_generated_test = x_hat_test.numpy()[:sample_number]
            # Reconstrcted testing images
            x_original_test = testing_images.numpy()[:sample_number]
            # Ground truth testing images
            
            z_batch, _ = model(training_images[:sample_number], reverse= False)
            z1 = z_batch[:1]
            z2 = z_batch[1:2]
            xinterp = latent_space_interplotion(model, z1, z2, latent=True , sample_number = sample_number)

            # Sampling from distribution
            z_random_base = pz.prior.sample(sample_number) # sampling from base (gaussian) with Temprature = 1
            z_random_base_T = (z_random_base - pz.mu) * T + pz.mu # sampling from base (gaussian) with Temprature = T
            z_random = latent_model(z_random_base , reverse = True)[0] # Intermediate samples with Temprature = 1
            z_random_T = latent_model(z_random_base_T , reverse = True)[0] # Intermediate samples with Temprature = T
            x_sampled = model(z_random , reverse = True)[0].numpy() # Samples with Temprature = 1
            x_sampled_T = model(z_random_T , reverse = True)[0].numpy() # Samples with Temprature = T
            
            
            # Saving experiment results
            samples_folder = os.path.join(exp_path, 'Generated_samples')
            if not os.path.exists(samples_folder):
                os.mkdir(samples_folder)
            image_path_inverse_train = os.path.join(
                samples_folder, 'inverse_train')
            image_path_inverse_test = os.path.join(
                samples_folder, 'inverse_test')

            if not os.path.exists(image_path_inverse_train):
                os.mkdir(image_path_inverse_train)

            if not os.path.exists(image_path_inverse_test):
                os.mkdir(image_path_inverse_test)

            ngrid = int(np.sqrt(sample_number))

            cv2.imwrite(os.path.join(image_path_inverse_train, 'epoch %d.png' % (epoch,)),
                        x_generated[:, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2)
                .reshape(ngrid*image_size, -1, c)*127.5 + 127.5) # Reconstructed training images

            cv2.imwrite(os.path.join(image_path_inverse_test, 'epoch %d.png' % (epoch,)),
                        x_generated_test[:, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2)
                .reshape(ngrid*image_size, -1, c)*127.5 + 127.5) # Reconstructed test images
            
            
            cv2.imwrite(os.path.join(image_path_inverse_test, 'original_epoch %d.png' % (epoch,)),
                        x_original_test[:, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2)
                .reshape(ngrid*image_size, -1, c)* 127.5 + 127.5) # Ground truth test images
            
            image_path_sampled = os.path.join(samples_folder, 'sampled')
            if os.path.exists(image_path_sampled) == False:
                os.mkdir(image_path_sampled)

            cv2.imwrite(os.path.join(image_path_sampled, 'sampled_epoch %d.png' % (epoch,)),
                        x_sampled[:, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2)
                .reshape(ngrid*image_size, -1, c)*127.5 + 127.5) # samples from distribution with Temprature = 1
            
            cv2.imwrite(os.path.join(image_path_sampled, 'Tempreture_sampled_epoch %d.png' % (epoch,)),
                        x_sampled_T[:, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2)
                .reshape(ngrid*image_size, -1, c)*127.5 + 127.5) # samples from distribution with Temprature = T
        

            cv2.imwrite(os.path.join(image_path_sampled, 'interp_epoch %d.png' % (epoch,)),
                        xinterp[:, :, :, ::-1].reshape(
                ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2)
                .reshape(ngrid*image_size, -1, c)*127.5 + 127.5) # Interpolation images between two test images

            
            # Saving logs
            with MSE_train_summary_writer.as_default():
                tf.summary.scalar(
                    'MSE_train', MSE_train_loss_metric.result(), step=epoch)

            with MSE_test_summary_writer.as_default():
                tf.summary.scalar(
                    'MSE_test', MSE_test_loss_metric.result(), step=epoch)

            with ML_summary_writer.as_default():
                tf.summary.scalar(
                    'ML_loss', ML_loss_metric.result(), step=epoch)

            
            with pz_summary_writer.as_default():
                tf.summary.scalar(
                    'pz', pz_metric.result(), step=epoch)
                
            
            with jacobian_summary_writer.as_default():
                tf.summary.scalar(
                    'jacobian', jacobian_metric.result(), step=epoch)
                
            
            print("Epoch {:03d}: MSE train: {:.3f} / MSE test: {:.3f} / ML Loss: {:.3f} "
                  .format(epoch, MSE_train_loss_metric.result().numpy(), MSE_test_loss_metric.result().numpy(),
                          ML_loss_metric.result().numpy()))
            
            

            MSE_train_loss_metric.reset_states()
            MSE_test_loss_metric.reset_states()
            ML_loss_metric.reset_states()
            pz_metric.reset_states()
            jacobian_metric.reset_states()

            save_path = manager.save()
            print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))
            
            epoch_end = time()
            time_vector[epoch] = epoch_end - epoch_start
            np.save(os.path.join(exp_path, 'time_vector.npy') , time_vector)
            print('epoch time:{}'.format(time_vector[epoch]))
            
    if calc_logdet:
        zs = pz.prior.sample(1000)
        ld = pz.prior.log_prob(zs)
        f = lambda z: model(latent_model(z, reverse=True)[0], reverse=True)[0]

        ld -= wrapper_logdet(zs, f)/2.0

        ld = ld.numpy()
        print(ld)

        print('logdet stats')
        print('Mean: %f'%np.mean(ld))
        print('std: %f'%np.std(ld))


    if run_inv:
        operator = inv_probs.RandomGaussian(n_measurements=250)
        inv_probs.solve_inv_problem(testing_images[:25], 
            'inv_probs/', dataset,  operator, snr, model, latent_model, 
            pz=None, ckpt_obj=ckpt, ckpt_manager=manager)
        print('Finished the Random Gaussian problem.')

        operator = inv_probs.CT(n=30)
        inv_probs.solve_inv_problem(testing_images[:25], 
            'inv_probs/', dataset,  operator, snr, model, latent_model, 
            pz=None, ckpt_obj=ckpt, ckpt_manager=manager)
        print('Finished the CT problem.')

        operator = inv_probs.Mask(size=10)
        inv_probs.solve_inv_problem(testing_images[:25], 
            'inv_probs/', dataset,  operator, snr, model, latent_model, 
            pz=None, ckpt_obj=ckpt, ckpt_manager=manager)
        print('Finished the image completion problem.')

        operator = inv_probs.SuperResolution(r=4)
        inv_probs.solve_inv_problem(testing_images[:25], 
            'inv_probs/', dataset,  operator, 100, model, latent_model, 
            pz=None, ckpt_obj=ckpt, ckpt_manager=manager)
        print('Finished the super-resolution problem.')

        operator = inv_probs.RandomMask(prob_to_keep=0.15)
        inv_probs.solve_inv_problem(testing_images[:25], 
            'inv_probs/', dataset,  operator, snr, model, latent_model, 
            pz=pz, ckpt_obj=ckpt, ckpt_manager=manager)
        print('Finished the Random Mask problem.')

        operator = inv_probs.RandomMask(prob_to_keep=0.20)
        inv_probs.solve_inv_problem(testing_images[:25], 
            'inv_probs/', dataset,  operator, snr, model, latent_model, 
            pz=pz, ckpt_obj=ckpt, ckpt_manager=manager)
        print('Finished the Random Mask problem.')

    if train_posterior:
        post_path = os.path.join(exp_path, 'posterior/')
        os.makedirs(post_path, exist_ok=True)

        latent_dim = 192 if dataset!='mnist' else 64
        latent_dim = 256 if dataset=='chest' else latent_dim
        operator = inv_probs.CT(n=60)
        
        ## dummy forward op to get size of measurements
        print('dummy forward to init weights')
        out = operator(training_images[:5])
        c = tf.shape(out)[-1]

        post = posterior(input_dim=latent_dim, output_dim=c, mid_units=128, 
            depth=6, layer_type='additive')

        ## save ground truth image
        gt_img = testing_images[0,:,:,::-1].numpy()
        image_size,_,ch = gt_img.shape
        cv2.imwrite(os.path.join(post_path, 'gt.png'),
                    gt_img*127.5 + 127.5)

        true_z = latent_model(model(testing_images[:1], reverse=False)[0], 
            reverse=False)[0]
        print('True z norm')
        print(tf.linalg.norm(true_z, axis=-1))

        print('Snr is %f'%snr)


        ## init posterior
        dummy_z = pz.prior.sample(1000)
        z, _ = post(dummy_z, reverse=False)
        print(tf.reduce_mean(tf.linalg.norm(dummy_z, axis=-1)))
        print(tf.reduce_mean(tf.linalg.norm(z, axis=-1)))

        measurement = operator(testing_images[:1])
        dim_y = tf.shape(measurement)[1]

        ckpt = tf.train.Checkpoint(post=post)
        manager = tf.train.CheckpointManager(
            ckpt, post_path, max_to_keep=5)

        t_prior = tfd.MultivariateNormalDiag(
            tf.zeros(latent_dim), tf.ones(latent_dim)/np.sqrt(latent_dim))

        ut = t_prior.sample(50)
       
        dummy_z = pz.prior.sample(1000) 
        z, _ = post(ut, reverse=True)

        warmup_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        def warmup():
            ut = t_prior.sample(50)
            dummy_z = pz.prior.sample(1000)
            with tf.GradientTape() as tape:
                z, _ = post(ut, reverse=True)

                loss = tf.reduce_mean(pz.prior.log_prob(z))

                grads = tape.gradient(loss, tape.watched_variables())
                warmup_optimizer.apply_gradients(zip(grads, tape.watched_variables()))

            return loss

        for i in range(5):
            loss = warmup()
            print(i)
            print(loss)


        post_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        ## training
        @tf.function
        def train_step_posterior(measurement):
            noise_sigma = 10**(-snr/20.0)*tf.reduce_mean(
                tf.linalg.norm(measurement, axis=-1))

            noise = tf.random.normal((25, dim_y))
            noise /= tf.linalg.norm(noise, axis=-1, keepdims=True)
            noise *= noise_sigma
            noisy_measurement = measurement + noise

            beta = 0.01

            with tf.GradientTape() as tape:
                tape.watch(post.trainable_variables)
                ts = t_prior.sample(25)
                ut, logdet_term = post(ts, reverse=True)
                pz_term = pz.prior.log_prob(ut)

                obj = logdet_term*beta - pz_term

                image = model(latent_model(ut, reverse=True)[0], reverse=True)[0]

                mse_loss = tf.reduce_sum(tf.square(noisy_measurement - operator(image)))/25.0
                nll_loss = tf.reduce_sum(obj)/25.0*noise_sigma**2
                loss = mse_loss + nll_loss
                grads = tape.gradient(loss, post.trainable_variables)
                post_optimizer.apply_gradients(zip(grads, post.trainable_variables))

            return loss, mse_loss, nll_loss

        N = 32000

        for epoch in range(N):
            loss, mse_loss, nll_loss = train_step_posterior(measurement)

            if (epoch)%100 == 0:
                print('[%d/%d] loss: %f, %f, %f'%(epoch, N, 
                    loss.numpy(),mse_loss.numpy(),nll_loss.numpy()))

            if (epoch)%1000 == 0:
                save_path = manager.save()
                print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

                
                ts = t_prior.sample(25)
                ut, _ = post(ts, reverse=False)
                image = model(latent_model(ut, reverse=True)[0], reverse=True)[0]

                noise_sigma = 10**(-snr/20.0)*tf.reduce_mean(
                    tf.linalg.norm(measurement, axis=-1))

                noise = tf.random.normal((25, dim_y))
                noise /= tf.linalg.norm(noise, axis=-1, keepdims=True)
                noise *= noise_sigma
                noisy_measurement = measurement + noise

                pinv = operator.T(noisy_measurement).numpy()
                cv2.imwrite(os.path.join(post_path, 'pinv_recon.png'),
                            pinv[:, :, :, ::-1].reshape(
                    5, 5,
                    image_size, image_size, ch).swapaxes(1, 2)
                    .reshape(5*image_size, -1, ch)*127.5 + 127.5)


                cv2.imwrite(os.path.join(post_path, 'epoch_%d.png' % (epoch,)),
                            image[:, :, :, ::-1].numpy().reshape(
                    5, 5,
                    image_size, image_size, ch).swapaxes(1, 2)
                    .reshape(5*image_size, -1, ch)*127.5 + 127.5)



if __name__ == '__main__':
    train(num_epochs,
          batch_size,
          dataset,
          lr,
          exp_path)
