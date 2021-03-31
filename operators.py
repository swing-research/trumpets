from tqdm import tqdm
import numpy as np
import os
import cv2
import tensorflow as tf
import tensorflow.keras.layers as layers
from skimage.transform import radon, iradon

def generate_radon_op(n, n_measurements):
    """Makes a radon forward op"""
    A = np.zeros((n*n_measurements, n**2))
    theta = np.linspace(30.0, 150.0, n_measurements, endpoint=False)
    for i in range(n**2):
        img = np.zeros(n**2)
        img[i] = 1
        img = img.reshape(n,n)
        t = radon(img, theta=theta, circle=True).reshape(-1)
        A[:,i]= t
        
    return A


class Operator(layers.Layer):
    """Base class of operators"""
    def __init__(self):
        super(Operator, self).__init__()
        self.opname = 'opname'

    def call(self, x):
        return x

    def T(self, x):
        return x

    def save(self, path):
        return None


class CT(layers.Layer):
    """Builds a limited-view imaging operator"""
    def __init__(self, n=25):
        super(CT, self).__init__()
        self.n = n
        self.opname = 'CT_%d'%n


    def build(self, input_shape):
        _,h,_,c = input_shape
        self.in_shape = input_shape
        if os.path.exists('radon%d_%d.npy'%(self.n, h)):
            w = np.load('radon%d_%d.npy'%(self.n, h))
        else:
            w = tf.convert_to_tensor(generate_radon_op(h, self.n), tf.float32)
            if c>1:
                w = tf.concat((w, w, w), axis=-1)
            np.save('radon%d_%d.npy'%(self.n, h), w)
            
        winv = tf.linalg.pinv(w)
        self.w = w
        self.winv = winv


    def call(self, x):
        bs = tf.shape(x)[0]
        return tf.matmul(tf.reshape(x, (bs, -1)), 
            self.w, transpose_b=True)

    def T(self, x):
        bs = tf.shape(x)[0]
        y = tf.matmul(tf.reshape(x, (bs, -1)), 
            self.winv, transpose_b=True)
        return tf.reshape(y, (bs,) + self.in_shape[1:])

    def save(self, path):
        np.save(os.path.join(path, 'radon'), self.w.numpy())
        return None

class SuperResolution(Operator):
    """Builds a donwsampling operator"""

    def __init__(self, r = 2):
        super(SuperResolution, self).__init__()
        self.r = r
        self.opname = 'srx%d'%r
        self.upsample = tf.keras.layers.UpSampling2D(size=(r,r), 
            interpolation='nearest')

    def call(self, x):
        _, h, w, ch = x.shape
        size = (h//self.r, w//self.r)
        return tf.image.resize(x, size, antialias=True)

    def T(self, x):
        return self.upsample(x)

    def save(self, path):
        return None


class RandomMask(Operator):
    """Random masking operator, (channelwise-consistent)"""

    def __init__(self, prob_to_keep=0.1):
        super(RandomMask, self).__init__()
        self.p = prob_to_keep
        self.opname = 'randmask_p_%f'%self.p


    def build(self, input_shape):
        h,w = input_shape[1:3]
        shape = (1,h,w,1)
        self.mask = tf.cast(tf.random.uniform(shape) < self.p, tf.float32)

    def call(self, x):
        return x*self.mask

    def T(self, x):
        return x*self.mask

    def save(self, path):
        np.save(os.path.join(path, 'mask'), self.mask.numpy())
        return None

class Mask(Operator):
    """Builds a square mask operator in the center of the image"""

    def __init__(self, size=20):
        super(Mask, self).__init__()
        self.size = size
        self.opname = 'mask_size_%f'%self.size

    def build(self, input_shape):
        h,w = input_shape[1:3]
        shape = (1,h,w,1)
        sh = h//2 - self.size//2
        eh = sh + self.size 
        sw = w//2 - self.size//2
        ew = sw + self.size 
        mask = np.ones(shape, dtype=np.float32)
        mask[0,sh:eh, sw:ew,0] *= 0.0

        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    def call(self, x):
        return x*self.mask

    def T(self, x):
        return x*self.mask

    def save(self, path):
        np.save(os.path.join(path, 'mask'), self.mask.numpy())
        return None

class RandomGaussian(Operator):
    """Builds a CS random gaussian measurement operator"""

    def __init__(self, n_measurements=100):
        super(RandomGaussian, self).__init__()
        self.n = n_measurements
        self.opname = 'randgauss_%d'%self.n


    def build(self, input_shape):
        h,w,ch = input_shape[1:]
        self.shape = h, w, ch
        self.w = tf.random.normal((h*w*ch, self.n))/np.sqrt(self.n)
        self.winv = tf.linalg.pinv(self.w)

    def call(self, x):
        b = tf.shape(x)[0]
        return tf.linalg.matmul(tf.reshape(x, (b, -1)),self.w)

    def T(self, y):
        bs, _ = y.shape
        xhat = tf.linalg.matmul(y, self.winv)
        return tf.reshape(xhat, (bs,)+self.shape)

    def save(self, path):
        np.save(os.path.join(path, 'weights'), self.w.numpy())
        return None


class InjFlow_PGD(object):
    """Builds a solver"""
    def __init__(self, flow, encoder, operator, 
                 nsteps=1000, 
                 latent_dim = 192,
                 sample_shape = (64,64,3),
                 learning_rate=1e-2):

        self.op = operator

        self.flow = flow
        self.encoder = encoder

        self.nsteps = nsteps
        self.latent_dim = latent_dim
        self.sample_shape = sample_shape
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate)

    def csgm(self, measurements, initial_points=None, restarts=10):
        """Bora et al"""
        bs = tf.shape(measurements)[0]
        latent_dim = self.latent_dim

        losses = np.zeros((restarts,), dtype=np.float32)
        guesses = []

        for k in range(restarts): 
            if initial_points is None:
                z_guess = tf.Variable(tf.random.normal((bs,latent_dim))/np.sqrt(latent_dim), trainable=True)
            else:
                z_guess = tf.Variable(initial_points, trainable=True)

            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


            with tqdm(total=self.nsteps) as pbar:

                for i in range(self.nsteps):
                    with tf.GradientTape() as tape:
                        x = self.encoder(z_guess, reverse=True)[0]
                        loss = tf.reduce_sum(tf.square(self.op(x)-measurements)) #+ 1e-3*tf.reduce_sum(z_guess**2)
                        grads = tape.gradient(loss, [z_guess])
                        optimizer.apply_gradients(zip(grads, [z_guess]))
                    pbar.set_description('Loss: %1.3f '%(loss.numpy()))
                    pbar.update(1)

            losses[k] =  loss.numpy()
            guesses.append(self.encoder(z_guess, reverse=True)[0])

        return guesses[np.argmin(losses)]

    def hegde(self, measurements, initial_points=None):
        """Shah and Hegde"""
        latent_dim = self.latent_dim
        bs = tf.shape(measurements)[0]
        inner_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        x_guess = tf.Variable(tf.zeros_like(self.op.T(measurements)), trainable=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        with tqdm(total=self.nsteps) as pbar:

            for i in range(self.nsteps):
                for _ in range(10):
                    if initial_points is None:
                        z_guess = tf.Variable(tf.random.normal((bs,latent_dim)), trainable=True)
                    else:
                        z_guess = tf.Variable(initial_points, trainable=True)
                    with tf.GradientTape() as tape1:
                        x = self.encoder(z_guess, reverse=True)[0]
                        loss = tf.reduce_sum(tf.square(x - x_guess))
                        grads = tape1.gradient(loss, [z_guess])
                        inner_opt.apply_gradients(zip(grads, [z_guess]))

                with tf.GradientTape() as tape:
                    outer_loss = tf.reduce_sum(tf.square(self.op(x_guess) - measurements))
                    grads = tape.gradient(outer_loss, [x_guess])
                    optimizer.apply_gradients(zip(grads, [x_guess]))

                x_guess.assign(x)
                pbar.set_description('Loss: %1.3f '%(outer_loss.numpy()))
                pbar.update(1)

        return x_guess

    def dip(self, measurements, initial_points=None):
        """Deep image prior"""
        latent_dim = self.latent_dim
        bs = tf.shape(measurements)[0]
        
        if initial_points is None:
            z_guess = tf.Variable(tf.random.normal((bs,latent_dim)), trainable=False)
        else:
            z_guess = tf.Variable(initial_points, trainable=False)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        with tqdm(total=800) as pbar:

            for i in range(800):
                with tf.GradientTape() as tape:
                    tape.watch(self.encoder.trainable_variables)
                    x = self.encoder(z_guess, reverse=True)[0]
                    loss = tf.reduce_sum(tf.square(self.op(x)-measurements))
                    grads = tape.gradient(loss, tape.watched_variables())
                    optimizer.apply_gradients(zip(grads, tape.watched_variables()))

                pbar.set_description('Loss: %1.3f '%(loss.numpy()))
                pbar.update(1)

        return x

    def __call__(self, measurements, lam=1e-3):
        """iFlow-L, when lam!=0 and iFlow when lam=0"""

        def projection(x):
            z, rev_obj = self.encoder(x, reverse=False)
            zhat, flow_obj = self.flow(z, reverse=False)
            flow_obj = self.flow.log_prob(zhat)
            proj_x, fwd_obj = self.encoder(z, reverse=True)
            return proj_x, fwd_obj + flow_obj
 

        # initialize with A.T @ y
        x_guess = tf.Variable(self.op.T(measurements), trainable=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

        with tqdm(total=self.nsteps) as pbar:

            for i in range(self.nsteps):

                with tf.GradientTape() as tape:
                    proj_x_guess, likelihood = projection(x_guess)
                    loss1 = tf.reduce_sum(tf.square(self.op(proj_x_guess)-measurements))
                    loss2 = tf.reduce_sum(likelihood)
                    # loss2 = loss1*0
                    loss = loss1 + lam*loss2
                    grads = tape.gradient(loss, [x_guess])
                    optimizer.apply_gradients(zip(grads, [x_guess]))

                pbar.set_description('Loss: %1.3f, NLL: %1.3f, Data: %1.3f'%(
                    loss.numpy(), loss2.numpy(), loss1.numpy()))
                pbar.update(1)

        return projection(x_guess)[0]


def solve_inv_problem(
    testing_images,
    root, 
    dataset, 
    operator,
    measurement_snr,
    model,
    latent_model,
    pz=None,
    ckpt_obj=None,
    ckpt_manager=None):
    """
    Args:
        testing_images (tf.Tensor): ground truth images (x) in the inverse problem
        root (str): root directory in which to save inverse problem results
        dataset (str): Name of the dataset
        operator (Operator): forward operator
        measurement_snr (float): snr of the measurement vector
        model (tf.keras.Model): the injective part of trumpet
        latent_model (tf.keras.Model): the bijective part of trumpet
        pz (None, tf.distributions): the prior distribution
        ckpt_obj (None, optional): tf checkpoint pointing to the saved model
        ckpt_manager (None, optional): tf checkpoint manager storing the saved model attributes
    """

    ngrid = 5
    image_size = 32 if dataset != 'celeba' else 64
    image_size = 128 if dataset == 'chest' else image_size
    
    c = 1 if dataset == 'mnist' else 3
    c = 1 if dataset == 'chest' else c

    bs = tf.shape(testing_images)[0]

    prob_folder = os.path.join(root, '%s_%s'%(dataset, operator.opname))
    if not os.path.exists(prob_folder):
        os.makedirs(prob_folder, exist_ok=True)


    x_sampled = model(model(testing_images, reverse=False)[0], reverse=True)[0].numpy()

    cv2.imwrite(os.path.join(prob_folder, 'test_load.png'),
                x_sampled[:, :, :, ::-1].reshape(
        ngrid, ngrid,
        image_size, image_size, c).swapaxes(1, 2)
        .reshape(ngrid*image_size, -1, c)*127.5 + 127.5)


    latent_dim = 64 if dataset=='mnist' else 192
    solver = InjFlow_PGD(latent_model, model, operator, 
        latent_dim=latent_dim, learning_rate=1e-3)
    measurements = solver.op(testing_images[:ngrid**2])

    n_snr = measurement_snr
    noise_sigma = 10**(-n_snr/20.0)*tf.reduce_mean(tf.linalg.norm(
        tf.reshape(measurements, (ngrid**2, -1)), axis=-1))
    noise = tf.random.normal(tf.shape(measurements))*noise_sigma
    measurements = measurements + noise


    cv2.imwrite(os.path.join(prob_folder, 'gt.png'),
                testing_images[:, :, :, ::-1].numpy().reshape(
        ngrid, ngrid,
        image_size, image_size, c).swapaxes(1, 2)
        .reshape(ngrid*image_size, -1, c)*127.5 + 127.5)

    cv2.imwrite(os.path.join(prob_folder, 'init.png'),
                operator.T(measurements).numpy()[:, :, :, ::-1].reshape(
        ngrid, ngrid,
        image_size, image_size, c).swapaxes(1, 2)
        .reshape(ngrid*image_size, -1, c)*127.5 + 127.5)

    injflow_result = solver(measurements, lam=1e-2) 
    injflow_path = os.path.join(prob_folder, 'injflow_result.png')
    cv2.imwrite(injflow_path,
                injflow_result[:, :, :, ::-1].numpy().reshape(
        ngrid, ngrid,
        image_size, image_size, c).swapaxes(1, 2)
        .reshape(ngrid*image_size, -1, c)*127.5 + 127.5)

    injflow_result = solver(measurements, lam=0) 
    injflow_path = os.path.join(prob_folder, 'injflow_result_wo_likelihood.png')
    cv2.imwrite(injflow_path,
                injflow_result[:, :, :, ::-1].numpy().reshape(
        ngrid, ngrid,
        image_size, image_size, c).swapaxes(1, 2)
        .reshape(ngrid*image_size, -1, c)*127.5 + 127.5)


    latent_samples = latent_model(latent_model.pz.prior.sample(bs), reverse=True)[0]
    csgm_result = solver.csgm(measurements, initial_points=latent_samples) 
    csgm_path = os.path.join(prob_folder, 'csgm_result.png')
    cv2.imwrite(csgm_path,
                csgm_result[:, :, :, ::-1].numpy().reshape(
        ngrid, ngrid,
        image_size, image_size, c).swapaxes(1, 2)
        .reshape(ngrid*image_size, -1, c)*127.5 + 127.5)

    latent_samples = latent_model(latent_model.pz.prior.sample(bs), reverse=True)[0]
    dip_result = solver.dip(measurements, initial_points=latent_samples) 
    dip_path = os.path.join(prob_folder, 'dip_result.png')
    cv2.imwrite(dip_path,
                dip_result[:, :, :, ::-1].numpy().reshape(
        ngrid, ngrid,
        image_size, image_size, c).swapaxes(1, 2)
        .reshape(ngrid*image_size, -1, c)*127.5 + 127.5)

    ckpt_obj.restore(ckpt_manager.latest_checkpoint)

    # hegde_result = solver.hegde(measurements, initial_points=latent_samples)
    # hegde_path = os.path.join(prob_folder, 'hegde_result.png')
    # cv2.imwrite(hegde_path,
    #             hegde_result[:, :, :, ::-1].numpy().reshape(
    #     ngrid, ngrid,
    #     image_size, image_size, c).swapaxes(1, 2)
    #     .reshape(ngrid*image_size, -1, c)*127.5 + 127.5)

    
    solver.op.save(prob_folder)


def unit_test_operator():
    x = tf.random.uniform((1, 32, 32, 5))
    A = RandomMask(0.2)
    y = A(x)
    z = A.T(y)
    
    print(y.shape)

    print(tf.linalg.norm(y - z))


if __name__ == '__main__':
    unit_test_operator()
