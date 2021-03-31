import tensorflow as tf
from tensorflow.keras import layers
import scipy
import numpy as np
from Unet_util import Unet


class upsqueeze(layers.Layer):
    def __init__(self, factor=2):
        super(upsqueeze, self).__init__()
        self.f = factor

    def call(self, x, reverse=False):
        f = self.f

        # upsampling via squeeze
        b, N1, N2, nch = x.get_shape().as_list()
        if not reverse:
            x = tf.reshape(
                tf.transpose(
                    tf.reshape(x, shape=[b, N1//f, f, N2//f, f, nch]),
                    [0, 1, 3, 2, 4, 5]),
                [b, N1//f, N2//f, nch*f*f])
        else:
            x = tf.reshape(tf.transpose(
                tf.reshape(x, shape=[b, N1, N2, f, f, nch//f**2]),
                [0, 1, 3, 2, 4, 5]), [b, N1*f, N2*f, nch//f**2])


        return x, 0.0

    def jvp(self, v):
        """calculates jvp when reverse=True"""
        return self.call(v, reverse=True)

    def vjp(self, v):
        """calculates jvp when reverse=True"""
        # note that this is just a reshape op and as such its Jacobian 
        # is a unit-norm op that changes the shape of a vector.
        return self.call(v, reverse=False)


class actnorm(layers.Layer):
    """Activation normalization layers that 
    initialized via data"""

    def __init__(self, **kwargs):
        super(actnorm, self).__init__()
        
        # assign checks for first call
        self.assigned = False

    def build(self, input_shape):
        if len(input_shape) == 2:
            self.b = self.add_weight(name='bias',
                                     shape=(1, input_shape[1]),
                                     trainable= True)
            self.scale = self.add_weight(name='scale',
                                         shape=(1, input_shape[1]),
                                         trainable= True)
        else:
            self.b = self.add_weight(name='bias',
                                     shape=(1, 1, 1, input_shape[3]),
                                     trainable= True)
            self.scale = self.add_weight(name='scale',
                                         shape=(1, 1, 1, input_shape[3]),
                                         trainable= True)

    def call(self, x, reverse=False):
        if len(x.shape) == 2:
            red_axes = [0]
            dim = x.get_shape().as_list()[-1]
        else:
            red_axes = [0, 1, 2]
            _, height, width, channels = x.get_shape().as_list()
            dim = height*width
        
        if not self.assigned:
            """https://github.com/tensorflow/tensor2tensor/blob/21dba2c1bdcc7ab582a2bfd8c0885c217963bb4f/tensor2tensor/models/research/glow_ops.py#L317"""
            self.b.assign(-tf.reduce_mean(x, red_axes, keepdims=True))

            x_var = tf.reduce_mean((x+self.b)**2, red_axes, keepdims=True)
            init_value = tf.math.log(1.0/(tf.math.sqrt(x_var) + 1e-6))


            self.scale.assign(init_value)
            self.assigned = True


        if not reverse:
            x += self.b
            x *= tf.math.exp(self.scale)

        else:
            x *= tf.math.exp(-self.scale)
            x -= self.b
        
        log_s = self.scale
        dlogdet = tf.reduce_sum(log_s)* \
            tf.cast(dim, log_s.dtype)
        if reverse:
            dlogdet *= -1

        return x, dlogdet

    def jvp(self, v):
        """Calculates the jacobian-vector product with reverse=True"""
        # here J is a diagonal matrix and therefore, v^T J or Jv will 
        # be the same, except the transpose. 

        return v*tf.math.exp(-self.scale)

    def vjp(self, v):
        """Calculates the vector-Jacobian product with reverse=True"""
        return v*tf.math.exp(-self.scale)





class invertible_1x1_conv(layers.Layer):
    """Invertible 1x1 convolutional layers"""

    def __init__(self, **kwargs):
        super(invertible_1x1_conv, self).__init__()
        self.type = kwargs.get('op_type', 'bijective')
        self.gamma = kwargs.get('gamma', 0.0)
        self.activation = kwargs.get('activation', 'linear')
        
    def build(self, input_shape):
        _, height, width, channels = input_shape
        
        if self.type=='bijective':
            random_matrix = np.random.randn(channels, channels).astype("float32")
            np_w = scipy.linalg.qr(random_matrix)[0].astype("float32")
            self.activation = 'linear'
            
        else:
            if self.activation == 'linear':
                random_matrix_1 = np.random.randn(channels//2, channels//2).astype("float32")
                random_matrix_2 = np.random.randn(channels//2, channels//2).astype("float32")
                np_w_1 = scipy.linalg.qr(random_matrix_1)[0].astype("float32")
                np_w_2 = scipy.linalg.qr(random_matrix_2)[0].astype("float32")
                np_w = np.concatenate([np_w_1, np_w_2], axis=0)/(np.sqrt(2.0))
                
            elif self.activation == 'relu':
                random_matrix_1 = np.random.randn(channels//2, channels//2).astype("float32")
                np_w = scipy.linalg.qr(random_matrix_1)[0].astype("float32")
        self.w = tf.Variable(np_w, name='W', trainable=True)


    def call(self, x, reverse=False):
        # If height or width cannot be statically determined then they end up as
        # tf.int32 tensors, which cannot be directly multiplied with a floating
        # point tensor without a cast.
        _, height, width, channels = x.get_shape().as_list()
        s = tf.linalg.svd(self.w, 
            full_matrices=False, compute_uv=False)
        
        
        log_s = tf.math.log(s + self.gamma**2/(s + 1e-8))
        objective = tf.reduce_sum(log_s) * \
            tf.cast(height * width, log_s.dtype)
    
        if not reverse:
            
            if self.activation == 'relu':
                x = x[:,:,:,:channels//2] - x[:,:,:,channels//2:]
            w = tf.reshape(self.w , [1, 1] + self.w.get_shape().as_list())
            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format="NHWC")
        
        
        else:
            
            if self.activation=='relu':
                prefactor = tf.matmul(self.w, self.w, transpose_a=True) + \
                    self.gamma**2*tf.eye(tf.shape(self.w)[1])
                
                w_inv = tf.matmul(tf.linalg.inv(prefactor), self.w, transpose_b=True)
                conv_filter = tf.concat([w_inv, -w_inv], axis=1)
                conv_filter = tf.reshape(conv_filter, [1, 1] + conv_filter.get_shape().as_list())
                x = tf.nn.conv2d(x, conv_filter, [1, 1, 1, 1], "SAME", data_format="NHWC")
                x = tf.nn.relu(x)
            
            else:
                prefactor = tf.matmul(self.w, self.w, transpose_a=True) + \
                    self.gamma**2*tf.eye(tf.shape(self.w)[1])

                w_inv = tf.matmul(  tf.linalg.inv(prefactor) , self.w, transpose_b=True)
                conv_filter = w_inv
                conv_filter = tf.reshape(conv_filter, [1, 1] + conv_filter.get_shape().as_list())
                x = tf.nn.conv2d(x, conv_filter, [1, 1, 1, 1], "SAME", data_format="NHWC")
            
            objective *= -1
        return x, objective

    def jvp(self, v):
        """Calculates the jacobian-vector product with reverse=True"""
        # here J is a diagonal matrix and therefore, v^T J or Jv will 
        # be the same, except the transpose. 

        return self.call(v, reverse=True)

    def vjp(self, v):
        """Calculates the vector-Jacobian product with reverse=True"""
        ## here we just transpose the pseudo-inverted filter and apply to v
        if self.activation=='relu':
            prefactor = tf.matmul(self.w, self.w, transpose_a=True) + \
                self.gamma**2*tf.eye(tf.shape(self.w)[1])
            
            w_inv = tf.transpose(tf.matmul(tf.linalg.inv(prefactor), self.w, transpose_b=True))
            conv_filter = tf.concat([w_inv, -w_inv], axis=1)
            conv_filter = tf.reshape(conv_filter, [1, 1] + conv_filter.get_shape().as_list())
            out = tf.nn.conv2d(v, conv_filter, [1, 1, 1, 1], "SAME", data_format="NHWC")
            out = tf.nn.relu(out)
        
        else:
            prefactor = tf.matmul(self.w, self.w, transpose_a=True) + \
                self.gamma**2*tf.eye(tf.shape(self.w)[1])

            w_inv = tf.transpose(tf.matmul(tf.linalg.inv(prefactor) , self.w, transpose_b=True))
            conv_filter = w_inv
            conv_filter = tf.reshape(conv_filter, [1, 1] + conv_filter.get_shape().as_list())
            out = tf.nn.conv2d(out, conv_filter, [1, 1, 1, 1], "SAME", data_format="NHWC")

        return out


class conv_stack(layers.Layer):
    def __init__(self, mid_channels,
                 output_channels, activation="relu"):
        super(conv_stack, self).__init__()

        self.conv1 = layers.Conv2D(
            mid_channels, 3, 1, padding='same',
            activation=activation, use_bias=False)
        self.conv2 = layers.Conv2D(
            mid_channels, 1, 1, padding='same',
            activation=activation, use_bias=False)
        self.conv3 = layers.Conv2D(
            output_channels, 1, 1, padding='same',
            activation=None, use_bias=False)

    def call(self, x , training = True):
        return self.conv3(self.conv2(self.conv1(x)))


class additive_coupling(layers.Layer):
    def __init__(self, mid_channels=128, activation="relu"):
        super(additive_coupling, self).__init__()
        self.activation = activation
        self.mid_channels = mid_channels

    def build(self, input_shape):
        out_channels = input_shape[-1]//2
        # self.conv_stack = conv_stack(self.mid_channels,
        #                               out_channels, self.activation) # regular convolutions
        
        self.conv_stack = Unet(out_channels) # Unet conv stack
        
    def call(self, x, reverse=False):
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        out_ch = x1.get_shape().as_list()[-1]

        z1 = x1

        shift = self.conv_stack(z1)
        if not reverse:
            z2 = x2 + shift
        else:
            z2 = x2 - shift

        return tf.concat([z1, z2], axis=-1), 0.0


class affine_coupling(layers.Layer):
    def __init__(self, mid_channels=128, activation="relu"):
        super(affine_coupling, self).__init__()
        self.activation = activation
        self.mid_channels = mid_channels

    def build(self, input_shape):
        out_channels = input_shape[-1]
        # self.conv_stack = conv_stack(self.mid_channels,
        #                               out_channels, self.activation) # regular convolutions
        self.conv_stack = Unet(out_channels) # Unet conv stack

    def call(self, x, reverse=False):
        out_ch = x.get_shape().as_list()[-1]
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        
        alpha = 0.01
        z1 = x1
        log_scale_and_shift = self.conv_stack(z1)
        shift = log_scale_and_shift[:, :, :, 0::2]
        scale =  tf.nn.sigmoid(log_scale_and_shift[:, :, :, 1::2] + 2.0)
        scale = alpha + (1-alpha)*scale # to be more stable in reverse
        
        if not reverse:
            z2 = (x2 + shift) * scale
        else:
            z2 = x2/scale - shift

        objective = tf.reduce_sum(tf.math.log(scale), axis=[1, 2, 3])
        if reverse:
            objective *= -1

        return tf.concat([z1, z2], axis=3), objective


class revnet_step(layers.Layer):
    """One layer of this is:
    [1] Actnorm -- data normalization
    [2] 1x1 conv -- permutation
    [3] coupling layer -- Jacobian
    """
    def __init__(self, **kwargs):
        super(revnet_step, self).__init__()
        self.coupling_type = kwargs.get('coupling_type', 'affine')
        self.layer_type = kwargs.get('layer_type', 'bijective')
        self.mid_ch = kwargs.get('mid_channels', 128)
        self.latent_model = kwargs.get('latent_model', False)
        self.activation = kwargs.get('activation', 'linear')
        self.norm = actnorm()
        
        gamma = 0 if self.latent_model else 1e-3
        self.conv = invertible_1x1_conv(
            op_type=self.layer_type , activation = self.activation , gamma = gamma)

        if self.coupling_type == 'affine':
            self.coupling = affine_coupling(
                mid_channels=self.mid_ch)

        else:
            self.coupling = additive_coupling(
                mid_channels=self.mid_ch)

    def call(self, x, reverse=False , training = True):
        obj = 0
        ops = [self.norm, self.conv, self.coupling]
        if reverse:
            ops = ops[::-1]

        for op in ops:  
            x, curr_obj = op(x, reverse=reverse , training = training)
            obj += curr_obj

        return x, obj


class revnet(layers.Layer):
    """Composition of revnet steps"""
    def __init__(self, **kwargs):
        super(revnet, self).__init__()
        self.coupling_type = kwargs.get('coupling_type', 'affine')
        self.depth = kwargs.get('depth', 3)
        self.latent_model = kwargs.get('latent_model', False)
        self.steps = [revnet_step(coupling_type=self.coupling_type ,
                                  layer_type = 'bijective',
                                  latent_model = self.latent_model ,
                                  activation = 'linear')
                      for _ in range(self.depth)]
        
    def call(self, x, reverse=False , training = True):
        objective = 0.0
        if reverse:
            steps = self.steps[::-1]
        else:
            steps = self.steps

        for i in range(self.depth):
            step = steps[i]
            x, curr_obj = step(x,
                               reverse=reverse , training = training)
            objective += curr_obj

        return x, objective

########################################################
# LINEAR REVNETS
########################################################
class permute(layers.Layer):
    """Builds a permutation op
    """

    def __init__(self, **kwargs):
        super(permute, self).__init__()

        
    def build(self, input_shape):
        _, dim = input_shape
        indices = tf.range(start=0, limit=dim, dtype=tf.int32)

        # indices used for shuffling
        self.shuffled_indices = tf.random.shuffle(indices)

        # gets the indices that restores the shuffle
        self.inverse_indices = tf.argsort(self.shuffled_indices)


    def call(self, x, reverse=False):
        if not reverse:
            y = tf.gather(x, self.shuffled_indices, axis=-1)
        else:
            y = tf.gather(x, self.inverse_indices, axis=-1)

        objective = 0

        return y, objective

class linear_dim_change(layers.Layer):
    def __init__(self, out_dim):
        super(linear_dim_change, self).__init__()
        self.out_dim = out_dim

    def build(self, input_shape):
        in_dim = input_shape[-1]
        self.w = tf.Variable(tf.random.normal((in_dim,self.out_dim))/np.sqrt(in_dim))

    def call(self, x, reverse=False):
        s = tf.linalg.svd(self.w, 
            full_matrices=False, compute_uv=False)
        obj = tf.reduce_sum(tf.math.log(s))

        if reverse:
            return tf.matmul(x, tf.linalg.pinv(self.w)), -obj

        return tf.matmul(x, self.w), obj



class linear_stack(layers.Layer):
    def __init__(self, mid_channels,
                 output_channels, activation="relu"):
        super(linear_stack, self).__init__()

        self.layer1 = layers.Dense(
            mid_channels, activation=activation, 
            use_bias=False)
        self.layer2 = layers.Dense(
            mid_channels, activation=activation, 
            use_bias=False)
        self.layer3 = layers.Dense(
            output_channels, activation=None,
            use_bias=False)


    def check_weights(self):
        """Used only for debugging"""
        def check_tensor(w):
            if tf.reduce_any(tf.math.is_nan(w)):
                print(w.name)
                print(w)
                import sys
                sys.exit()
        
        check_tensor(self.layer1.kernel)
        check_tensor(self.layer2.kernel)
        check_tensor(self.layer3.kernel)

    def call(self, x , training = True):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

class additive_coupling_linear(layers.Layer):
    def __init__(self, mid_units=128, activation="relu"):
        super(additive_coupling_linear, self).__init__()
        self.activation = activation
        self.mid_units = mid_units

    def build(self, input_shape):
        out_channels = input_shape[-1]
        self.stack = linear_stack(self.mid_units, 
            out_channels - out_channels//2)

    def call(self, x, reverse=False):

        out_ch = x.get_shape().as_list()[-1]
        
        splits = [out_ch//2, out_ch - out_ch//2]
        x1, x2 = tf.split(x, splits, axis=-1)
        
        z1 = x1
        shift = self.stack(z1)
        
        if not reverse:
            z2 = x2 + shift
        else:
            z2 = x2 - shift

        return tf.concat([z1, z2], axis=-1), tf.zeros(tf.shape(x)[0])


class affine_coupling_linear(layers.Layer):
    def __init__(self, mid_units=128, activation="relu"):
        super(affine_coupling_linear, self).__init__()
        self.activation = activation
        self.mid_units = mid_units

    def build(self, input_shape):
        out_channels = input_shape[-1]
        if out_channels%2!=0:
            out_channels += 1
        self.stack = linear_stack(self.mid_units, out_channels)

    def call(self, x, reverse=False):
        out_ch = x.get_shape().as_list()[-1]
        splits = [out_ch//2, out_ch - out_ch//2]
        x1, x2 = tf.split(x, splits, axis=-1)
        
        z1 = x1
        log_scale_and_shift = self.stack(z1)

        shift = log_scale_and_shift[:, 0::2]
        scale =  0.1 + 0.9*tf.nn.sigmoid(log_scale_and_shift[:,1::2] + 2.0)

        
        if not reverse:
            z2 = (x2 + shift) * scale
        else:
            z2 = x2/scale - shift

        objective = tf.reduce_sum(tf.math.log(scale), axis=[1])
        if reverse:
            objective *= -1

        return tf.concat([z1, z2], axis=-1), objective

class revnet_step_linear(layers.Layer):
    """One layer of this is:
    [1] Actnorm -- data normalization
    [2] 1x1 conv -- permutation
    [3] coupling layer -- Jacobian
    """
    def __init__(self, **kwargs):
        super(revnet_step_linear, self).__init__()
        self.mid_ch = kwargs.get('mid_units', 128)
        self.layer_type = kwargs.get('layer_type', 'additive')
        
        
        self.norm = actnorm()
        self.conv = permute()
        if self.layer_type=='affine':
            self.coupling = affine_coupling_linear(mid_units=self.mid_ch)
        else:
            self.coupling = additive_coupling_linear(mid_units=self.mid_ch)


    def call(self, x, reverse=False , training = True):
        obj = 0
        
        ops = [self.norm, self.conv, self.coupling]
            
        if reverse:
            ops = ops[::-1]

        for op in ops:  
            x, curr_obj = op(x, reverse=reverse)
            obj += curr_obj

        return x, obj

class revnet_linear(layers.Layer):
    """Composition of revnet steps"""
    def __init__(self, **kwargs):
        super(revnet_linear, self).__init__()
        self.layer_type = kwargs.get('layer_type', 'affine')
        self.depth = kwargs.get('depth', 3)
        self.mid_units = kwargs.get('mid_units', 64)
        self.steps = [revnet_step_linear(mid_units=self.mid_units,
            layer_type=self.layer_type) for _ in range(self.depth)]


    def call(self, x, reverse=False):
        objective = 0.0
        if reverse:
            steps = self.steps[::-1]
        else:
            steps = self.steps

        for i in range(self.depth):
            step = steps[i]
            x, curr_obj = step(x, reverse=reverse)
            objective += curr_obj

        return x, objective


def test_(op, x):
    y, _ = op(x)
    xhat, _ = op(y, reverse=True)

    err = tf.linalg.norm(x - xhat).numpy()

    if err < 1e-4:
        print('PASS. Error: %e' % err)
    else:
        print('FAIL. Error: %e' % err)


def unit_test_revnet():
    from time import time

    print('Unit testing revnets...')
    x = tf.random.normal((1, 16, 16, 32))

    ## dummy test so that all cuda kernels 
    ## are launched and they do not affect 
    ## the timing
    print('####START DUMMY TEST####\n')
    g = revnet_step(coupling_type='additive',
        layer_type='injective', gamma=1e-4)
    test_(g, x)
    print('####END DUMMY TEST####\n')

    print('Testing additive coupling...\n')
    g = revnet_step(coupling_type='additive',
        layer_type='injective', gamma=1e-4)
    t = time()
    test_(g, x)
    print('Testing took %fs\n'%(time()-t))

    print('Testing affine coupling...\n')
    g = revnet_step(coupling_type='affine', 
        layer_type='injective', gamma=1e-4)
    t = time()
    test_(g, x)
    print('Testing took %fs\n'%(time()-t))

    return None
    
    

if __name__ == '__main__':
    unit_test_revnet()