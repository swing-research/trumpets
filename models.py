import tensorflow as tf
import numpy as np
from utils import *
import glow_ops as g

class posterior(tf.keras.Model):
    def __init__(self, **kwargs):
        super(posterior, self).__init__()
        """Model architecture
        posterior :=> 

        """
        self.in_dim = kwargs.get('input_dim', 192)
        self.depth = kwargs.get('depth', 3)
        self.mid_units = kwargs.get('mid_units', 128)
        self.layer_type = kwargs.get('layer_type', 'additive')
        
        
        self.revnets = [g.revnet_linear(depth= self.depth,
            mid_units=self.mid_units,
            layer_type=self.layer_type) for _ in range(6)]
        
        

    def call(self, x, reverse=False):
        
        ops = [
            self.revnets[0],
            self.revnets[1],
            self.revnets[2],
            self.revnets[3],
            self.revnets[4],
            self.revnets[5]]

        # if self.inside:
        #     ops = [self.dim_change_op] + ops
        # else:
        #     ops = ops + [self.dim_change_op]

        if reverse:
            ops = ops[::-1]

        objective = 0.0

        for op in ops:
            x, curr_obj = op(x, reverse=reverse)
            # print(op.name)
            # if tf.reduce_any(tf.math.is_nan(x)):
            #     print(op.name)

            objective += curr_obj

        return x, objective


class generator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(generator, self).__init__()
        """Injective Model architecture
        . upsqueeze
        --> revnet
        |-> inj_rev_step

        + 4x4x12 --> 4x4x12 |-> 4x4x24 . 8x8x6
         --> 8x8x6 |-> 8x8x12 |-> 8x8x24 --> 8x8x24
        |-> 8x8x48 --> 8x8x48 . 16x16x12 |-> 16x16x24
        --> 16x16x24 . 32x32x6 |-> 32x32x12 --> 32x32x12
        . 64x64x3
        
        summary for celeba: 
        6 bijective revnets
        6 injective revnet_steps
        4 upsqueeze
        """
        self.problem = kwargs.get('dataset', 'cifar10')
        self.depth = kwargs.get('revnet_depth', 3) # revnet depth
        self.activation = kwargs.get('activation', 'linear') # activation ofinvertible 1x1 convolutional layer
        
        self.squeeze = g.upsqueeze(factor=2)
        self.revnets = [g.revnet(coupling_type='affine', depth= self.depth , latent_model = False) 
        for _ in range(4+4)] # Bijective revnets
        
        self.inj_rev_steps = [g.revnet_step(layer_type='injective', 
            coupling_type='affine' , latent_model = False, activation = self.activation) for _ in range(4+4)]
        

    def call(self, x, reverse=False , training = True):
        
        
        c = 1 if self.problem == 'mnist' or self.problem == 'chest' else 3
        f = 2 if self.problem == 'chest' else 1
        
        if reverse:
            x = tf.reshape(x, [-1,4,4,4 *f * f* c])
      
            
        ops = [
        self.squeeze,
        self.revnets[0],
        self.inj_rev_steps[0],
        self.squeeze,
        self.revnets[1],
        self.inj_rev_steps[1],
        self.squeeze,
        self.revnets[2],
        self.inj_rev_steps[2],
        self.revnets[3],
        self.inj_rev_steps[3],
        ]
        
        if self.problem == 'chest':
            
            ops = [self.squeeze] + ops
        
        if self.problem =='celeba' or self.problem =='imagenet' or self.problem =='rheo' or self.problem =='church' or self.problem == 'chest':
            
            ops += [self.inj_rev_steps[4],
            self.revnets[4],
            self.squeeze,
            self.inj_rev_steps[5],
            self.revnets[5]
            ]
   

        if reverse:
            ops = ops[::-1]

        objective = 0.0

        for op in ops:
           
            x, curr_obj = op(x, reverse= reverse , training = training)
            objective += curr_obj

        if not reverse:
            x = tf.reshape(x, (-1, 4*f *4*f *4*c))

        return x, objective
 


class latent_generator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(latent_generator, self).__init__()
        """ Bijective Model architecture
        --> revnet
        
        + 4x4x12 --> 4x4x12 --> 4x4x12 --> 4x4x12 -->
        4x4x12 --> 4x4x12 --> 4x4x12 --> 4x4x12 -->
        4x4x12 --> 4x4x12 --> 4x4x12 --> 4x4x12 -->
        4x4x12 --> 4x4x12 --> 4x4x12 --> 4x4x12 -->
        4x4x12
        
        summary for celeba: 
        8 bijective revnets
        """
        self.problem = kwargs.get('dataset', 'cifar10')
        self.depth = kwargs.get('revnet_depth', 3)
        self.pz = kwargs.get('pz', None)
        self.revnets = [g.revnet(coupling_type='affine', depth = self.depth , latent_model = True) 
        for _ in range(8)]

        
    def call(self, x, reverse=False , training = True):
        
        c = 1 if self.problem == 'mnist' or self.problem == 'chest' else 3
        f = 2 if self.problem == 'chest' else 1
    
        x = tf.reshape(x, [-1,4,4,4 *f * f* c])
        
        ops = [
        self.revnets[0],
        self.revnets[1],
        self.revnets[2],
        self.revnets[3],
        self.revnets[4],
        self.revnets[5],
        self.revnets[6],
        self.revnets[7]
        ]

        if reverse:
            ops = ops[::-1]

        objective = 0.0

        for op in ops:
            
            x, curr_obj = op(x, reverse=reverse , training = training)

            objective += curr_obj

        x = tf.reshape(x, (-1, 4*f *4*f *4*c))
        return x, objective
 
    def log_prob(self, sample):

        rev_sample, obj = self(sample, reverse=False)
        if self.pz is not None:
            p = -tf.reduce_mean(self.pz.prior.log_prob(rev_sample))
        else:
            print('pz was not passed into the instantiation of the object! ')
            raise NotImplementedError

        j = -tf.reduce_mean(obj)

        return p + j
