import tensorflow as tf
from tensorflow.keras import layers


class Conv_block(layers.Layer):
    def __init__(self , num_filters):
        super(Conv_block, self).__init__()

        self.conv1 = layers.Conv2D(num_filters, (3, 3), padding="same", use_bias=False)
        self.conv2 = layers.Conv2D(num_filters, (3, 3), padding="same", use_bias=False)
        self.act1 = layers.Activation("relu")
        self.act2 = layers.Activation("relu")

    def call(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        return x
    



class Unet(layers.Layer):
    def __init__(self , output_channels):
        super(Unet, self).__init__()
        
        self.num_filters = [32, 64]
        
        self.conv_blocks1 = [Conv_block(num_filters = f) 
                            for f in self.num_filters]
        
        self.conv_blocks2 = [Conv_block(num_filters = f) 
                            for f in self.num_filters[::-1]]
            
        self.conv_block_bridge = Conv_block(self.num_filters[-1])
        self.maxpool = layers.MaxPool2D((2, 2))
        self.upsample = layers.UpSampling2D((2, 2))
        self.concat = layers.Concatenate()
        self.conv = layers.Conv2D(output_channels, (1, 1), padding="same" , use_bias=False)
        self.act = layers.Activation("sigmoid")
        

    def call(self, x , training = True):
        
        skip_x = []
        
        ## Encoder
        for i in range(len(self.num_filters)):
            x = self.conv_blocks1[i](x)
            skip_x.append(x)
            x = self.maxpool(x)
        
        ## Bridge
        x = self.conv_block_bridge(x)

        skip_x = skip_x[::-1]
        ## Decoder
        for i in range(len(self.num_filters)):
            x = self.upsample(x)
            xs = skip_x[i]
            x = self.concat([x, xs])
            x = self.conv_blocks2[i](x)

    
        ## Output
        x = self.conv(x)
        x = self.act(x)
        return x

def gen_unet_grad_graph(net):

    @tf.function
    def vjp(x, v):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = net(x)
        
        return tape.gradient(y, x, output_gradients=v)

    @tf.function
    def jvp(x, v_x, v_y):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = net(x)

            with tf.GradientTape() as tape2:
                tape2.watch(v_y)
                u = tape.gradient(y, x, output_gradients=v_y)
                final_out = tape2.gradient(u, v_y, output_gradients=v_x)

        return final_out

    return vjp, jvp




def unit_test_grad_unet(x, v, net):

    

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = net(x)
        print('Shape of y is '  )
        print(y.shape)

    grad1 = tape.gradient(y, x, output_gradients=v)

    return grad1

if __name__ == '__main__':
    x = tf.Variable(tf.random.uniform((5,64,64,3)),
        trainable=True)
    net = Unet(30) # build unet with 3 output channels

    print('y shape')
    y = net(x)
    print(y.shape)

    v_x = tf.random.uniform(x.shape)
    v_y = tf.random.uniform(y.shape)
    
    vjp_fn, jvp_fn = gen_unet_grad_graph(net)

    out = unit_test_grad_unet(x,v_y,net)
    print('vjp shape')
    print(out.shape)

    out_vjp_fn = vjp_fn(x, v_y)
    print('vjp fn shape')
    print(out_vjp_fn.shape)

    print('jvp fn shape')
    out_jvp_fn = jvp_fn(x, v_x, v_y)
    print(out_jvp_fn.shape)

    print(tf.linalg.norm(out- out_vjp_fn))