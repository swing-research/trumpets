import tensorflow as tf
import numpy as np

def wrapper_logdet(x, f):

    # @tf.function
    def power_iteration(f, n):
        v = tf.random.normal(x.shape, dtype=tf.float32)
        v /= tf.linalg.norm(v, axis=-1, keepdims=True)

        for _ in range(n):
            with tf.autodiff.ForwardAccumulator(primals=x, tangents=v) as acc:
                y = f(x)

            u1 = acc.jvp(y) # Jv

            with tf.GradientTape() as tape:
                tape.watch(x)
                y = f(x)

            u1 = tape.gradient(y, x, output_gradients=u1) # J^T v

            # current estimate of eigval
            eigval = tf.reduce_sum(v*u1, axis=-1)

            # calculate the norm
            u1_norm = tf.linalg.norm(u1, axis=-1, keepdims=True)

            # re normalize the vector
            v = u1 / u1_norm


        return tf.reduce_max(eigval)


    # @tf.function
    def logdet_1(x, f, n, beta):
        logdet_val = tf.zeros(tf.shape(x)[0], dtype=tf.float32)

        v = tf.random.normal(x.shape, dtype=tf.float32)
        v1 = tf.identity(v)

        for k in range(1,n+1):
            with tf.autodiff.ForwardAccumulator(primals=x, tangents=v1) as acc:
                y = f(x)

            u1 = acc.jvp(y)

            with tf.GradientTape(persistent=False) as tape:
                tape.watch(x)
                y = f(x)

            u2 = tape.gradient(y, x, output_gradients=u1)
            v1 = v1 - beta*u2

            logdet_val -= tf.reduce_sum(v1*v, axis=-1)/tf.cast(k, tf.float32)

        return logdet_val

    def logdet(x, f, n, nv=10, beta=1):
        logdet_val = 0

        y = f(x)
        d = tf.math.minimum(y.shape[1], x.shape[1])

        for _ in range(nv):
            logdet_val += logdet_1(x, f, n, beta)
        logdet_val /= nv

        return logdet_val - tf.cast(d, tf.float32)*np.log(beta)

    def get_logdet():
        val = power_iteration(f, 10)
        beta = 0.95/val.numpy()
        print('beta is')
        print(beta)

        nevals = 50
        ld = 0
        for _ in range(nevals):
            n = 10
            ld += logdet(x, f, np.int32(n), beta=np.float32(beta), nv=10)

        ld /= nevals
        
        return ld


    return get_logdet()


def unit_test():
    DIM = 10
    x = tf.Variable(tf.ones((45, DIM), dtype=tf.float32), trainable=True)*2


    def f(x, reverse=False):
        return tf.concat((x[:,:DIM//2]**2, x**2), axis=1)/2.0

    ld = wrapper_logdet(x, f)

    print('true_value is %f'%(np.log(2)*25))

    print(ld)
    print('mean and std deviation are:')
    print(np.mean(ld.numpy()))
    print(np.std(ld.numpy()))

    return ld


if __name__ == '__main__':
    ld = unit_test()

