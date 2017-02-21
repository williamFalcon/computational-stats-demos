import tensorflow as tf
import automatic_diff as ad


def test_ad():
    x = ad.tensor(ad.int32, name='x')
    y = ad.tensor(ad.int32, name='y')

    z = (x*x)
    grads = ad.gradients(z, [x])
    basic_f = grads
    print(basic_f(x=3, y=5))

    basic_f = ad.compile(z)
    print(basic_f(x=3, y=5))


# -------------------------
# DEMOS
# -------------------------
def jacobian_demo():
    x = tf.placeholder(tf.float64, name='x')
    y = tf.placeholder(tf.float64, name='y')

    # demo function
    f = 6*(x**5) - 5*(y**4)

    J = jacobian(f, [x, y])
    sess = tf.Session()
    print(sess.run(J, feed_dict={x: 3, y:5}))


def hessian_demo():
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')

    f = 6*(x**5) - 5*(y**4)*x**2

    H = hessian(f, [x,y])
    sess = tf.Session()
    print(sess.run(H, feed_dict={x: 3, y: 4}))

def newton_method_univar():
    x = tf.placeholder(tf.float64, name='x')
    f = 6*(x**5) - 5*(x**4) - 4*(x**3) + 3*(x**2)

    def newton(fn, x_0, vars):
        dx = tf.gradients(fn, vars)[0]
        x_1 = x_0 - (fn/dx)
        return x_1

    newton_opt = newton(f, x, [x])
    # create the session
    sess = tf.Session()

    # termination error
    e = 0.0001

    # initial delta
    delta = 1.0

    # initial x_0 guess
    x_0 = -0.75

    while delta > e:
        x_1 = sess.run(newton_opt, feed_dict={x: x_0})
        delta = abs(x_1 - x_0)
        x_0 = x_1
        print('error :', delta, 'root ', x_0)
    print('done')

# ---------------------------
# UTILS
# ---------------------------
def jacobian(fn, vars):
    grads = tf.gradients(fn, vars)
    return grads


def hessian(fn, vars):
        cons = lambda x: tf.constant(x, dtype=tf.float32)
        mat = []
        for v1 in vars:
            temp = []
            for v2 in vars:
                # compute grad
                print('calculating: d{} d{}'.format(v2.name, v1.name))
                df = tf.gradients(fn, v2)[0]
                dg = tf.gradients(df, v1)[0]
                temp.append(dg)

            # replace non grads with 0s
            temp = [cons(0) if t is None else t for t in temp]

            temp = tf.stack(temp)
            mat.append(temp)
        mat = tf.stack(mat)
        return mat



test_ad()