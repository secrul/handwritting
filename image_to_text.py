import tensorflow as tf
import os
import numpy as np


def get_files(filename):
    class_train = []
    label_train = []
    for train_class in os.listdir(filename):
        for pic in os.listdir(filename+train_class):
            class_train.append(filename+train_class+'/'+pic)
            label_train.append(train_class)
    temp = np.array([class_train,label_train])
    temp = temp.transpose()
    #shuffle the samples
    np.random.shuffle(temp)
    #after transpose, images is in dimension 0 and label in dimension 1
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
    #print(label_list)
    return image_list,label_list


def get_batches(image, label, resize_w, resize_h, batch_size, capacity):
    # convert the list of images and labels to tensor
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)
    queue = tf.train.slice_input_producer([image, label])
    label = queue[1]
    image_c = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_c, channels=3)
    # resize
    image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)
    # (x - mean) / adjusted_stddev
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    images_batch = tf.cast(image_batch, tf.float32)
    labels_batch = tf.reshape(label_batch, [batch_size])
    return images_batch, labels_batch

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev = 0.01))
#init weights
weights = {
    "w1":init_weights([3,3,3,16]),
    "w2":init_weights([3,3,16,128]),
    "w3":init_weights([3,3,128,256]),
    "w4":init_weights([4096,4096]),
    "wo":init_weights([4096,2])
    }

#init biases
biases = {
    "b1":init_weights([16]),
    "b2":init_weights([128]),
    "b3":init_weights([256]),
    "b4":init_weights([4096]),
    "bo":init_weights([2])
    }


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev = 0.01))
#init weights
weights = {
    "w1":init_weights([3,3,3,16]),
    "w2":init_weights([3,3,16,128]),
    "w3":init_weights([3,3,128,256]),
    "w4":init_weights([4096,4096]),
    "wo":init_weights([4096,2])
    }

#init biases
biases = {
    "b1":init_weights([16]),
    "b2":init_weights([128]),
    "b3":init_weights([256]),
    "b4":init_weights([4096]),
    "bo":init_weights([2])
    }

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev = 0.01))
#init weights
weights = {
    "w1":init_weights([3,3,3,16]),
    "w2":init_weights([3,3,16,128]),
    "w3":init_weights([3,3,128,256]),
    "w4":init_weights([4096,4096]),
    "wo":init_weights([4096,2])
    }

#init biases
biases = {
    "b1":init_weights([16]),
    "b2":init_weights([128]),
    "b3":init_weights([256]),
    "b4":init_weights([4096]),
    "bo":init_weights([2])
    }

def loss(logits,label_batches):
     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_batches)
     cost = tf.reduce_mean(cross_entropy)
     return cost


def get_accuracy(logits,labels):
     acc = tf.nn.in_top_k(logits,labels,1)
     acc = tf.cast(acc,tf.float32)
     acc = tf.reduce_mean(acc)
     return acc

def training(loss,lr):
     train_op = tf.train.RMSPropOptimizer(lr,0.9).minimize(loss)
     return train_op


def run_training():
    data_dir = 'C:/Users/wk/Desktop/bky/dataSet/'
    image, label = inputData.get_files(data_dir)
    image_batches, label_batches = inputData.get_batches(image, label, 32, 32, 16, 20)
    p = model.mmodel(image_batches)
    cost = model.loss(p, label_batches)
    train_op = model.training(cost, 0.001)
    acc = model.get_accuracy(p, label_batches)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(1000):
            print(step)
            if coord.should_stop():
                break
            _, train_acc, train_loss = sess.run([train_op, acc, cost])
            print("loss:{} accuracy:{}".format(train_loss, train_acc))
    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


