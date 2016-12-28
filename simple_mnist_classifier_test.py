import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mnist import MNIST
import math
import time
from display_utils import DynamicConsoleTable

print 'Loading images..'
mndata = MNIST('./mnist')
training_images, training_labels = mndata.load_training()
validation_images, validation_labels = mndata.load_testing()
print 'Training images: {}'.format(len(training_images))
print 'Validation images: {}'.format(len(validation_images))

print 'Reshaping images..'
for i in range(len(training_images)):
    training_images[i] = np.expand_dims(np.reshape(training_images[i], (28, 28)), axis=2)
for i in range(len(validation_images)):
    validation_images[i] = np.expand_dims(np.reshape(validation_images[i], (28, 28)), axis=2)
print 'Done'

#fig = plt.figure()
#for i in range(len(training_images)):
#    print "{} / {}".format(i + 1, len(training_images))
#    fig.clear()
#    plt.imshow(training_images[i], cmap='gray')
#    plt.pause(0.001)

#plt.imshow(training_images[0], cmap='gray')
#plt.show()

###############################################################################

def model(learning_rate=0.01):
    # Parameters
    weights = {
        'conv1': tf.Variable(tf.random_normal([5, 5, 1, 20], stddev=np.sqrt(2./(5*5*1)))),
        'out': tf.Variable(tf.random_normal([28*28*20, 10], stddev=np.sqrt(2./(28*28*20)))),
    }
    biases = {
        'conv1': tf.Variable(tf.zeros(20)),
        'out': tf.Variable(tf.zeros(10)),
    }

    # Placeholders for training data
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.int64, [None])

    # Input -> Conv + ReLU
    conv1 = tf.nn.conv2d(x, weights['conv1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['conv1']))
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    
    # FC -> Output FC
    out = tf.reshape(lrn1, [-1, weights['out'].get_shape().as_list()[0]])
    out = tf.add(tf.matmul(out, weights['out']), biases['out'])

    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(out, y))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(out, y, 1), tf.float32))
    
    return train_op, x, y, out, loss, accuracy, weights, biases

###############################################################################
### Settings

# Training settings
# Note: Training terminates when the sustained loss is below loss_threshold, or when training has reached max_epochs
max_epochs = 1000
batch_size = 100
validation_set_size = 1000
learning_rate = 0.05
loss_threshold = 0 #1e-12
decay_rate = 0.30 # Exponential decay used to calculate sustained loss
use_GPU = True # Use CUDA acceleration

# Display settings
show_progress = False
display_step = 50 # in batches
delay = 0.001
interpolation = None # None to use default (eg. "nearest", "bilinear")
cmap = None # None to use default (eg. "gray", "inferno")
progress_bar_size = 20

###############################################################################
# Display setup
fig = None
def display(weights_val):
    fig.clear()
    plot_height = int(weights_val['conv1'].shape[3] ** 0.5)
    plot_width = math.ceil(float(weights_val['conv1'].shape[3]) / plot_height)
    for j in range(weights_val['conv1'].shape[3]):
        ax = fig.add_subplot(plot_height, plot_width, j + 1)
        ax.imshow(weights_val['conv1'][:,:,0,j], interpolation=interpolation, cmap=cmap)
    plt.pause(delay)
###############################################################################

# Build model and get variable handles
train_op, x, y, out, loss, accuracy, weights, biases = model(learning_rate)

# Initialize environment
initialize = tf.global_variables_initializer()

# Session config
config = tf.ConfigProto(device_count = {'GPU': 1 if use_GPU == True else 0})

# Run model
with tf.Session(config=config) as session:
    session.run(initialize)
        
    done = False
    epoch = 0
    iteration = 0
    sustained_loss = 0.0
    loss_values = []
    
    assert validation_set_size <= len(validation_images), 'validation_set_size must be smaller than len(validation_images)'
    assert float(len(training_images)) / batch_size % 1 == 0, 'batch_size must evenly divide len(training_images)'
    assert float(validation_set_size) / batch_size % 1 == 0, 'batch_size must evenly divide validation_set_size'
    num_training_batches = len(training_images) / batch_size
    num_validation_batches = validation_set_size / batch_size
    
    training_image_batches = []
    training_label_batches = []
    validation_image_batches = []
    validation_label_batches = []
    for i in range(num_training_batches):
        training_image_batches.append(training_images[i*batch_size:(i+1)*batch_size])
        training_label_batches.append(training_labels[i*batch_size:(i+1)*batch_size])
    for i in range(num_validation_batches):
        validation_image_batches.append(validation_images[i*batch_size:(i+1)*batch_size])
        validation_label_batches.append(validation_labels[i*batch_size:(i+1)*batch_size])
    
    max_validation_accuracy = 0.0
    max_accuracy_weights = None
    max_accuracy_biases = None
    
    if show_progress:
        fig = plt.figure()
        weights_val = session.run(weights)
        display(weights_val)
        
    print
    
    layout = [
        dict(name='Ep.', width=3, align='center'),
        dict(name='Batch', width=2*len(str(num_training_batches))+1, suffix='/'+str(num_training_batches)),
        dict(name='Loss', width=8),
        dict(name='Val Acc', width=6, suffix='%'),
        dict(name='Max Acc', width=6, suffix='%'),
        dict(name='Time', width=progress_bar_size+2, align='center'),
    ]
    table = DynamicConsoleTable(layout)
    table.print_header()
    
    while not done:
        epoch += 1

        # Trains on the data, in batches
        for i in range(num_training_batches):
            iteration += 1
                        
            images_batch = training_image_batches[i]
            labels_batch = training_label_batches[i]
            _, loss_val = session.run([train_op, loss], feed_dict={x: images_batch, y: labels_batch})
            sustained_loss = decay_rate * sustained_loss + (1.0 - decay_rate) * loss_val
            loss_values.append(loss_val)
            
            images_batch = validation_image_batches[iteration % num_validation_batches]
            labels_batch = validation_label_batches[iteration % num_validation_batches]
            
            validation_accuracy = 0.0
            for j in range(num_validation_batches):
                images_batch = validation_image_batches[j]
                labels_batch = validation_label_batches[j]
                accuracy_val = session.run(accuracy, feed_dict={x: images_batch, y: labels_batch})
                validation_accuracy += accuracy_val
            validation_accuracy /= num_validation_batches
            
            if validation_accuracy > max_validation_accuracy:
                weights_val, biases_val = session.run([weights, biases])
                max_validation_accuracy = validation_accuracy
                max_accuracy_weights = weights_val
                max_accuracy_biases = biases_val
            
            progress = int(math.ceil(progress_bar_size * float((iteration - 1) % num_training_batches) / (num_training_batches - 1)))
            progress_string = '[' + '#' * progress + ' ' * (progress_bar_size - progress) + ']'
            if iteration % num_training_batches == 0:
                progress_string = time.strftime("%I:%M:%S %p", time.localtime())
            table.update(epoch,
                         (iteration - 1) % num_training_batches + 1,
                         sustained_loss,
                         validation_accuracy * 100,
                         max_validation_accuracy * 100,
                         progress_string)
            
            # Termination condition
            if sustained_loss < loss_threshold:
                done = True
                break

            # Show/update display
            if iteration % display_step == 0 and show_progress:
                display(weights_val)
        
        table.finalize()
            
        # Termination condition
        if epoch >= max_epochs or sustained_loss < loss_threshold:
            done = True
            if show_progress:
                display(weights_val)
            
# Display results
plt.show()

#plt.figure('Loss')
#plt.plot(loss_values)
#plt.show()

