import numpy as np
import os
import time
import sys
sys.path.insert(0, './utils')
sys.path.insert(0, './model')
from Graph2Property import Graph2Property
from utils import *
import tensorflow as tf
np.set_printoptions(precision=3)

def loadInputs(FLAGS, idx, modelName, unitLen):
    if(FLAGS.validation_database == 'QM9'):
        adj1 = np.load('./Data/'+FLAGS.validation_database+'/adj_explicit/'+str(idx)+'.npy')
        features = np.load('./Data/'+FLAGS.validation_database+'/features_explicit/'+str(idx)+'.npy')
    else:
        adj1 = np.load('./Data/'+FLAGS.validation_database+'/adj/'+str(idx)+'.npy')
        features = np.load('./Data/'+FLAGS.validation_database+'/features/'+str(idx)+'.npy')
    retInput = (adj1, features)
    retOutput = (np.load('./Data/'+FLAGS.validation_database+'/'+FLAGS.output+'.npy')[idx*unitLen:(idx+1)*unitLen])

    return retInput, retOutput

def training(model, FLAGS, modelName):
    num_epochs = FLAGS.epoch_size
    batch_size = FLAGS.batch_size
    decay_rate = FLAGS.decay_rate
    save_every = FLAGS.save_every
    learning_rate = FLAGS.learning_rate
    num_DB = FLAGS.num_DB
    unitLen = FLAGS.unitLen
    total_iter = 0
    total_st = time.time()
    print ("Start Training XD")
    for epoch in range(num_epochs):
        # Learning rate scheduling 
        model.assign_lr(learning_rate * (decay_rate ** epoch))

        for i in range(0,num_DB):
            _graph, _property = loadInputs(FLAGS, i, modelName, unitLen)
            num_batches = int(_graph[0].shape[0]/batch_size)

            st = time.time()
            for _iter in range(num_batches):
                total_iter += 1
                A_batch = _graph[0][_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]
                X_batch = _graph[1][_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]
                P_batch = _property[_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]
                if total_iter % 5 != 0:
                    # Training
                    cost = model.train(A_batch, X_batch, P_batch)
                    print ("train_iter : ", total_iter, ", epoch : ", epoch, ", cost :  ", cost)

                elif total_iter % 5 == 0:
                    # Test accuracy
                    Y, cost = model.test(A_batch, X_batch, P_batch)
                    print ("test_iter : ", total_iter, ", epoch : ", epoch, ", cost :  ", cost)
                    if( total_iter % 100 == 0 ):
                        print (Y.flatten())
                        print (P_batch)
                        print (Y.flatten() - P_batch)
                        mse = (np.mean(np.power((Y.flatten() - P_batch),2)))
                        mae = (np.mean(np.abs(Y.flatten() - P_batch)))
                        print ("MSE : ", mse, "\t MAE : ", mae)
        
                if total_iter % save_every == 0:
                    # Save network! 
                    ckpt_path = 'save/'+modelName+'.ckpt'
                    model.save(ckpt_path, total_iter)

            et = time.time()
            print ("time : ", et-st)
            st = time.time()

    total_et = time.time()
    print ("Finish training! Total required time for training : ", (total_et-total_st))
    return

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set FLAGS for environment setting
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'GAT', 'Options : GAT, GCN') 
flags.DEFINE_string('output', 'logP', '')
flags.DEFINE_string('loss_type', 'MSE', 'Options : MSE')  ### Using MSE or Hinge for predictor 
flags.DEFINE_string('database', 'ZINC', 'Options : ZINC, QM9, ZINC')  
flags.DEFINE_string('optimizer', 'Adam', 'Options : Adam, SGD, RMSProp') 
flags.DEFINE_integer('latent_dim', 512, 'Dimension of a latent vector for autoencoder')
flags.DEFINE_integer('epoch_size', 100, 'Epoch size')
flags.DEFINE_integer('batch_size', 100, 'Batch size')
flags.DEFINE_integer('save_every', 1000, 'Save every')
flags.DEFINE_float('learning_rate', 0.001, 'Batch size')
flags.DEFINE_float('decay_rate', 0.95, 'Batch size')
flags.DEFINE_integer('num_DB', 45, '')
flags.DEFINE_integer('unitLen', 10000, '')

modelName = FLAGS.model + '_' + FLAGS.output + '_' + str(FLAGS.latent_dim) + '_' + str(FLAGS.batch_size) + '_' + FLAGS.database

print ("Summary of this training & testing")
print ("Model name is", modelName)
print ("A Latent vector dimension is", str(FLAGS.latent_dim))
print ("A learning rate is", str(FLAGS.learning_rate), "with a decay rate", str(FLAGS.decay_rate))
print ("Using", FLAGS.loss_type, "for loss function in an optimization")

model = Graph2Property(FLAGS)
training(model, FLAGS, modelName)
