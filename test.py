import numpy as np
import os
import time
import sys
sys.path.insert(0, './utils')
sys.path.insert(0, './model')
from Graph2Property import Graph2Property
from utils import *
import tensorflow as tf
from rdkit.Chem import Draw
import smilesToGraph2
import smilesToGraph_explicit
np.set_printoptions(precision=3)

def loadInputs(FLAGS, idx, modelName, unitLen):
    if(FLAGS.validation_database == 'qm9'):
        adj1 = np.load('./Data/'+FLAGS.validation_database+'/adj_explicit/'+str(idx)+'.npy')
        features = np.load('./Data/'+FLAGS.validation_database+'/features_explicit/'+str(idx)+'.npy')
    else:
        adj1 = np.load('./Data/'+FLAGS.validation_database+'/adj/'+str(idx)+'.npy')
        features = np.load('./Data/'+FLAGS.validation_database+'/features/'+str(idx)+'.npy')
    retInput = (adj1, features)
    retOutput = (np.load('./Data/'+FLAGS.validation_database+'/'+FLAGS.output+'.npy')[idx*unitLen:(idx+1)*unitLen])

    return retInput, retOutput

def getLatent(model, FLAGS, modelName):
    batch_size = FLAGS.batch_size
    total_st = time.time()
    model.restore("./save/"+modelName+".ckpt-450000")

    S = np.load("./Data/"+FLAGS.validation_database+'/smiles.txt')
    num_valid = S.shape[0]

    S = list(S)
    A = list(A)
    X = list(X)

    S_zero = np.zeros(120)
    A_zero = np.zeros((50,50))
    X_zero = np.zeros((50,28))

    batch_size = FLAGS.batch_size
    emptyNum = len(S)%batch_size
    print ("Start Feature Extraction XD")
    for i in range(0,batch_size-emptyNum):
        A.append(A_zero)
        X.append(X_zero)
        S.append(S_zero)

    A_batch = np.asarray(A)
    X_batch = np.asarray(X)
    S_batch = np.asarray(S)
    num_batch = int(S_batch.shape[0]/batch_size)

    latent = []
    adjacency = []
    for i in range(num_batch):
        idx = np.arange(batch_size) + i*batch_size
        Z = model.get_latent_vector(A_batch[idx], X_batch[idx])
        _A = model.get_adjacency(A_batch[idx], X_batch[idx])
        for z in Z:
            latent.append(z)
        for _a in _A:
            adjacency.append(_a)
    latent = np.asarray(latent)
    adjacency = np.asarray(adjacency)
    np.save('./latent/'+modelName+'_'+FLAGS.validation_database+'.npy', latent)
    np.save('./adjacency/'+modelName+'_'+FLAGS.validation_database+'.npy', adjacency)

    total_et = time.time()
    print ("Finish feature extraction! Total required time for extraction : ", (total_et-total_st))

def predict(model, FLAGS, modelName):
    batch_size = FLAGS.batch_size
    total_st = time.time()
    model.restore("./save/"+modelName+".ckpt-"+str(FLAGS.num_model))

    smiles_f = open(FLAGS.validation_database+'.txt', 'r')
    smiles_list = smiles_f.readlines()
    num_valid = len(smiles_list)
    #A, X = smilesToGraph_explicit.convertToGraph(smiles_list, 1)
    A, X = smilesToGraph2.convertToGraph(smiles_list, 1)
    A = list(A)
    X = list(X)

    A_zero = np.zeros((50,50))
    X_zero = np.zeros((50,58))

    batch_size = FLAGS.batch_size
    emptyNum = num_valid%batch_size
    print ("Start Feature Extraction XD")
    for i in range(0,batch_size-emptyNum):
        A.append(A_zero)
        X.append(X_zero)

    A_batch = np.asarray(A)
    X_batch = np.asarray(X)
    num_batch = int(A_batch.shape[0]/batch_size)

    prediction = []
    for i in range(num_batch):
        idx = np.arange(batch_size) + i*batch_size
        _P = model.predict(A_batch[idx], X_batch[idx])
        print (_P[0][:num_valid])
        prediction.append(_P)
    np.save('./prediction/'+modelName+'_'+FLAGS.validation_database+'.npy', prediction)

    total_et = time.time()
    print ("Finish feature extraction! Total required time for extraction : ", (total_et-total_st))

def getLatent_DB(model, FLAGS, modelName):
    batch_size = FLAGS.batch_size
    total_st = time.time()
    unitLen = FLAGS.unitLen
    print ("Start Training XD")

    model.restore("./save/"+modelName+".ckpt-"+str(FLAGS.num_model))

    total_st = time.time()
    if( (FLAGS.model) == 'GCN') ):
        for i in range(FLAGS.start_DB, FLAGS.start_DB + FLAGS.num_DB):
            _graph, _property = loadInputs(FLAGS, i, modelName, FLAGS.unitLen)
            num_batches = int(_graph[0].shape[0]/batch_size)

            st = time.time()
            latent = []
            adjacency = []
            nodes = []
            for _iter in range(num_batches):
                A_batch = _graph[0][_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]
                X_batch = _graph[1][_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]
                Z = model.get_latent_vector(A_batch, X_batch)
                X = model.get_nodes(A_batch, X_batch)
                for z in Z:
                    latent.append(z)
                for x in X:
                    nodes.append(x)
            latent = np.asarray(latent)
            np.save('./latent/'+modelName+'_'+FLAGS.validation_database+'_'+str(i)+'.npy', latent)
            np.save('./nodes/'+modelName+'_'+FLAGS.validation_database+'_'+str(i)+'.npy', nodes)


    elif( (FLAGS.model == 'GAT') ):
        for i in range(FLAGS.start_DB, FLAGS.start_DB + FLAGS.num_DB):
            _graph, _property = loadInputs(FLAGS, i, modelName, FLAGS.unitLen)
            num_batches = int(_graph[0].shape[0]/batch_size)

            st = time.time()
            latent = []
            adjacency = []
            nodes = []
            for _iter in range(num_batches):
                A_batch = _graph[0][_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]
                X_batch = _graph[1][_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]
                Z = model.get_latent_vector(A_batch, X_batch)
                _A = model.get_adjacency(A_batch, X_batch)
                X = model.get_nodes(A_batch, X_batch)
                for z in Z:
                    latent.append(z)
                for _a in _A:
                    adjacency.append(_a)
                for x in X:
                    nodes.append(x)
            latent = np.asarray(latent)
            adjacency = np.asarray(adjacency)
            np.save('./latent/'+modelName+'_'+FLAGS.validation_database+'_'+str(i)+'.npy', latent)
            np.save('./adjacency/'+modelName+'_'+FLAGS.validation_database+'_'+str(i)+'.npy', adjacency)
            np.save('./nodes/'+modelName+'_'+FLAGS.validation_database+'_'+str(i)+'.npy', nodes)

            et = time.time()
            print ("Time for feature extraction of", i, "th batch : ", et-st)
            st = time.time()

    total_et = time.time()
    print ("Finish feature extraction! Total required time for extraction : ", (total_et-total_st))

def test(model, FLAGS, modelName):
    ### For test property ###
    batch_size = FLAGS.batch_size
    total_st = time.time()
    idx_start = FLAGS.start_DB
    _dim = FLAGS.num_DB
    unitLen = FLAGS.unitLen
    print ("Start Training XD")

    P_batch_total = []
    P_pred_total = []
    model.restore("./save/"+modelName+".ckpt-"+str(FLAGS.num_model))

    for i in range(FLAGS.start_DB,FLAGS.start_DB+_dim):
        _graph, _property = loadInputs(FLAGS, i, modelName, unitLen)
        num_batches = int(_graph[0].shape[0]/batch_size)

        st = time.time()
        for _iter in range(num_batches):
            A_batch = _graph[0][_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]
            X_batch = _graph[1][_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]
            P_batch = _property[_iter*FLAGS.batch_size:(_iter+1)*FLAGS.batch_size]

            if( A_batch.shape[0] == FLAGS.batch_size):
                # Test accuracy
                Y, cost = model.test(A_batch, X_batch, P_batch)
                P_pred_total.append(Y.flatten())
                P_batch_total.append((P_batch))
        et = time.time()
        print ("time : ", et-st)

        st = time.time()

    P_batch_total = np.asarray(P_batch_total).flatten()
    P_pred_total =  np.asarray(P_pred_total).flatten()
    mae_total = np.mean(np.abs(P_batch_total - P_pred_total))
    print ("MAE :", mae_total)

    total_et = time.time()
    print ("Finish property validation! Total required time for validation :", (total_et-total_st))

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'GAT', 'Options : AE, AAE, GAN, predictor') 
flags.DEFINE_string('output', 'AE', 'Options : graph, smiles, property; logP, TPSA, ...')
flags.DEFINE_string('loss_type', 'MSE', 'Options : MSE, CrossEntropy, Hinge')  ### Using MSE or Hinge for predictor 
flags.DEFINE_string('database', 'qm9', 'Options : ZINC, ZINC2')  
flags.DEFINE_string('validation_database', 'AE_lowest', 'Options : ZINC, ZINC2, pl3ka')  ### Using MSE or Hinge for predictor 
flags.DEFINE_string('optimizer', 'Adam', 'Options : ')  ### Using MSE or Hinge for predictor 
flags.DEFINE_integer('latent_dim', 512, 'Dimension of a latent vector for autoencoder')
flags.DEFINE_integer('epoch_size', 100, 'Epoch size')
flags.DEFINE_integer('batch_size', 100, 'Batch size')
flags.DEFINE_integer('save_every', 1000, 'Save every')
flags.DEFINE_float('learning_rate', 0.0001, 'Batch size')
flags.DEFINE_float('decay_rate', 0.95, 'Batch size')
flags.DEFINE_integer('num_DB', 1, '')
flags.DEFINE_integer('start_DB', 0, '')
flags.DEFINE_integer('unitLen', 10000, '')
flags.DEFINE_integer('num_model', 120000, '')
modelName = FLAGS.model + '_' + FLAGS.output + '_' + str(FLAGS.latent_dim) + '_' + str(FLAGS.batch_size) + '_' + FLAGS.database

print ("Summary of this training & testing")
print ("Model name is", modelName)
print ("A Latent vector dimension is", str(FLAGS.latent_dim))
print ("A learning rate is", str(FLAGS.learning_rate), "with a decay rate", str(FLAGS.decay_rate))
print ("Using", FLAGS.loss_type, "for loss function in an optimization")

model = Graph2Property(FLAGS)
predict(model, FLAGS, modelName)
