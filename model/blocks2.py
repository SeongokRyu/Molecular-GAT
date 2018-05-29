import tensorflow as tf

def attentionMatrix(A, X, attn_weight):

    # A : [batch, N, N]
    # X : [batch, N, F']
    # weight_attn : F' 
    num_atoms = int(X.get_shape()[1])
    hidden_dim = int(X.get_shape()[2])

    _X1 = tf.einsum('ij,ajk->aik', attn_weight, tf.transpose(X, [0,2,1]))
    _X2 = tf.matmul(X, _X1)
    _A = tf.multiply(A, _X2)
    _A = tf.nn.tanh(_A)
    return _A

def graphConv1D(X, out_dim):
    _X = tf.layers.dense(X, units = out_dim, use_bias=False)
    return _X

def getSkipConnection(_X, X):
    if( int(_X.get_shape()[2]) != int(X.get_shape()[2]) ):
       out_dim = int(_X.get_shape()[2])
       _X = tf.nn.relu(_X + graphConv1D(X, out_dim)) 
    else:
       _X = tf.nn.relu(_X + X) 

    return _X

def graphConv(A, X, W, b, dim, skip_connection):
    numAtoms = int(A.get_shape()[1])
    #dim = int(W.get_shape()[1])
    b = tf.reshape(tf.tile(b, [numAtoms]), [numAtoms, dim])
    _X = tf.einsum('ijk,kl->ijl', X, W) + b
    _X = tf.matmul(A, _X)
    if(skip_connection == True):
        _X = getSkipConnection(_X, X) 
    else:
        _X = tf.nn.relu(_X)

    return _X

def graphConvAttention(A, X, W, b, attn_weight, dim, skip_connection):
    numAtoms = int(A.get_shape()[1])
    #dim = int(W.get_shape()[1])
    b = tf.reshape(tf.tile(b, [numAtoms]), [numAtoms, dim])
    _X = tf.einsum('ijk,kl->ijl', X, W) + b
    _A = attentionMatrix(A, _X, attn_weight)
    _X = tf.matmul(_A, _X)

    if(skip_connection == True):
        _X = getSkipConnection(_X, X) 
    else:
        _X = tf.nn.relu(_X)

    return _X, _A

def graphConvMulti(A, X, weight, bias, attn_weight, dim, skip_connection):
    numAtoms = int(A.get_shape()[1])
    bias_head1 = tf.reshape( tf.tile( bias['enc_head1'], [numAtoms] ), [numAtoms, dim] )
    bias_head2 = tf.reshape( tf.tile( bias['enc_head2'], [numAtoms] ), [numAtoms, dim] )
    bias_head3 = tf.reshape( tf.tile( bias['enc_head3'], [numAtoms] ), [numAtoms, dim] )
    bias_head4 = tf.reshape( tf.tile( bias['enc_head4'], [numAtoms] ), [numAtoms, dim] )

    X_head1 = tf.einsum('ijk,kl->ijl', X, weight['enc_head1']) + bias_head1
    X_head2 = tf.einsum('ijk,kl->ijl', X, weight['enc_head2']) + bias_head2
    X_head3 = tf.einsum('ijk,kl->ijl', X, weight['enc_head3']) + bias_head3
    X_head4 = tf.einsum('ijk,kl->ijl', X, weight['enc_head4']) + bias_head4

    A_head1 = attentionMatrix(A, X_head1, attn_weight['enc_attn11'])
    A_head2 = attentionMatrix(A, X_head2, attn_weight['enc_attn21'])
    A_head3 = attentionMatrix(A, X_head3, attn_weight['enc_attn31'])
    A_head4 = attentionMatrix(A, X_head4, attn_weight['enc_attn41'])

    X_head1 = tf.nn.relu(tf.matmul(A_head1, X_head1))
    X_head2 = tf.nn.relu(tf.matmul(A_head2, X_head2))
    X_head3 = tf.nn.relu(tf.matmul(A_head3, X_head3))
    X_head4 = tf.nn.relu(tf.matmul(A_head4, X_head4))

    _X = tf.concat([X_head1, X_head2, X_head3, X_head4], 2)
    _A = tf.reduce_mean( [A_head1, A_head2, A_head3, A_head4], 0) 

    return _X, _A

def graphConvMulti_sum(A, X, weight, bias, attn_weight, dim, skip_connection):

    numAtoms = int(X.get_shape()[1])
    bias_head1 = tf.reshape( tf.tile( bias['enc_head1'], [numAtoms] ), [numAtoms, dim] )
    bias_head2 = tf.reshape( tf.tile( bias['enc_head2'], [numAtoms] ), [numAtoms, dim] )
    bias_head3 = tf.reshape( tf.tile( bias['enc_head3'], [numAtoms] ), [numAtoms, dim] )
    bias_head4 = tf.reshape( tf.tile( bias['enc_head4'], [numAtoms] ), [numAtoms, dim] )

    X_head1 = tf.einsum('ijk,kl->ijl', X, weight['enc_head1']) + bias_head1
    X_head2 = tf.einsum('ijk,kl->ijl', X, weight['enc_head2']) + bias_head2
    X_head3 = tf.einsum('ijk,kl->ijl', X, weight['enc_head3']) + bias_head3
    X_head4 = tf.einsum('ijk,kl->ijl', X, weight['enc_head4']) + bias_head4

    A_head1 = attentionMatrix(A, X_head1, attn_weight['enc_attn11'])
    A_head2 = attentionMatrix(A, X_head2, attn_weight['enc_attn21'])
    A_head3 = attentionMatrix(A, X_head3, attn_weight['enc_attn31'])
    A_head4 = attentionMatrix(A, X_head4, attn_weight['enc_attn41'])

    X_head1 = tf.matmul(A_head1, X_head1)
    X_head2 = tf.matmul(A_head2, X_head2)
    X_head3 = tf.matmul(A_head3, X_head3)
    X_head4 = tf.matmul(A_head4, X_head4)

    _X = tf.nn.relu( tf.reduce_mean( [X_head1, X_head2, X_head3, X_head4], 0) )
    _A = tf.reduce_mean( [A_head1, A_head2, A_head3, A_head4], 0) 

    if(skip_connection == True):
        _X = getSkipConnection(_X, X) 
    else:
        _X = tf.nn.relu(_X)

    return _X, _A

def encoder_gat_deep(X, A, batch_size, latent_size, skip_connection):
    # X : Atomic Feature, A : Adjacency Matrix
    hidden_dim = [32, 32, 32, 16, 16, 16]
    #hidden_dim = [16, 16, 16, 16, 16, 16]
    numAtoms = int(X.get_shape()[1])
    input_dim = int(X.get_shape()[2])

    weight = {
            'enc_f1': tf.get_variable("efw1", initializer=tf.contrib.layers.xavier_initializer(), shape=[numAtoms*hidden_dim[5], latent_size], dtype=tf.float64),
    }
    bias = {
            'enc_f1': tf.get_variable("efb1", initializer=tf.contrib.layers.xavier_initializer(), shape=[latent_size], dtype=tf.float64),
    }

    weight_c1 = {
            'enc_head1': tf.get_variable("ecw1h1", initializer=tf.contrib.layers.xavier_initializer(), shape=[input_dim, hidden_dim[0]], dtype=tf.float64),
            'enc_head2': tf.get_variable("ecw1h2", initializer=tf.contrib.layers.xavier_initializer(), shape=[input_dim, hidden_dim[0]], dtype=tf.float64),
            'enc_head3': tf.get_variable("ecw1h3", initializer=tf.contrib.layers.xavier_initializer(), shape=[input_dim, hidden_dim[0]], dtype=tf.float64),
            'enc_head4': tf.get_variable("ecw1h4", initializer=tf.contrib.layers.xavier_initializer(), shape=[input_dim, hidden_dim[0]], dtype=tf.float64),
    }
    weight_c2 = {
            'enc_head1': tf.get_variable("ecw2h1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0], hidden_dim[1]], dtype=tf.float64),
            'enc_head2': tf.get_variable("ecw2h2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0], hidden_dim[1]], dtype=tf.float64),
            'enc_head3': tf.get_variable("ecw2h3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0], hidden_dim[1]], dtype=tf.float64),
            'enc_head4': tf.get_variable("ecw2h4", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0], hidden_dim[1]], dtype=tf.float64),
    }
    weight_c3 = {
            'enc_head1': tf.get_variable("ecw3h1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1], hidden_dim[2]], dtype=tf.float64),
            'enc_head2': tf.get_variable("ecw3h2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1], hidden_dim[2]], dtype=tf.float64),
            'enc_head3': tf.get_variable("ecw3h3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1], hidden_dim[2]], dtype=tf.float64),
            'enc_head4': tf.get_variable("ecw3h4", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1], hidden_dim[2]], dtype=tf.float64),
    }
    weight_c4 = {
            'enc_head1': tf.get_variable("ecw4h1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[2], hidden_dim[3]], dtype=tf.float64),
            'enc_head2': tf.get_variable("ecw4h2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[2], hidden_dim[3]], dtype=tf.float64),
            'enc_head3': tf.get_variable("ecw4h3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[2], hidden_dim[3]], dtype=tf.float64),
            'enc_head4': tf.get_variable("ecw4h4", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[2], hidden_dim[3]], dtype=tf.float64),
    }
    weight_c5 = {
            'enc_head1': tf.get_variable("ecw5h1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[3], hidden_dim[4]], dtype=tf.float64),
            'enc_head2': tf.get_variable("ecw5h2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[3], hidden_dim[4]], dtype=tf.float64),
            'enc_head3': tf.get_variable("ecw5h3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[3], hidden_dim[4]], dtype=tf.float64),
            'enc_head4': tf.get_variable("ecw5h4", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[3], hidden_dim[4]], dtype=tf.float64),
    }
    weight_c6 = {
            'enc_head1': tf.get_variable("ecw6h1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[4], hidden_dim[5]], dtype=tf.float64),
            'enc_head2': tf.get_variable("ecw6h2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[4], hidden_dim[5]], dtype=tf.float64),
            'enc_head3': tf.get_variable("ecw6h3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[4], hidden_dim[5]], dtype=tf.float64),
            'enc_head4': tf.get_variable("ecw6h4", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[4], hidden_dim[5]], dtype=tf.float64),
    }
    bias_c1 = {
            'enc_head1': tf.get_variable("ecb1h1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0]], dtype=tf.float64),
            'enc_head2': tf.get_variable("ecb1h2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0]], dtype=tf.float64),
            'enc_head3': tf.get_variable("ecb1h3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0]], dtype=tf.float64),
            'enc_head4': tf.get_variable("ecb1h4", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0]], dtype=tf.float64),
    }
    bias_c2 = {
            'enc_head1': tf.get_variable("ecb2h1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1]], dtype=tf.float64),
            'enc_head2': tf.get_variable("ecb2h2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1]], dtype=tf.float64),
            'enc_head3': tf.get_variable("ecb2h3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1]], dtype=tf.float64),
            'enc_head4': tf.get_variable("ecb2h4", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1]], dtype=tf.float64),
    }
    bias_c3 = {
            'enc_head1': tf.get_variable("ecb3h1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[2]], dtype=tf.float64),
            'enc_head2': tf.get_variable("ecb3h2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[2]], dtype=tf.float64),
            'enc_head3': tf.get_variable("ecb3h3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[2]], dtype=tf.float64),
            'enc_head4': tf.get_variable("ecb3h4", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[2]], dtype=tf.float64),
    }
    bias_c4 = {
            'enc_head1': tf.get_variable("ecb4h1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[3]], dtype=tf.float64),
            'enc_head2': tf.get_variable("ecb4h2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[3]], dtype=tf.float64),
            'enc_head3': tf.get_variable("ecb4h3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[3]], dtype=tf.float64),
            'enc_head4': tf.get_variable("ecb4h4", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[3]], dtype=tf.float64),
    }
    bias_c5 = {
            'enc_head1': tf.get_variable("ecb5h1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[4]], dtype=tf.float64),
            'enc_head2': tf.get_variable("ecb5h2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[4]], dtype=tf.float64),
            'enc_head3': tf.get_variable("ecb5h3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[4]], dtype=tf.float64),
            'enc_head4': tf.get_variable("ecb5h4", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[4]], dtype=tf.float64),
    }
    bias_c6 = {
            'enc_head1': tf.get_variable("ecb6h1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[5]], dtype=tf.float64),
            'enc_head2': tf.get_variable("ecb6h2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[5]], dtype=tf.float64),
            'enc_head3': tf.get_variable("ecb6h3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[5]], dtype=tf.float64),
            'enc_head4': tf.get_variable("ecb6h4", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[5]], dtype=tf.float64),
    }
    atten_weight_c1 = {
            'enc_attn11' : tf.get_variable("eattn1h11", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[0], hidden_dim[0]], dtype= tf.float64),
            'enc_attn21' : tf.get_variable("eattn1h21", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[0], hidden_dim[0]], dtype= tf.float64),
            'enc_attn31' : tf.get_variable("eattn1h31", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[0], hidden_dim[0]], dtype= tf.float64),
            'enc_attn41' : tf.get_variable("eattn1h41", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[0], hidden_dim[0]], dtype= tf.float64),
    }
    atten_weight_c2 = {
            'enc_attn11' : tf.get_variable("eattn2h11", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[1], hidden_dim[1]], dtype= tf.float64),
            'enc_attn21' : tf.get_variable("eattn2h21", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[1], hidden_dim[1]], dtype= tf.float64),
            'enc_attn31' : tf.get_variable("eattn2h31", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[1], hidden_dim[1]], dtype= tf.float64),
            'enc_attn41' : tf.get_variable("eattn2h41", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[1], hidden_dim[1]], dtype= tf.float64),
    }
    atten_weight_c3 = {
            'enc_attn11' : tf.get_variable("eattn3h11", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[2], hidden_dim[2]], dtype= tf.float64),
            'enc_attn21' : tf.get_variable("eattn3h21", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[2], hidden_dim[2]], dtype= tf.float64),
            'enc_attn31' : tf.get_variable("eattn3h31", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[2], hidden_dim[2]], dtype= tf.float64),
            'enc_attn41' : tf.get_variable("eattn3h41", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[2], hidden_dim[2]], dtype= tf.float64),
    }
    atten_weight_c4 = {
            'enc_attn11' : tf.get_variable("eattn4h11", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[3], hidden_dim[3]], dtype= tf.float64),
            'enc_attn21' : tf.get_variable("eattn4h21", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[3], hidden_dim[3]], dtype= tf.float64),
            'enc_attn31' : tf.get_variable("eattn4h31", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[3], hidden_dim[3]], dtype= tf.float64),
            'enc_attn41' : tf.get_variable("eattn4h41", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[3], hidden_dim[3]], dtype= tf.float64),
    }
    atten_weight_c5 = {
            'enc_attn11' : tf.get_variable("eattn5h11", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[4], hidden_dim[4]], dtype= tf.float64),
            'enc_attn21' : tf.get_variable("eattn5h21", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[4], hidden_dim[4]], dtype= tf.float64),
            'enc_attn31' : tf.get_variable("eattn5h31", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[4], hidden_dim[4]], dtype= tf.float64),
            'enc_attn41' : tf.get_variable("eattn5h41", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[4], hidden_dim[4]], dtype= tf.float64),
    }
    atten_weight_c6 = {
            'enc_attn11' : tf.get_variable("eattn6h11", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[5], hidden_dim[5]], dtype= tf.float64),
            'enc_attn21' : tf.get_variable("eattn6h21", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[5], hidden_dim[5]], dtype= tf.float64),
            'enc_attn31' : tf.get_variable("eattn6h31", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[5], hidden_dim[5]], dtype= tf.float64),
            'enc_attn41' : tf.get_variable("eattn6h41", initializer = tf.contrib.layers.xavier_initializer(), shape = [hidden_dim[5], hidden_dim[5]], dtype= tf.float64),
    }

    _X1, _A1 = graphConvMulti_sum( A, X, weight_c1, bias_c1, atten_weight_c1, hidden_dim[0], False )
    _X2, _A = graphConvMulti_sum( A, _X1, weight_c2, bias_c2, atten_weight_c2, hidden_dim[1], skip_connection )
    _X3, _A = graphConvMulti_sum( A, _X2, weight_c3, bias_c3, atten_weight_c3, hidden_dim[2], skip_connection )
    _X4, _A = graphConvMulti_sum( A, _X3, weight_c4, bias_c4, atten_weight_c4, hidden_dim[3], skip_connection )
    _X5, _A = graphConvMulti_sum( A, _X4, weight_c5, bias_c5, atten_weight_c5, hidden_dim[4], skip_connection )
    _X6, _A = graphConvMulti_sum( A, _X5, weight_c6, bias_c6, atten_weight_c6, hidden_dim[5], skip_connection )
    _Z = tf.reshape(_X6, [batch_size, -1])
    latent = tf.nn.sigmoid(tf.nn.xw_plus_b(_Z, weight['enc_f1'], bias['enc_f1']))

    return latent, _X6, _A1

def encoder_gcn_deep(X, A, batch_size, latent_size, skip_connection):
    # X : Atomic Feature, A : Adjacency Matrix
    hidden_dim = [32, 32, 32, 16, 16, 16]
    numAtoms = int(X.get_shape()[1])
    input_dim = int(X.get_shape()[2])

    weight = {
            'enc_f1': tf.get_variable("efw1", initializer=tf.contrib.layers.xavier_initializer(), shape=[numAtoms*hidden_dim[5], latent_size], dtype=tf.float64),
            'enc_c1': tf.get_variable("ecw1", initializer=tf.contrib.layers.xavier_initializer(), shape=[input_dim, hidden_dim[0]], dtype=tf.float64),
            'enc_c2': tf.get_variable("ecw2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0], hidden_dim[1]], dtype=tf.float64),
            'enc_c3': tf.get_variable("ecw3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1], hidden_dim[2]], dtype=tf.float64),
            'enc_c4': tf.get_variable("ecw4", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[2], hidden_dim[3]], dtype=tf.float64),
            'enc_c5': tf.get_variable("ecw5", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[3], hidden_dim[4]], dtype=tf.float64),
            'enc_c6': tf.get_variable("ecw6", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[4], hidden_dim[5]], dtype=tf.float64),
    }
    bias = {
            'enc_f1': tf.get_variable("efb1", initializer=tf.contrib.layers.xavier_initializer(), shape=[latent_size], dtype=tf.float64),
            'enc_c1': tf.get_variable("ecb1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0]], dtype=tf.float64),
            'enc_c2': tf.get_variable("ecb2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1]], dtype=tf.float64),
            'enc_c3': tf.get_variable("ecb3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[2]], dtype=tf.float64),
            'enc_c4': tf.get_variable("ecb4", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[3]], dtype=tf.float64),
            'enc_c5': tf.get_variable("ecb5", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[4]], dtype=tf.float64),
            'enc_c6': tf.get_variable("ecb6", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[5]], dtype=tf.float64),
    }

    _X1 = graphConv( A, X, weight['enc_c1'], bias['enc_c1'], hidden_dim[0], False )
    _X2 = graphConv( A, _X1, weight['enc_c2'], bias['enc_c2'], hidden_dim[1], skip_connection )
    _X3 = graphConv( A, _X2, weight['enc_c3'], bias['enc_c3'], hidden_dim[2], skip_connection )
    _X4 = graphConv( A, _X3, weight['enc_c4'], bias['enc_c4'], hidden_dim[3], skip_connection )
    _X5 = graphConv( A, _X4, weight['enc_c5'], bias['enc_c5'], hidden_dim[4], skip_connection )
    _X6 = graphConv( A, _X5, weight['enc_c6'], bias['enc_c6'], hidden_dim[5], skip_connection )
    _Z  = tf.reshape(_X6, [batch_size, -1])
    latent = tf.nn.sigmoid(tf.nn.xw_plus_b(_Z, weight['enc_f1'], bias['enc_f1']))

    return latent, _X6

def predictor_mlp(Z):

    Z = tf.cast(Z, tf.float64)
    latent_size = int(Z.get_shape()[1])
    hidden_dim = [latent_size, latent_size, 1]
    weight = {
        'mlp_f1': tf.get_variable("fw1", initializer=tf.contrib.layers.xavier_initializer(), shape=[latent_size, hidden_dim[0]], dtype=tf.float64),
        'mlp_f2': tf.get_variable("fw2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0], hidden_dim[1]], dtype=tf.float64),
        'mlp_f3': tf.get_variable("fw3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1], hidden_dim[2]], dtype=tf.float64),
    }
    bias = {
        'mlp_f1': tf.get_variable("fb1", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[0]], dtype=tf.float64),
        'mlp_f2': tf.get_variable("fb2", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[1]], dtype=tf.float64),
        'mlp_f3': tf.get_variable("fb3", initializer=tf.contrib.layers.xavier_initializer(), shape=[hidden_dim[2]], dtype=tf.float64),
    }
    _Y = tf.nn.relu(tf.nn.xw_plus_b(Z, weight['mlp_f1'], bias['mlp_f1']))
    _Y = tf.nn.tanh(tf.nn.xw_plus_b(_Y, weight['mlp_f2'], bias['mlp_f2']))
    _Y = tf.nn.xw_plus_b(_Y, weight['mlp_f3'], bias['mlp_f3'])

    return _Y
