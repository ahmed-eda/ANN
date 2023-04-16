def test_cosinedense_correctness():
    X = np.random.randn(1, 20)
    model = Sequential()
    model.add(core.CosineDense(1, use_bias=True, input_shape=(20,)))
    model.compile(loss='mse', optimizer='rmsprop')
    W = model.get_weights()
    W[0] = X.T
    W[1] = np.asarray([1.])
    model.set_weights(W)
    out = model.predict(X)
    assert_allclose(out, np.ones((1, 1), dtype=K.floatx()), atol=1e-5)

    X = np.random.randn(1, 20)
    model = Sequential()
    model.add(core.CosineDense(1, use_bias=False, input_shape=(20,)))
    model.compile(loss='mse', optimizer='rmsprop')
    W = model.get_weights()
    W[0] = -2 * X.T
    model.set_weights(W)
    out = model.predict(X)
    assert_allclose(out, -np.ones((1, 1), dtype=K.floatx()), atol=1e-5)



    #################################################

    def test_cosineconvolution_2d_correctness():
    if data_format == 'channels_first':
        X = np.random.randn(1, 3, 5, 5)
        input_dim = (3, 5, 5)
        W0 = X[:, :, ::-1, ::-1]
    elif data_format == 'channels_last':
        X = np.random.randn(1, 5, 5, 3)
        input_dim = (5, 5, 3)
        W0 = X[0, :, :, :, None]

    model = Sequential()
    model.add(CosineConvolution2D(1, (5, 5), use_bias=True,
                                  input_shape=input_dim,
                                  data_format=data_format))
    model.compile(loss='mse', optimizer='rmsprop')
    W = model.get_weights()
    W[0] = W0
    W[1] = np.asarray([1.])
    model.set_weights(W)
    out = model.predict(X)
    assert_allclose(out, np.ones((1, 1, 1, 1), dtype=K.floatx()), atol=1e-5)

    model = Sequential()
    model.add(CosineConvolution2D(1, (5, 5),
                                  use_bias=False,
                                  input_shape=input_dim,
                                  data_format=data_format))
    model.compile(loss='mse', optimizer='rmsprop')
    W = model.get_weights()
    W[0] = -2 * W0
    model.set_weights(W)
    out = model.predict(X)
    assert_allclose(out, -np.ones((1, 1, 1, 1), dtype=K.floatx()), atol=1e-5)


    #################################

    def _test_equivalence(channel_order=None):

    from kfs.layers.convolutional import Convolution2DEnergy_TemporalBasis
    from keras.models import Sequential
    #from keras.layers import Flatten, Dense
    input_shape = (12, 3, 64, 64)
    if channel_order is None:
        channel_order = K.image_data_format()
    if channel_order == 'channels_last':
        input_shape = (12, 64, 64, 3)


    nn = Sequential()
    nn.add(Convolution2DEnergy_TemporalBasis(8, 16, 4, (5, 5), 7,
                                            padding='same',
                                            input_shape=input_shape,
                                            data_format=channel_order))

    rng = np.random.RandomState(42)
    datums = rng.randn(6, 12, 3, 64, 64).astype('float32')
    if channel_order == 'channels_last':
        datums = datums.transpose(0, 1, 3, 4, 2)


    nn.compile(loss='mse', optimizer='sgd')

    nn2 = Sequential()
    nn2.add(Convolution2DEnergy_TemporalCorrelation(8, 16, 4, (5, 5), 7,
                                            padding='same',
                                            input_shape=input_shape,
                                            data_format=channel_order))
    nn2.compile(loss='mse', optimizer='sgd')
    nn2.set_weights(nn.get_weights())

    pred1 = nn.predict(datums)
    pred2 = nn2.predict(datums)
    assert ((pred1 - pred2) == 0.).all()

    return nn, nn.predict(datums), nn2, nn2.predict(datums)


#######################################

def build_model(self, X):
    # assumes that data axis order is same as the backend
    input_shape = X.shape[1:]
    np.random.seed(self.random_state)
    tf.set_random_seed(self.random_state)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape, name='conv1'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), name='conv2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', name='conv3'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), name='conv4'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, name='dense1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(self.n_classes, name='dense2'))
    model.add(Activation('softmax'))

    try:
      optimizer = getattr(keras.optimizers, self.solver)
    except:
      raise NotImplementedError('optimizer not implemented in keras')
    # All optimizers with the exception of nadam take decay as named arg
    try:
      opt = optimizer(lr=self.learning_rate, decay=self.lr_decay)
    except:
      opt = optimizer(lr=self.learning_rate, schedule_decay=self.lr_decay)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    # Save initial weights so that model can be retrained with same
    # initialization
    self.initial_weights = copy.deepcopy(model.get_weights())

    self.model = model



    ##############################################

    def build_model(self, X):
    # assumes that data axis order is same as the backend
    input_shape = X.shape[1:]
    np.random.seed(self.random_state)
    tf.set_random_seed(self.random_state)

    model = Sequential()
    model.add(Conv2D(96, (3, 3), padding='same',
                     input_shape=input_shape, name='conv1'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), name='conv2', padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), strides=(2, 2), padding='same', name='conv3'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), name='conv4', padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), name='conv5', padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), strides=(2, 2), name='conv6', padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), name='conv7', padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1), name='conv8', padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1),  name='conv9', padding='valid'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax', name='activation_top'))
    model.summary()

    try:
      optimizer = getattr(keras.optimizers, self.solver)
    except:
      raise NotImplementedError('optimizer not implemented in keras')
    # All optimizers with the exception of nadam take decay as named arg
    try:
      opt = optimizer(lr=self.learning_rate, decay=self.lr_decay)
    except:
      opt = optimizer(lr=self.learning_rate, schedule_decay=self.lr_decay)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    # Save initial weights so that model can be retrained with same
    # initialization
    self.initial_weights = copy.deepcopy(model.get_weights())

    self.model = model


    ##########################################

    def test_elastic_state(self):
        with self.test_session(config=self.config) as sess:
            K.set_session(sess)

            v = 1.0 if hvd.rank() == 0 else 2.0
            model1 = keras.models.Sequential([
                keras.layers.Dense(2, activation='softmax')
            ])
            model1.build((2, 2))
            model1.set_weights(
                [np.array([[v,  v], [v, v]], dtype=np.float32),
                 np.array([v, v], dtype=np.float32)])

            model2 = keras.models.Sequential([
                keras.layers.Dense(2, activation='softmax')
            ])
            model2.build((2, 2))
            model2.set_weights(
                [np.array([[1.0,  2.0], [3.0, 4.0]], dtype=np.float32),
                 np.array([0.0, 0.0], dtype=np.float32)])

            optimizer = keras.optimizers.Adam(0.001 * hvd.size())

            state = hvd.elastic.KerasState(model1, optimizer, batch=20 + hvd.rank(), epoch=10 + hvd.rank())
            state.sync()

            model1_weights = model1.get_weights()
            model2_weights = model2.get_weights()

            # After sync, all values should match the root rank
            for w in state.model.get_weights():
                self.assertAllClose(w, np.ones_like(w))
            assert state.batch == 20
            assert state.epoch == 10

            # Partially modify then restore
            model1.set_weights(model2_weights)
            state.batch = 21
            state.epoch = 11

            state.restore()

            for w1, w2 in zip(model1.get_weights(), model1_weights):
                self.assertAllClose(w1, w2)
            assert state.batch == 20
            assert state.epoch == 10

            # Partially modify then commit
            model1.set_weights(model2_weights)
            state.batch = 21
            state.epoch = 11

            state.commit()
            state.restore()

            for w1, w2 in zip(model1.get_weights(), model2_weights):
                self.assertAllClose(w1, w2)
            assert state.batch == 21
            assert state.epoch == 11


            ############################################
            