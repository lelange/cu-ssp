from keras.activations import relu, elu
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.activations import sigmoid
from talos import early_stopper
from talos import Scan
from talos import Evaluate
from talos import Deploy


p = {'activation1':[relu, elu],
     'activation2':[relu, elu],
     'optimizer': ['Adam', "RMSprop"],
     'losses': ['logcosh', keras.losses.binary_crossentropy],
     'first_hidden_layer': [10, 8, 6],
     'second_hidden_layer': [2, 4, 6],
     'batch_size': [100, 1000, 10000],
     'epochs': [10, 15]}



def fraud_model(X_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(Dense(params['first_hidden_layer'],
                    input_shape=(29,),
                    activation=params['activation1'],
                    use_bias=True))
    model.add(Dropout(0.2))
    model.add(Dense(params['second_hidden_layer'],
                    activation=params['activation2'],
                    use_bias=True))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation=sigmoid))

    model.compile(optimizer=params['optimizer'],
                  loss=params['losses'],
                  metrics=[keras.metrics.binary_accuracy])
    history = model.fit(X_train_resampled,
                    y_train_resampled,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=1,
                    validation_data=[X_val_resampled, y_val_resampled],
                    callbacks=[early_stopper(epochs=params['epochs'],
                                                    mode='moderate',
                                                    monitor='val_loss')])
    return history, model

h = Scan(X_train_resampled,
         y_train_resampled,
         model=fraud_model,
         params=p,
         grid_downsample=0.1,
         print_params=True,
         dataset_name="creditcardfraud",
         experiment_no='1',
         reduction_metric="val_loss",
         reduce_loss=True)

e = ta.Evaluate(h)
evaluation = e.evaluate(X_test,
                        y_test,
                        model_id=None,
                        folds=folds,
                        shuffle=True,
                        metric='val_loss',
                        asc=True)

#deploy and restore
#https://neurospace.io/blog/2019/04/using-talos-for-feature-hyperparameter-optimization/