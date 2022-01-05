import argparse
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def parse_args():
    
    parser = argparse.ArgumentParser()

    # 사용자가 전달한 하이퍼 파라미터를 command-line argument로 전달받아 사용함
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    
    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()

    
def get_train_data(train_dir):
    
    x_train = np.load(os.path.join(train_dir, 'data_train_x.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(train_dir, 'data_train_y.npy'), allow_pickle=True)
    print('x_train', x_train.shape,'y_train', y_train.shape)

    return x_train, y_train


def get_validation_data(validation_dir):
    
    x_validation = np.load(os.path.join(validation_dir, 'data_val_x.npy'), allow_pickle=True)
    y_validation = np.load(os.path.join(validation_dir, 'data_val_y.npy'), allow_pickle=True)
    print('x_validation', x_validation.shape,'y_validation', y_validation.shape)

    return x_validation, y_validation

if __name__ == "__main__":
    args, _ = parse_args()
    
    x_train, y_train = get_train_data(args.train)
    x_validation, y_validation = get_validation_data(args.validation)
    
    device = '/cpu:0' 
    print(device)
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    print('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, learning_rate))

    with tf.device(device):
        model = tf.keras.Sequential([
                # input layer
                tf.keras.layers.Dense(30, input_shape=(30,), activation='relu'),
                tf.keras.layers.Dense(15, activation='relu'),
                tf.keras.layers.Dense(10,activation = 'relu'),
                # we use sigmoid for binary output
                # output layer
                tf.keras.layers.Dense(1, activation='sigmoid')
            ]
        )

        model.summary()
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'mse'])    
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                  validation_data=(x_validation, y_validation))

        # evaluate on test set
        scores = model.evaluate(x_validation, y_validation, batch_size, verbose=2)
        print("\nTest MSE :", scores)
        
        model.save(args.model_dir + '/1')
