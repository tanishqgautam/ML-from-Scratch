from tensorflow.keras.layers import (
    Input, 
    Conv2D, 
    MaxPooling2D, 
    ZeroPadding2D,
    Flatten, 
    BatchNormalization, 
    AveragePooling2D, 
    Dense, 
    Activation, 
    Add ) 

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def identity_block(x, filters): 
    x_skip = x 
    f1, f2 = filters

    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
 
    x = Add()([x, x_skip])
    x = Activation("relu")(x)

    return x


def conv_block(x, s, filters):
    x_skip = x
    f1, f2 = filters
    
    x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
     
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
    x_skip = BatchNormalization()(x_skip)

    x = Add()([x, x_skip])
    x = Activation("relu")(x)

    return x

def resnet50(num_classes):
    '''
    Model Architecture:
    Resnet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL  // conv1
        -> CONVBLOCK -> IDBLOCK * 2         // conv2_x
        -> CONVBLOCK -> IDBLOCK * 3         // conv3_x
        -> CONVBLOCK -> IDBLOCK * 5         // conv4_x
        -> CONVBLOCK -> IDBLOCK * 2         // conv5_x
        -> AVGPOOL
        -> TOPLAYER
    '''

    input_im = Input(shape=(224, 224, 3)) 
    x = ZeroPadding2D(padding=(3, 3))(input_im)
    
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    x = conv_block(x, s=1, filters=(64, 256))
    x = identity_block(x, filters=(64, 256))
    x = identity_block(x, filters=(64, 256))
    
    x = conv_block(x, s=2, filters=(128, 512))
    x = identity_block(x, filters=(128, 512))
    x = identity_block(x, filters=(128, 512))
    x = identity_block(x, filters=(128, 512))
    
    x = conv_block(x, s=2, filters=(256, 1024))
    x = identity_block(x, filters=(256, 1024))
    x = identity_block(x, filters=(256, 1024))
    x = identity_block(x, filters=(256, 1024))
    x = identity_block(x, filters=(256, 1024))
    x = identity_block(x, filters=(256, 1024))
    
    x = conv_block(x, s=2, filters=(512, 2048))
    x = identity_block(x, filters=(512, 2048))
    x = identity_block(x, filters=(512, 2048))

    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x) 

    model = Model(inputs=input_im, outputs=x, name='Resnet50')

    return model