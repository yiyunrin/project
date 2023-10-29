def residual_block(x, filters, kernel_size):
    y = Conv2D(kernel_size=kernel_size,
               filters=filters,
               padding='same')(x)
    y = ReLU()(y)
    y = Conv2D(kernel_size=kernel_size,
               filters=filters,
               padding='same')(y)
    output = Add()([x,y])
    output = ReLU()(output)
    return output

def go_res():
    inputs = Input(shape=(19, 19, 13))
    conv5x5 = Conv2D(kernel_size=5,
                     filters=256,
                     padding="same",
                     name='conv5x5')(inputs)
    conv1x1 = Conv2D(kernel_size=1,
                     filters=256,
                     padding="same",
                     name='conv1x1')(inputs)
    outputs = Add()([conv5x5, conv1x1])
    outputs = ReLU()(outputs)
    outputs = residual_block(x=outputs,
                             filters=256,
                             kernel_size=3)
    outputs = residual_block(x=outputs,
                             filters=256,
                             kernel_size=3)
    outputs = residual_block(x=outputs,
                             filters=256,
                             kernel_size=3)
    outputs = residual_block(x=outputs,
                             filters=256,
                             kernel_size=3)
    outputs = residual_block(x=outputs,
                             filters=256,
                             kernel_size=3)
    outputs = residual_block(x=outputs,
                             filters=256,
                             kernel_size=3)
    outputs = Conv2D(kernel_size=3,
                     filters=1,
                     padding="same")(outputs)
    outputs = ReLU()(outputs)
    outputs = Flatten()(outputs)
    outputs = Softmax()(outputs)
    model = Model(inputs, outputs)
    
    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
