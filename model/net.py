import tensorflow as tf

from .layer import inception


def angry_fox(filters, num_layers, num_channels, kernel_size, lora):
    signals = tf.keras.Input(shape=(None, num_channels), name='signals')
    layer_inputs = signals

    layer_inputs = tf.keras.layers.Dense(units=filters, name=f"input_transform")(layer_inputs)
    layer_inputs = tf.keras.layers.Dropout(rate=0.1, name=f"input_dropout")(layer_inputs)
    layer_inputs = tf.keras.layers.LayerNormalization(epsilon=1e-12, name=f"input_norm")(layer_inputs)
    layer_inputs = tf.keras.layers.Conv1D(filters=filters*4, kernel_size=11, strides=5, padding="same", name=f"stem_conv")(layer_inputs)
    layer_inputs = tf.keras.layers.Activation(activation="gelu", name=f"stem_act")(layer_inputs)

    for i in range(1,1+num_layers,1):
        layer_inputs = inception(filters=filters, kernel_size=kernel_size, lora=lora, name=f"inception_{i}")(layer_inputs)
        if i != num_layers:
            layer_inputs = tf.keras.layers.MaxPool1D(pool_size=2, name=f"downsample_{i}_pool")(layer_inputs)
            layer_inputs = tf.keras.layers.Dense(units=filters, name=f"downsample_{i}_ffn_down")(layer_inputs)
            ln = tf.keras.layers.LayerNormalization(epsilon=1e-12, name=f"downsample_{i}_ln")(layer_inputs)
            layer_inputs = tf.keras.layers.Dense(units=filters*4, name=f"downsample_{i}_ffn_up")(ln)
            if lora:
                delta_h_down = tf.keras.layers.Dense(units=filters//2, kernel_initializer='glorot_normal', name=f"downsample_{i}_LoRA_ffn_down")(ln)
                delta_h_up = tf.keras.layers.Dense(units=filters*4, kernel_initializer='glorot_normal', name=f"downsample_{i}_LoRA_ffn_up")(delta_h_down)
                layer_inputs = tf.keras.layers.Add(name=f"downsample_{i}_LoRA_add")([layer_inputs, delta_h_up])
            layer_inputs = tf.keras.layers.Activation(activation="gelu", name=f"downsample_{i}_act")(layer_inputs)

    gap = tf.keras.layers.GlobalAveragePooling1D(name=f"gap")(layer_inputs)
    gmp = tf.keras.layers.GlobalMaxPooling1D(name=f"gmp")(layer_inputs)
    outputs = tf.keras.layers.Concatenate(axis=-1, name=f"cat")([gap, gmp])

    model = tf.keras.Model(inputs=[signals], outputs=[outputs])

    return model


def sequential_angry_fox(filters, num_layers, sequence_length, lora, dropout_rate=0.5):
    time_domain_signals = tf.keras.Input(shape=(None, sequence_length, 8), name='time_domain_signals')
    frequency_domain_signals = tf.keras.Input(shape=(None, (sequence_length//2)+1, 2), name='frequency_domain_signals')
    cut = tf.keras.Input(shape=(None,), name='cut')

    time_layer_inputs = time_domain_signals
    frequency_layer_inputs = frequency_domain_signals

    time_angry_fox = angry_fox(filters=filters, num_layers=num_layers, num_channels=8, kernel_size=9, lora=lora)
    frequency_angry_fox = angry_fox(filters=filters, num_layers=num_layers, num_channels=2, kernel_size=9, lora=lora)
    
    # if lora:
    #     for layer in time_angry_fox.layers:
    #         if "LoRA" in layer.name:
    #             layer.trainable = True
    #         else:
    #             layer.trainable = False
        
    #     for layer in frequency_angry_fox.layers:
    #         if "LoRA" in layer.name:
    #             layer.trainable = True
    #         else:
    #             layer.trainable = False
        
    time_layer_inputs = tf.keras.layers.TimeDistributed(
        layer=time_angry_fox,
        name="time_feature_extractor",
    )(time_layer_inputs)
        
    frequency_layer_inputs = tf.keras.layers.TimeDistributed(
        layer=frequency_angry_fox,
        name="frquency_feature_extractor",
    )(frequency_layer_inputs)
    
    time_layer_inputs = tf.keras.layers.Dense(units=filters, name="time_ffn")(time_layer_inputs)
    frequency_layer_inputs = tf.keras.layers.Dense(units=filters, name="frequency_ffn")(frequency_layer_inputs)
    cut_embedding = tf.keras.layers.Embedding(input_dim=2, output_dim=filters, name="cut_embedding")(cut)
    layer_inputs = tf.keras.layers.Concatenate(name="fuse_concat")([time_layer_inputs, frequency_layer_inputs, cut_embedding])
    layer_inputs = tf.keras.layers.Dropout(rate=0.1, name=f"fuse_dropout")(layer_inputs)
    ln = tf.keras.layers.LayerNormalization(epsilon=1e-12, name=f"fuse_norm")(layer_inputs)
    layer_inputs = tf.keras.layers.Dense(units=filters*4, name="fuse_ffn")(ln)
    if lora:
        delta_h_down = tf.keras.layers.Dense(units=filters//2, kernel_initializer='glorot_normal', name=f"fuse_LoRA_ffn_down")(ln)
        delta_h_up = tf.keras.layers.Dense(units=filters*4, kernel_initializer='glorot_normal', name=f"fuse_LoRA_ffn_up")(delta_h_down)
        layer_inputs = tf.keras.layers.Add(name=f"fuse_LoRA_add")([layer_inputs, delta_h_up])
    layer_inputs = tf.keras.layers.Activation(activation="gelu", name="fuse_act")(layer_inputs)

    layer_inputs = inception(filters=filters, kernel_size=3, lora=lora, name="fuse_inception")(layer_inputs)
    gap = tf.keras.layers.GlobalAveragePooling1D(name="fuse_gap")(layer_inputs)
    gmp = tf.keras.layers.GlobalMaxPooling1D(name="fuse_gmp")(layer_inputs)
    layer_inputs = tf.keras.layers.Concatenate(axis=-1, name="fuse_concate")([gap, gmp])
    layer_inputs = tf.keras.layers.Dense(units=filters, name="fuse_ffn_down")(layer_inputs)
    
    y_hat = tf.keras.layers.Dense(units=filters, name="regressor_ffn_1")(layer_inputs)
    y_hat = tf.keras.layers.Activation(activation="tanh", name="regressor_act_1")(y_hat)
    y_hat = tf.keras.layers.Dropout(rate=dropout_rate, name="regressor_dropout")(y_hat)
    y_hat = tf.keras.layers.Dense(units=1, name="regressor_ffn_2")(y_hat)
    y_hat = tf.keras.layers.Activation(activation="linear", dtype=tf.float32, name="regressor_act_2")(y_hat)

    model = tf.keras.Model(inputs=[time_domain_signals, frequency_domain_signals, cut], outputs=[y_hat], name='sequential_angry_fox')

    return model
