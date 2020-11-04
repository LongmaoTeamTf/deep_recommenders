"""
@Description: Deep Interest Network
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-08-07 15:46:28
@LastEditors: Wang Yao
@LastEditTime: 2020-08-11 10:59:10
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer


class LocalActivationUnit(Layer):
    """局部激活单元"""
    
    def __init__(self, 
                 hidden_units,
                 hidden_activation='relu', 
                 regularizer=None, 
                 trainable=True,
                 relevance='diff',
                 in_product=True,
                 **kwargs):
        super(LocalActivationUnit, self).__init__(**kwargs)
        self._hidden_units = hidden_units
        self._hidden_activation = activations.get(hidden_activation)
        self._regularizer = regularizers.get(regularizer)
        self._trainable = trainable
        self._relevance = relevance
        self._in_product = in_product
        
    def build(self, input_shape):
        if self._relevance == 'diff':
            _concat_shape = input_shape[0][-1] * 3
        elif self._relevance == 'out_product':
            # 待补充
            pass
        else:
            raise ValueError("Invalid relevance `{}`. "
                             "Should be in [`diff`, `out_product`]")
        if self._in_product is True:
            _concat_shape += 1
        self._kernel_weights = self.add_weight(
            shape=(_concat_shape, self._hidden_units),
            initializer=initializers.glorot_uniform,
            regularizer=self._regularizer,
            trainable=self._trainable,
            name='kernel_weights'
        )
        self._linear_weights = self.add_weight(
            shape=(self._hidden_units, 1),
            initializer=initializers.glorot_uniform,
            regularizer=self._regularizer,
            trainable=self._trainable,
            name='linear_weights'
        )
        super(LocalActivationUnit, self).build(input_shape)
    
    @tf.function
    def call(self, inputs):
        user_behavior_input, candidate_input = inputs
        if self._relevance == 'diff':
            relevance_rep = user_behavior_input - candidate_input
        elif self._relevance == 'out_product':
            # 待补充
            pass    
        if self._in_product is True:
            inner_prod = K.dot(user_behavior_input, K.transpose(candidate_input))
            relevance_rep = K.concatenate((relevance_rep, inner_prod), axis=-1)
        ff_inputs = K.concatenate(
            (
                user_behavior_input,
                relevance_rep,
                candidate_input
            ), axis=-1
        )
        outputs = K.dot(ff_inputs, self._kernel_weights)
        outputs = K.dot(outputs, self._linear_weights)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_units': self._hidden_units,
            'hidden_activation': self._hidden_activation,
            'regularizer': self._regularizer,
            'trainable': self._trainable,
            'relevance': self._relevance,
            'in_product': self._in_product
        })
        return config


class InterestsAttention(Layer):
    """用户兴趣注意力"""
    
    def __init__(self, **kwargs):
        super(InterestsAttention, self).__init__(**kwargs)

    def call(self, inputs):
        user_behavior_inputs, interests_weights = inputs
        user_behavior_inputs = K.stack(user_behavior_inputs, axis=1)
        interests_weights = K.stack(interests_weights, axis=1)
        sum_pooling = K.sum(
            tf.multiply(user_behavior_inputs, interests_weights),
            axis=1
        )
        return sum_pooling

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])


class AdaptiveEmbedding(Layer):
    """Mini-batch Aware Regularizer"""

    def __init__(self, 
                 embedding_dim=10, 
                 regularizer='l2', 
                 trainable=True, 
                 **kwargs):
        super(AdaptiveEmbedding, self).__init__(**kwargs)
        self._regularizer = regularizers.get(regularizer)
        self._embedding_dim = embedding_dim
        self._trainable = trainable

    def build(self, input_shape):
        self._embeddings = self.add_weight(
            shape=(input_shape[-1], self._embedding_dim),
            initializer=initializers.glorot_uniform,
            trainable=True,
            name='embeddings'
        )
        self.built = True

    @tf.function
    def _l2_adaptive_norm(self, inputs, lambdas):
        return tf.multiply(lambdas, K.sum(K.square(inputs)))

    @tf.function   
    def call(self, inputs):
        indices = tf.math.argmax(inputs, output_type=tf.int32, axis=-1)
        embeddings = tf.nn.embedding_lookup(self._embeddings, indices)
        return embeddings

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1], self._embedding_dim)


class FeedForward(Layer):
    """前馈神经网络（PRelu/Dice）"""
    
    def __init__(self, kernel_size, activation='dice', **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self._kernel_size = kernel_size
        if activation in ('dice', 'prelu'):
            self._activation = activation
        else:
            self._activation = activations.get(activation)

    def build(self, input_shape):
        self._kernel_weights = self.add_weight(
            shape=(input_shape[-1], self._kernel_size),
            initializer=initializers.glorot_uniform,
            trainable=True,
            name='kernel_weights'
        )
        if self._activation in ('dice', 'prelu'):
            self._alpha = self.add_weight(
                shape=(self._kernel_size,),
                initializer=initializers.glorot_uniform,
                trainable=True,
                name='alpha'
            )
        super(FeedForward, self).build(input_shape)

    @tf.function
    def _prelu(self, inputs):
        """Parametric Rectified Linear Unit."""
        pos = K.relu(inputs)
        neg = -self._alpha * K.relu(-inputs)
        return pos + neg

    @tf.function
    def _dice(self, inputs, epsilon=1e-8):
        """Dice Adaptive Activation."""
        mean = K.mean(inputs, axis=0)
        var = K.std(inputs, axis=0)
        indicator = (inputs-mean) / (K.sqrt(var + epsilon))
        indicator = K.sigmoid(indicator)
        pos = K.relu(inputs)
        neg = -self._alpha * K.relu(-inputs)
        return indicator * pos + (1. - indicator) * neg
        
    def call(self, inputs):
        outputs = K.dot(inputs, self._kernel_weights)
        if callable(self._activation):
            outputs = self._activation(outputs)
        elif self._activation == 'dice':
            outputs = self._dice(outputs)
        elif self._activation == 'prelu':
            outputs = self._prelu(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self._kernel_size)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel_size': self._kernel_size,
            'activation': self._activation
        })
        return config


def din_builder():
    user_profile_input = tf.keras.layers.Input(shape=(10,), name='user_profile_input')
    candidate_item_input = tf.keras.layers.Input(shape=(10,), name="candidate_item_input")

    num_behaviors = 5

    user_behaviors_inputs = []
    for i in range(num_behaviors):
        user_behaviors_inputs.append(
            tf.keras.layers.Input(shape=(10,), name=f"user_behaviors_{i}")
        )

    weights = []
    for i in range(num_behaviors):
        au = LocalActivationUnit(36)
        weight = au([user_behaviors_inputs[i], candidate_item_input])
        weights.append(weight)

    att = InterestsAttention()
    att_outputs = att([user_behaviors_inputs, weights])

    x = tf.keras.layers.Concatenate()((user_profile_input, att_outputs, candidate_item_input))

    x = FeedForward(200, activation='dice')(x)
    x = FeedForward(80, activation='dice')(x)
    x = FeedForward(2, activation=None)(x)
    
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.models.Model(
        inputs=[user_profile_input, user_behaviors_inputs, candidate_item_input],
        outputs=outputs
    )

    model.summary()

    tf.keras.utils.plot_model(model, to_file='din.png')
    

    


if __name__ == "__main__":
    inputs = tf.keras.layers.Input(shape=(2, ))
    x = AdaptiveEmbedding()(inputs)
    # x = tf.keras.layers.Embedding(2, 10)(inputs)
    model = tf.keras.models.Model(inputs, x)

    data = [[1, 0], [0, 1]]

    print(model.predict(data))

    
    
    

        