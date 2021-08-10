#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf


class Model(tf.keras.Model):

    """ Base Model for Deep Recommenders
    注意：继承该类后，必须重写compute_loss方法
    """
    
    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        """ 定义损失函数 """
        raise NotImplementedError(
            "Implementers must implement the `compute_loss` method.")

    def train_step(self, inputs):
        """ 自定义模型训练 """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(inputs, training=True)
    
            # 正则化损失
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics
        
    def test_step(self, inputs):
        """ 自定义模型验证 """
        loss = self.compute_loss(inputs, training=False)

        # 正则化损失
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss
        
        return metrics