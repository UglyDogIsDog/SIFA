"""Code for testing SIFA."""
import json
import numpy as np
import os
import sys
import medpy.metric.binary as mmb
import tensorflow as tf
import model
from stats_func import *
import data_loader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CHECKPOINT_PATH = './output/20200607-132844/sifa-39900'  # model path
# path of the .txt file storing the test filenames
KEEP_RATE = 0.5
IS_TRAINING = False
BATCH_SIZE = 64
SAMPLES = 50


class SIFA:
    """The SIFA module."""

    def __init__(self, config):
        self.keep_rate = KEEP_RATE
        self.is_training = IS_TRAINING
        self.checkpoint_pth = CHECKPOINT_PATH
        self.batch_size = BATCH_SIZE
        self._skip = bool(config['skip'])
        self._num_cls = int(config['num_cls'])
        self.samples = SAMPLES

    def model_setup(self):
        self.input_a = tf.placeholder(
            tf.float32, [
                self.batch_size,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="input_A")
        self.input_b = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="input_B")
        self.fake_pool_A = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="fake_pool_B")
        self.gt_a = tf.placeholder(
            tf.float32, [
                self.batch_size,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                self._num_cls
            ], name="gt_A")
        self.gt_b = tf.placeholder(
            tf.float32, [
                self.batch_size,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                self._num_cls
            ], name="gt_B")

        inputs = {
            'images_a': self.input_a,
            'images_b': self.input_b,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B,
        }

        outputs = model.get_outputs(
            inputs, skip=self._skip, is_training=self.is_training, keep_rate=self.keep_rate)

        self.latent_b_ll = outputs['latent_b_ll']
        self.pred_mask_b = outputs['pred_mask_b']
        self.predicter_b = tf.nn.softmax(self.pred_mask_b)

        self.latent_fake_b_ll = outputs['latent_fake_b_ll']
        self.pred_mask_fake_b = outputs['pred_mask_fake_b']
        self.predicter_fake_b = tf.nn.softmax(self.pred_mask_fake_b)

    def test(self):
        """Test Function."""

        self.inputs = data_loader.load_data(
            './data/datalist/training_mr.txt', './data/datalist/training_ct.txt', self.batch_size, False, False)

        self.model_setup()
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)

            saver.restore(sess, self.checkpoint_pth)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                # in most cases coord.should_stop() will return True
                # when there are no more samples to read
                # if num_epochs=0 then it will run for ever
                img_all = None
                gt_all = None
                pred_b_final_all = None
                pred_b_disagree_all = None
                latent_all = None
                while not coord.should_stop():

                    images_i, images_j, gts_i, gts_j = sess.run(self.inputs)
                    if sys.argv[1] == "mr":
                        inputs = {
                            "in": images_i,
                            "gt": gts_i,
                        }

                        latent = sess.run(self.latent_fake_b_ll, feed_dict={
                            self.input_a: inputs['in']})  # [B, 32, 32, 512]
                        pred = np.zeros(
                            (self.samples, inputs['in'].shape[0], model.IMG_WIDTH, model.IMG_HEIGHT, self._num_cls))
                        for i in range(self.samples):
                            pred[i] = sess.run(self.predicter_fake_b, feed_dict={
                                self.input_a: inputs['in']})

                    else:
                        inputs = {
                            "in": images_j,
                            "gt": gts_j,
                        }

                        latent = sess.run(self.latent_b_ll, feed_dict={
                            self.input_b: inputs['in']})  # [B, 32, 32, 512]
                        pred = np.zeros(
                            (self.samples, inputs['in'].shape[0], model.IMG_WIDTH, model.IMG_HEIGHT, self._num_cls))
                        for i in range(self.samples):
                            pred[i] = sess.run(self.predicter_b, feed_dict={
                                self.input_b: inputs['in']})

                    img = np.squeeze(inputs['in'], axis=3)
                    gt = np.argmax(inputs['gt'], axis=3)
                    latent = np.max(latent, (1, 2))

                    # pred [S, B, 256, 256, 5]
                    pred_b_avg = np.mean(pred, 0)  # [B, 256, 256, 5]
                    pred_b_final = np.argmax(pred_b_avg, 3)  # [B, 256, 256]
                    pred_b_final_extend = np.repeat(
                        np.expand_dims(pred_b_final, axis=0), self.samples, axis=0)  # [S, B, 256, 256]
                    pred_b_samples = np.argmax(pred, 4)  # [S, B, 256, 256]
                    pred_b_disagree = (self.samples - np.sum(pred_b_final_extend == pred_b_samples,
                                                             axis=0)) / (self.samples * 1.0)  # [B, 256, 256]

                    if pred_b_final_all is None:
                        img_all = img
                        gt_all = gt
                        pred_b_final_all = pred_b_final
                        pred_b_disagree_all = pred_b_disagree
                        latent_all = latent
                    else:
                        img_all = np.concatenate(
                            (img_all, img), axis=0)
                        gt_all = np.concatenate(
                            (gt_all, gt), axis=0)
                        pred_b_final_all = np.concatenate(
                            (pred_b_final_all, pred_b_final), axis=0)
                        pred_b_disagree_all = np.concatenate(
                            (pred_b_disagree_all, pred_b_disagree), axis=0)
                        latent_all = np.concatenate(
                            (latent_all, latent), axis=0)

            finally:
                coord.request_stop()
                coord.join(threads)

                print(pred_b_final_all.shape)

                np.save('input.npy', img_all)
                np.save('gt.npy', gt_all)
                np.save('pred.npy', pred_b_final_all)
                np.save('pred_var.npy', pred_b_disagree_all)
                np.save('latent.npy', latent_all)


if __name__ == '__main__':
    with open('./config_param.json') as config_file:
        config = json.load(config_file)

    sifa_model = SIFA(config)
    sifa_model.test()
