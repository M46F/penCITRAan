from keras.models import Model
import tensorflow as tf
import numpy as np
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Conv2D

from .preprocessing import BatchGenerator

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from keras.optimizers import Adam


class Trainer(object):
    def __init__(self, model, input_size, max_box_per_image, labels, anchors, have_head=False):
        _, self.grid_h, self.grid_w, _ = model.output_shape
        self.input_size = input_size
        self.max_box_per_image = max_box_per_image
        self.labels = labels
        self.nb_class = len(labels)
        self.anchors = anchors
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.box = len(anchors) // 2

#        self.normalize = normalize
        self.true_boxes = Input(shape=(1, 1, 1, self.max_box_per_image, 4),name='input_boxes')

        if have_head:
#            self.model = model
            output = Reshape((self.grid_h, self.grid_w, self.box,
                              5 + self.nb_class))(model.output)
            output = Lambda(lambda args: args[0])([output, self.true_boxes])
            self.model = Model([model.input, self.true_boxes], output)

        else:
            output = Conv2D(self.box * (4 + 1 + self.nb_class),
                            (1, 1), strides=(1, 1),
                            padding='same',
                            name='DetectionLayer',
                            kernel_initializer='lecun_normal')(model.output)
            output = Reshape((self.grid_h, self.grid_w, self.box,
                              5 + self.nb_class))(output)
            output = Lambda(lambda args: args[0])([output, self.true_boxes])
            self.model = Model([model.input, self.true_boxes], output)

        # print a summary of the whole model
        self.model.summary()

    def randomize_weight(self, seed=None):
        layer = self.model.layers[-4]
        weights = layer.get_weights()

        if seed is not None:
            np.random.seed(seed)

        new_kernel = np.random.normal(size=weights[0].shape)/(self.grid_h*self.grid_w)
        new_bias   = np.random.normal(size=weights[1].shape)/(self.grid_h*self.grid_w)

        layer.set_weights([new_kernel, new_bias])

    def normalize(self, img):
        return img / 255.
 
    def train(
            self,
            tensorboard_log,
            train_imgs,  # the list of images to train the model
            valid_imgs,  # the list of images used to validate the model
            train_times=10,  # the number of time to repeat the training set, often used for small datasets
            valid_times=1,  # the number of times to repeat the validation set, often used for small datasets
            nb_epochs=100,  # number of epoches
            learning_rate=1e-4,  # the learning rate
            batch_size=32,  # the size of the batch
            warmup_epochs=4,  # number of initial batches to let the model familiarize with the new dataset
            object_scale=5.0,
            no_object_scale=1.0,
            coord_scale=1.0,
            class_scale=1.0,
            saved_weights_name='best_weights',
            debug=False):

        self.batch_size = batch_size

        self.object_scale = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale = coord_scale
        self.class_scale = class_scale

        generator_config = {
            'IMAGE_H': self.input_size,
            'IMAGE_W': self.input_size,
            'GRID_H': self.grid_h,
            'GRID_W': self.grid_w,
            'BOX': self.box,
            'LABELS': self.labels,
            'CLASS': len(self.labels),
            'ANCHORS': self.anchors,
            'BATCH_SIZE': self.batch_size,
            'TRUE_BOX_BUFFER': self.max_box_per_image,
        }

        train_generator = BatchGenerator(
            train_imgs,
            generator_config,
            norm=self.normalize)
        valid_generator = BatchGenerator(
            valid_imgs,
            generator_config,
            norm=self.normalize,
            jitter=False)

        self.warmup_batches = warmup_epochs * (
            train_times * len(train_generator) +
            valid_times * len(valid_generator))

        optimizer = Adam(
            lr=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            decay=0.0005)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer)

        ############################################
        # Make a few callbacks
        ############################################

        early_stop = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=20,
            mode='min',
            verbose=1)
        checkpoint = ModelCheckpoint(
            saved_weights_name,
            monitor='val_loss',
            verbose=1,
            save_best_only=False,
            mode='min',
            period=1)
        tensorboard = TensorBoard(
            log_dir=tensorboard_log,
            histogram_freq=0,
            #write_batch_performance=True,
            write_graph=True,
            write_images=False)

        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=len(train_generator) * train_times,
            epochs=warmup_epochs + nb_epochs,
            verbose=2 if debug else 1,
            validation_data=valid_generator,
            validation_steps=len(valid_generator) * valid_times,
            callbacks=[early_stop, checkpoint, tensorboard],
            workers=3,
            max_queue_size=8)

    def custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.to_float(
            tf.reshape(
                tf.tile(tf.range(self.grid_w), [self.grid_h]),
                (1, self.grid_h, self.grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(
            tf.concat([cell_x, cell_y], -1),
            [self.batch_size, 1, 1, self.box, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        """
        Adjust prediction
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(
            self.anchors, [1, 1, 1, self.box, 2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]
        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:
                             2]  # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[
            ..., 2:4]  # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)
        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (
            1 - y_true[..., 4]) * self.no_object_scale

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale

        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(
            self.class_wt, true_box_class) * self.class_scale
        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < self.coord_scale / 2.)
        seen = tf.assign_add(seen, 1.)

        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_batches),
                              lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                       true_box_wh + tf.ones_like(true_box_wh) * np.reshape(self.anchors, [1,1,1,self.box,2]) * no_boxes_mask,
                                       tf.ones_like(coord_mask)],
                              lambda: [true_box_xy,
                                       true_box_wh,
                                       coord_mask])
        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        loss_xy = tf.reduce_sum(
            tf.square(true_box_xy - pred_box_xy) * coord_mask) / (
                nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(
            tf.square(true_box_wh - pred_box_wh) * coord_mask) / (
                nb_coord_box + 1e-6) / 2.
        loss_conf = tf.reduce_sum(
            tf.square(true_box_conf - pred_box_conf) * conf_mask) / (
                nb_conf_box + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(
            loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = loss_xy + loss_wh + loss_conf + loss_class

        nb_true_box = tf.reduce_sum(y_true[..., 4])
        nb_pred_box = tf.reduce_sum(
            tf.to_float(true_box_conf > 0.5) *
            tf.to_float(pred_box_conf > 0.3))
        """
        Debugging code
        """
        current_recall = nb_pred_box / (nb_true_box + 1e-6)
        total_recall = tf.assign_add(total_recall, current_recall)

        loss = tf.Print(
            loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
        loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
        loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
        loss = tf.Print(
            loss, [loss_conf], message='Loss Conf \t', summarize=1000)
        loss = tf.Print(
            loss, [loss_class], message='Loss Class \t', summarize=1000)
        loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
        loss = tf.Print(
            loss, [current_recall],
            message='Current Recall \t',
            summarize=1000)
        loss = tf.Print(
            loss, [total_recall / seen],
            message='Average Recall \t',
            summarize=1000)

        return loss
