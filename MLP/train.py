# -*- coding:utf-8 -*-
import tensorflow as tf
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from MLP import data_helper,Model

flags =tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size',64,'the batch_size of the training procedure')
flags.DEFINE_float('lr',0.1,'the learning rate')
flags.DEFINE_integer('hidden_neural_size',100,'hidden neural size')
flags.DEFINE_string('dataset_path','/home/yang/PythonProject/DeeplearningModels/data/','dataset path')
flags.DEFINE_integer('checkpoint_num',1000,'epoch num of checkpoint')
flags.DEFINE_integer('class_num',2,'class num')
flags.DEFINE_float('keep_prob',0.8,'dropout rate')
flags.DEFINE_integer('num_epoch',6,'num epoch')
flags.DEFINE_string('log_dir','/home/yang/PythonProject/DeeplearningModels/data/log/','output directory')
flags.DEFINE_string('out_dir','/home/yang/PythonProject/DeeplearningModels/data/','output directory')
flags.DEFINE_integer('check_point_every',10,'checkpoint every num step ')


def train_model():

    _x_ ,_y_ = data_helper.load_data(FLAGS.dataset_path,'data.csv')
    # Split the original dataset into train set and test set
    x_, x_test, y_, y_test = train_test_split(_x_, _y_, test_size=0.1)
    # Split the train set into train set and dev set
    x_train, x_dev, y_train, y_dev = train_test_split(x_, y_, test_size=0.1)
    print('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    print ('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))
    print("Starting training...\n")

    with tf.Graph().as_default(), tf.Session() as session:
        mlp = Model.MLP(x_train.shape[1],FLAGS.class_num,FLAGS.hidden_neural_size)
        # 可以设置 一个用于记录全局训练步骤的单值。以及使用minimize()操作，该操作不仅可以优化更新训练的模型参数，也可以为全局步骤(global step)计数
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
        train_op = optimizer.minimize(mlp.loss, global_step=global_step)

        checkpoint_dir = FLAGS.dataset_path+'model/'
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir)
        checkpoint_prefix = checkpoint_dir+'model'

        # add summary
        train_summary_dir = os.path.join(FLAGS.log_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)
        dev_summary_dir = os.path.join(FLAGS.log_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, session.graph)

        def train_step(x_batch, y_batch):
            feed_dict = {
                mlp.x:x_batch,
                mlp.y:y_batch,
                mlp.keep_prob:FLAGS.keep_prob
            }
            _, step, loss, accuracy,summary = session.run([train_op, global_step, mlp.loss, mlp.accuracy,mlp.summary], feed_dict)
            print('Accuracy on training set: {}'.format(accuracy))
            print('loss on training set: {}'.format(loss))
            return loss,accuracy,summary

        def dev_step(x_batch, y_batch):
            feed_dict = {
                mlp.x: x_batch,
                mlp.y: y_batch,
                mlp.keep_prob: 1.0
            }
            step, loss, accuracy, predictions,num_correct,summary = session.run(
                [global_step,mlp.loss, mlp.accuracy, mlp.predictions,mlp.num_correct,mlp.summary], feed_dict)
            return accuracy, loss, predictions,num_correct,summary

        saver = tf.train.Saver(tf.all_variables())
        session.run(tf.initialize_all_variables())

        # Training starts here
        train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epoch)
        best_accuracy, best_at_step = 0, 0

        # Train the model with x_train and y_train
        for train_batch in train_batches:
            x_train_batch, y_train_batch = zip(*train_batch)
            _,_,train_summary = train_step(x_train_batch, y_train_batch)
            current_step = tf.train.global_step(session, global_step)
            print(current_step)
            train_summary_writer.add_summary(train_summary, current_step)
            #train_summary_writer.flush()

            # Evaluate the model with x_dev and y_dev
            if current_step % FLAGS.check_point_every == 0:
                dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                total_dev_correct = 0
                for dev_batch in dev_batches:
                    x_dev_batch, y_dev_batch = zip(*dev_batch)
                    acc, loss, predictions,num_correct,dev_summary = dev_step(x_dev_batch, y_dev_batch)
                    total_dev_correct += num_correct
                    dev_summary_writer.add_summary(dev_summary, current_step)
                    # dev_summary_writer.flush()
                accuracy = float(total_dev_correct) / len(y_dev)
                print ('Accuracy on dev set: {}'.format(accuracy))

                if accuracy >= best_accuracy:
                    best_accuracy, best_at_step = accuracy, current_step
                    path = saver.save(session, checkpoint_prefix, global_step=current_step)
                    print('Saved model {} at step {}'.format(path, best_at_step))
                    print('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))
        print ('Training is complete, testing the best model on x_test and y_test\n')


        # Evaluate x_test and y_test
        saver.restore(session, checkpoint_prefix + '-' + str(best_at_step))
        test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), FLAGS.batch_size, 1, shuffle=False)
        total_test_correct = 0
        for test_batch in test_batches:
            x_test_batch, y_test_batch = zip(*test_batch)
            acc, loss, predictions,num_correct,summary = dev_step(x_test_batch, y_test_batch)
            total_test_correct += num_correct
        accuracy = float(total_test_correct) / len(y_test)
        print ('Accuracy on test set: {}\n'.format(accuracy))

        train_summary_writer.close()
        dev_summary_writer.close()

    # os.rename(path, FLAGS.out_dir + 'best_model.ckpt')
    # os.rename(path + '.meta', FLAGS.out_dir + 'best_model.meta')
    # shutil.rmtree(checkpoint_dir)

if __name__ == "__main__":
    train_model()

