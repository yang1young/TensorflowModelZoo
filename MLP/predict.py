import tensorflow as tf
import pandas as pd
import numpy as np
import data_helper
import train as t
import Model

def predict_new_data(new_data_path,data_name,model_path):
    x,y = data_helper.load_unseen_data(new_data_path,data_name)

    with tf.Graph().as_default(), tf.Session() as session:
        mlp = Model.MLP(x.shape[1],t.FLAGS.class_num,t.FLAGS.hidden_neural_size)

        def predict(x_batch):
            feed_dict = {
                mlp.x:x_batch,
                mlp.keep_prob:1.0
            }
            predicts = session.run([mlp.predictions], feed_dict)
            return predicts

        checkpoint_file = model_path
        saver = tf.train.Saver(tf.all_variables())
        saver = tf.train.import_meta_graph(model_path+'.meta')
        saver.restore(session, checkpoint_file)
        print('{} has been loaded'.format(checkpoint_file))

        batches = data_helper.batch_iter(list(x), t.FLAGS.batch_size, 1, shuffle=False)

        predictions = []
        for x_batch in batches:
            batch_predictions = predict(x_batch)[0]
            for batch_prediction in batch_predictions:
                #print(batch_prediction)
                predictions.append(batch_prediction)
        print(y)
        print(predictions)
        # df['PREDICTED'] = predict_labels
        # columns = sorted(df.columns, reverse=True)
        # df.to_csv(predicted_dir + 'predictions_all.csv', index=False, columns=columns, sep='|')
        #
        y_test = np.array(np.argmax(y, axis=1))
        accuracy = sum(np.array(predictions) == y_test) / float(len(y_test))
        print ('The prediction accuracy is: {}'.format(accuracy))


if __name__ == '__main__':
    predict_new_data(t.FLAGS.dataset_path,'data.csv','/home/yang/PythonProject/DeeplearningModels/data/model/model-1350')