import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

from utils import config

export_dir = 'saved_models/simple_4_4/'

clean_graph_def = None
with tf.Session(graph=tf.Graph()) as sess:
    checkpoint = tf.train.get_checkpoint_state(config.CHECKPOINT_DIR)
    ckpt = checkpoint.model_checkpoint_path
    
    saver = tf.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(config.CHECKPOINT_DIR))
    print('Model loaded.')

    graph_def = tf.get_default_graph().as_graph_def()

    print("%d ops in the graph." % len(graph_def.node))
    clean_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=graph_def,
        output_node_names=['result/dmap','result/count']
    )
    print("%d ops in the final graph." % len(clean_graph_def.node))

with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(clean_graph_def, name='')
    
    graph = tf.get_default_graph()
    graph_def = tf.get_default_graph().as_graph_def()

    for op in graph.get_operations():
        print(op.name)
    
    print('Saving model.')

    image = graph.get_tensor_by_name('inputs/image_input:0')
    #is_training = graph.get_tensor_by_name('inputs/is_training:0')
    
    dmap = graph.get_tensor_by_name('result/dmap:0')
    count = graph.get_tensor_by_name('result/count:0')
    
    inputs = {
        'image_input': image#,
        #'is_training': is_training
        }
    
    outputs = {
        'dmap': dmap,
        'count': count
        }
    
    signature = predict_signature_def(inputs=inputs, outputs=outputs)
    
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={'predict': signature},
        clear_devices=True,
        strip_default_attrs=True
    )
    
    builder.save()
        
    print('Model saved.')
