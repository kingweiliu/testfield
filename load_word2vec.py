import tensorflow as tf
import numpy as np

def load_embedding():
    with open("/Users/liujingwei/pythonenv/tensorflow/testfield/google.bin", "rb") as f:
        header = f.readline()
        print(header)
        vocab_size, frame_size = map(int, header.split())
        print(vocab_size, frame_size)
        np_result = np.zeros((vocab_size, frame_size))
        binary_len = np.dtype('float32').itemsize * frame_size
       
        vocabulary = { }
        index_vocab = { }
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == str.encode(' '):
                    word = ''.join(word)
                    break
                if ch != '\n':
                    try:
                        word.append(ch.decode())   
                    except:
                        pass
            npv = np.fromstring(f.read(binary_len), dtype='float32') 
            vocabulary[word] = line
            index_vocab[line] = word
            np_result[line,:] = npv
#            print(line, word)
        return vocabulary, index_vocab, np_result

class Distance:


    def load_1(self):
        vocab, index_vocab, embedding = load_embedding()
        graph = tf.Graph()
        print(embedding.shape)
        
        with graph.as_default():
            with tf.device("/cpu:0"):
                sess = tf.Session(graph=graph)
                X = tf.Variable(tf.zeros([3000000, 300]),dtype=tf.float32)
                tf_embed = tf.placeholder(dtype=tf.float32, shape=(3000000, 300)) 
                x_op = X.assign(tf_embed)

                sess.run(tf.global_variables_initializer())
                x = sess.run(x_op, feed_dict= {tf_embed:embedding})
                print(x[1,:])
                print(embedding[1,:])



        
    def run(self, word):
        vocab, index_vocab, embedding = load_embedding()
        print(embedding.shape)

        print("the:", vocab.get("the"))
        print("11:", index_vocab.get(11))

        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/cpu:0"):
                sess = tf.Session(graph=graph)
                indices = [vocab.get(word) ]
                print("indices:",indices)

#tf_embed = tf.Variable(tf.random_uniform((3000000, 3000)))
#tf_embed = tf.get_variable('embed', shape=[3000000,300], initializer=tf.constant_initializer(embedding))
                tf_ids = tf.placeholder(dtype=tf.int32, shape=[None])
                X = tf.Variable(tf.zeros([3000000, 300]),dtype=tf.float32)
                tf_embed_holder = tf.placeholder(dtype=tf.float32, shape=(3000000, 300)) 
                tf_embed = X.assign(tf_embed_holder)
                sess.run(tf.global_variables_initializer())
#word_embed = sess.run(tf_embed, feed_dict= {tf_embed_holder : embedding})
#               print(word_embed[1,:])

                l2_norm_embed = tf.nn.l2_normalize(tf_embed, dim=1)
                dd = tf.nn.embedding_lookup(tf_embed, tf_ids)
                
                sim = tf.matmul(dd, l2_norm_embed, transpose_b = True)
                sim_data = sess.run(sim, feed_dict={tf_ids:indices, tf_embed_holder : embedding})
                print("sim_data.shape:", sim_data.shape)
                print("sim_data data:", sim_data[0, indices[0]])
                print("simdata type:", type(sim_data))
                print("sim_data[0,:]:", sim_data[0,:])
                fu_sim_data = -sim_data[0,:]
                sort_sim = fu_sim_data.argsort()
                print("sort_sim:", sort_sim)
                print("sort_sim.shape:",sort_sim.shape)
                nearest_indices = sort_sim[:10]
                
                print("fu_sim_data:", fu_sim_data[nearest_indices] )
                print([index_vocab.get(x) for x in nearest_indices])


            
          





if __name__ == "__main__":
#load_embedding()
    dis = Distance()
#dis.load_1()
    dis.run("this")

#sess = tf.Session()
    	
#    with tf.device("/cpu:0"):
#        indices = [vocab.get(x)  for x in ["this", "it", "that"]]
#        tf_embed = tf.Variable(tf.random_uniform((3000000, 300)))
#        tf_ids = tf.placeholder(dtype=tf.int32, shape=[None])
#        dd = tf.nn.embedding_lookup(tf_embed, tf_ids)
#        sess.run(tf.global_variables_initializer())
#        ret = sess.run(dd, feed_dict={tf_ids:indices})
#        print(ret)
#
    
