import tensorflow as tf
import codecs
import sys
EOS_ID = 3

CHECKPOINT_PATH = './checkpoint_ckpt'
CHECKPOINT_PATH += "-46400"
NUM_EPOCH = 100
SRC_TRAIN_DATA = 'train.cn'
TRG_TRAIN_DATA = 'train.en'
OUT_TEST_DATA = 'out.txt'
SRC_TEST_DATA = 'test.txt'


BATCH_SIZE = 20
HIDDEN_SIZE = 256
BOS_ID = 1
DECODER_LAYERS = 2
CH_WORD_SIZE = 8000
EN_WORD_SIZE = 4002


DR_PRO = 0.8
MAX_GRAD_NORM = 5
MAX_LENGTH = 50


class myModel(object):
    """
    在模型的初始化函数中定义模型要用到的变量。
    """
    def __init__(self):
        """
        定义编码器和解码器所使用的LSTM结构。
        """
        self.encLSTM1 = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        self.encLSTM2 = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        self.decLSTM = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for temp in range(DECODER_LAYERS)])
        """
        定义源语言的词向量
        """
        self.srcEmbedding = tf.get_variable("srcEmb",
                                             [CH_WORD_SIZE, HIDDEN_SIZE])
        """
        定义源语言的词向量
        """
        self.trgEmbedding = tf.get_variable("tarEmb",
                                             [EN_WORD_SIZE, HIDDEN_SIZE])

        """
        定义softmax层的变量
        """
        self.softmax_weight = tf.transpose(self.trgEmbedding)

        
        self.softmax_bias = tf.get_variable("softmax_bias", [EN_WORD_SIZE])


    def inference(self, src_input):
        """
        虽然输入只有一个句子，但因为dynamic_rnn要求输入是batch的形式，因此这里
        将输入句子整理为大小为1的batch。
        """
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        with tf.variable_scope("encoder"):
            """
            使用bidirectional_dynamic_rnn构造编码器。这一步与训练时相同。
            """
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.encLSTM1, self.encLSTM2, src_emb, src_size,
                dtype=tf.float32)
            """
            将两个LSTM的输出拼接为一个张量。
            """
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)

        with tf.variable_scope("decoder"):
            """
            定义解码器使用的注意力机制。
            """
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                HIDDEN_SIZE, enc_outputs,
                memory_sequence_length=src_size)

            """
            将解码器的循环神经网络self.decLSTM和注意力一起封装成更高层的循环神经网络。
            """
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.decLSTM, attention_mechanism,
                attention_layer_size=HIDDEN_SIZE)

        """
        设置解码的最大步数。这是为了避免在极端情况出现无限循环的问题。
        """
        MAX_DEC_LEN = 100

        with tf.variable_scope("decoder/rnn/attention_wrapper"):
            """
            使用一个变长的TensorArray来存储生成的句子。
            """
            init_array = tf.TensorArray(dtype=tf.int32, size=0,
                                        dynamic_size=True, clear_after_read=False)
            """
            填入第一个单词<BOS>作为解码器的输入。
            """
            init_array = init_array.write(0, EOS_ID)
            """
            调用attention_cell.zero_state构建初始的循环状态。循环状态包含
            循环神经网络的隐藏状态，保存生成句子的TensorArray，以及记录解码
            步数的一个整数step。
            """
            init_loop_var = (
                attention_cell.zero_state(batch_size=1, dtype=tf.float32),
                init_array, 0)

            """
            tf.while_loop的循环条件：
            循环直到解码器输出<EOS>，或者达到最大步数为止。
            """
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), EOS_ID),
                    tf.less(step, MAX_DEC_LEN-1)))

            def loop_body(state, trg_ids, step):
                """
                读取最后一步输出的单词，并读取其词向量。
                """
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding,
                                                 trg_input)
                """
                调用attention_cell向前计算一步。
                """
                dec_outputs, next_state = attention_cell.call(
                    state=state, inputs=trg_emb)
                """
                计算每个可能的输出单词对应的logit，并选取logit值最大的单词作为
                这一步的而输出。
                """
                output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
                logits = (tf.matmul(output, self.softmax_weight)
                          + self.softmax_bias)
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                """
                将这一步输出的单词写入循环状态的trg_ids中。
                """
                trg_ids = trg_ids.write(step+1, next_id[0])
                return next_state, trg_ids, step+1

            """
            执行tf.while_loop，返回最终状态。
            """
            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()

            def beamsearch(state, trg_ids, step):
                for round in range(100) :
                    if k <=0 :
                        break
                    """
                     所有可能的序列
                     """
                    all_candidates = []
                    for i in range(len(sequences)):
                        """
                        获取句子的下标和分数
                        """
                        seq, score, h, c = sequences[i]
                        """
                        获取输入单词的词向量表示
                        """
                        pre_word_embed=self.trgEmbedding(keras.tensor(seq[-1]))
                        """
                        输入神经网络中进行预测
                        """
                        o , h, c=self.inference(srcEmbedding,h,c)
                        # 将输出变成0-1的
                        output=F.softmax(o,dim=2)
                        for index in range(output.shape[2]):
                            prob=-math.log(output[0][0][index])
                            candidate=[seq+[index], score * prob, h, c]
                            all_candidates.append(candidate)
                    """
                    所有候选根据分值从大到小排序
                    """
                    ordered = sorted(all_candidates, key=lambda tup: tup[1],reverse=True)
                    """
                    选择前k个
                    """
                    sequences = ordered[:k]
                    """
                    判断是否有句子到了EOS
                    """
                    for i in range(len(sequences)):
                        if sequences[i][0][-1]==EOS_ID:
                            result_sequences.append(sequences[i])
                            k-=1
                    if len(result_sequences)==0:
                    result_sequences=sequences
                return sorted(result_sequences, key=lambda tup: tup[1],reverse=True)[0][0]                



if __name__ == "__main__":
    with codecs.open(OUT_TEST_DATA, 'w', 'utf-8') as f:
        input_id = codecs.open(
            SRC_TEST_DATA, 'r', 'utf-8').readlines()[int(sys.argv[1])]
        input_ids = [int(s) for s in input_id.strip().split()]
        """
        定义训练用的循环神经网络模型。
        """
        with tf.variable_scope("nmt_model", reuse=None):
            model = myModel()

        """
        建立解码所需的计算图。
        """
        output_op = model.inference(input_ids)

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, CHECKPOINT_PATH)

        """
        读取翻译结果。
        """
        ouputs = sess.run(output_op)
        sess.close()
        for t in ouputs:
            f.write(str(t)+' ')
        f.write('\n')

