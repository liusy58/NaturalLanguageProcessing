import tensorflow as tf
NUM_EPOCH = 100
SRC_TRAIN_DATA = 'train.cn'
TRG_TRAIN_DATA = 'train.en'
BATCH_SIZE = 20
HIDDEN_SIZE = 256
BOS_ID = 1
DECODER_LAYERS = 2
CH_WORD_SIZE = 8000
EN_WORD_SIZE = 4002

CHECKPOINT_PATH = './checkpoint_ckpt'
DR_PRO = 0.8
MAX_GRAD_NORM = 5
MAX_LENGTH = 50
########################
###NOTE:读取训练数据并创建Dataset
########################


########################
###NOTE:读取训练数据并创建Dataset
########################
def GenerateData(file_path):
    ##读取文件
    res = tf.data.TextLineDataset(file_path)

    # 根据空格将单词编号切分开并放入一个一维向量。
    res = res.map(lambda string: tf.string_split([string]).values)

    # 将字符串形式的单词编号转化为整数。
    res = res.map(lambda string: tf.string_to_number(string, tf.int32))

    # 统计每个句子的单词数量，并与句子内容一起放入Dataset中。
    res = res.map(lambda x: (x, tf.size(x)))

    return res


############
##函数名：MakeSrcTrgDataset
##作用：从原语言文件src和目标文件trg中分别读取数据，并且进行填充和batch
##############
def CombineSrcTar(src_path, trg_path, batch_size):
    ##获取源语言和目标语言
    src_data = GenerateData(src_path)
    trg_data = GenerateData(trg_path)
    """
    运用zip合成一个数据集，方便后面的处理
    由4个张量组成：
       ds[0][0]是源句子
       ds[0][1]是源句子长度
       ds[1][0]是目标句子
       ds[1][1]是目标句子长度
       """
    dataset = tf.data.Dataset.zip((src_data, trg_data))


    """
    生成解码器所需要的数据类型
    """
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[BOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))

    dataset = dataset.map(MakeTrgInput)

    """
    随机打乱训练数据。
    """
    dataset = dataset.shuffle(8000)

    """
    规定填充后输出的数据维度。
    由于源句子
    """
    padded_shapes = (
        (
            tf.TensorShape([None]),  # 源句子是长度未知的向量
            tf.TensorShape([])),  # 源句子长度是单个数字
        (
            tf.TensorShape([None]),  # 目标句子（解码器输入）是长度未知的向量
            tf.TensorShape([None]),  # 目标句子（解码器目标输出）是长度未知的向量
            tf.TensorShape([])))  # 目标句子长度是单个数字
    """
    调用padded_batch方法进行batching操作。
    """
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset


"""
定义训练模型
"""
class myModel(object):
    """
    在模型的初始化函数中定义模型要用到的变量。
    """
    def __init__(self):
        """
        定义编码器的LSTM
        """
        self.encLSTM1 = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        self.encLSTM2 = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)

        """
        定义解码器的LSTM
        """
        self.decLSTM = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
            for temp in range(DECODER_LAYERS)
        ])

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
    """
     在forward函数中定义模型的前向计算图。
     src_input, src_size, trg_input, trg_label, trg_size分别是上面
    MakeSrcTrgDataset函数产生的五种张量。
    """
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]

        """
        将输入和输出单词编号转为词向量。
        """
        srcEmb = tf.nn.embedding_lookup(self.srcEmbedding, src_input)
        tarEmb = tf.nn.embedding_lookup(self.trgEmbedding, trg_input)
        """
        在词向量上进行dropout。
        """
        srcEmb = tf.nn.dropout(srcEmb, DR_PRO)
        tarEmb = tf.nn.dropout(tarEmb, DR_PRO)
        """
        使用dynamic_rnn构造编码器。
        编码器读取源句子每个位置的词向量，输出最后一步的隐藏状态encState。
        因为编码器是一个双层LSTM，因此encState是一个包含两个LSTMStateTuple类张量的tuple，
        每个LSTMStateTuple对应编码器中的一层。 张量的维度是 [batch_size, HIDDEN_SIZE]。
        encOutputs是LSTM在每一步的输出，它的维度是[batch_size,
        max_time, HIDDEN_SIZE]
        """
        with tf.variable_scope("encoder"):
            """
            构造编码器时，使用bidirectional_dynamic_rnn构造双向循环网络。
            双向循环网络的顶层输出encOutputs是一个包含两个张量的tuple，每个张量的
            维度都是[batch_size, max_time, HIDDEN_SIZE]，代表两个LSTM在每一步的输出。
            """
            encOutputs, encState = tf.nn.bidirectional_dynamic_rnn(
                self.encLSTM1,
                self.encLSTM2,
                srcEmb,
                src_size,
                dtype=tf.float32)
            """
            将两个LSTM的输出拼接为一个张量。
            """
            encOutputs = tf.concat([encOutputs[0], encOutputs[1]], -1)

        with tf.variable_scope("decoder"):
            """
            选择注意力权重的计算模型。BahdanauAttention是使用一个隐藏层的前馈神经网络。
            memory_sequence_length是一个维度为[batch_size]的张量，代表batch
            中每个句子的长度，Attention需要根据这个信息把填充位置的注意力权重设置为0。
            """
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                HIDDEN_SIZE, encOutputs, memory_sequence_length=src_size)
            """
            将解码器的循环神经网络self.decLSTM和注意力一起封装成更高层的循环神经网络。
            """
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.decLSTM,
                attention_mechanism,
                attention_layer_size=HIDDEN_SIZE)
            """
            使用attention_cell和dynamic_rnn构造编码器。
            这里没有指定init_state，也就是没有使用编码器的输出来初始化输入，而完全依赖
            注意力作为信息来源。
            """
            DecOutputs, temp = tf.nn.dynamic_rnn(attention_cell,
                                               tarEmb,
                                               trg_size,
                                               dtype=tf.float32)
        """
        计算解码器每一步的log perplexity。
        """
        output = tf.reshape(DecOutputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(trg_label, [-1]), logits=logits)
        """
        在计算平均损失时，需要将填充位置的权重设置为0，以避免无效位置的预测干扰模型的训练。
        """
        label_weights = tf.sequence_mask(trg_size,
                                         maxlen=tf.shape(trg_label)[1],
                                         dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        """
        定义反向传播操作。
        """
        trainable_variables = tf.trainable_variables()
        """
        控制梯度大小，定义优化方法和训练步骤。
        """
        grads = tf.gradients(cost / tf.to_float(batch_size),
                             trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.9)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        return cost_per_token, train_op
"""
使用给定的模型model上训练一个epoch
每训练200步便保存一个checkpoint。
"""
def train_net(session, cost_op, train_op, saver, step):
    """
    训练一个epoch。
    重复训练步骤直至遍历完Dataset中所有数据。
    """
    while True:
        try:
            """
            运行train_op并计算损失值。训练数据在main()函数中以Dataset方式提供。
            """
            cost, temp = session.run([cost_op, train_op])
            if step % 10 == 0:
                print("After %d steps,loss is %.3f" % (step, cost))
            """
            每200步保存一个checkpoint。
            """
            if step % 200 == 0:
                saver.save(session, CHECKPOINT_PATH, global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step


if __name__ == "__main__":
    """
    定义初始化函数。
    """
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    """
    定义训练用的循环神经网络模型。
    """
    
    train_model = myModel()

    """
    定义输入数据。
    """
    data = CombineSrcTar(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    """
    定义前向计算图。输入数据以张量形式提供给forward函数。
    """
    cost_op, train_op = train_model.forward(src, src_size, trg_input,
                                            trg_label, trg_size)

    """
    训练模型。
    """
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            sess.run(iterator.initializer)
            step = train_net(sess, cost_op, train_op, saver, step)




