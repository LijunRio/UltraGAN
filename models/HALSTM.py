import torch
import torch.nn as nn

# ## Sentence Attention model with bias
class AttentionSentRNNv2(nn.Module):

    def __init__(self,
                 sent_gru_hidden,
                 word_gru_hidden,
                 feature_base_dim,
                 dropout,
                 bidirectional=True,
                 batch_first = False):

        super(AttentionSentRNNv2, self).__init__()

        self.sent_gru_hidden = sent_gru_hidden
        self.feature_base_dim = feature_base_dim
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        self.num_dir = 2 if bidirectional else 1

        self.sent_rnn = nn.LSTM(self.num_dir * word_gru_hidden,
                                sent_gru_hidden,
                                dropout= dropout,
                                bidirectional=bidirectional,
                                batch_first=batch_first)
        self.sent_proj_fc = nn.Linear(self.num_dir * self.sent_gru_hidden,
                                      self.num_dir * self.sent_gru_hidden)
        self.sent_context_fc = nn.Linear(self.num_dir * word_gru_hidden, 1, bias=False)

    def forward(self, word_attention_vectors, state_sent=None):

        self.sent_rnn.flatten_parameters()
        # batch x sent_len x word_hidden -> batch x sent_len x sent_hidden
        output_sent, state_sent = self.sent_rnn(word_attention_vectors, state_sent)

        # batch x sent_len x sent_hidden -> batch x sent_len x sent_hidden
        sent_squish = torch.tanh(self.sent_proj_fc(output_sent))

        # batch x sent_len x sent_hidden -> batch x sent_len
        sent_attn = self.sent_context_fc(sent_squish).squeeze(-1)

        # batch x sent_len -> batch x sent_len
        sent_attn_norm = torch.softmax(sent_attn, dim=1)

        # batch x sent_len x sent_hidden -> batch x sent_hidden
        sent_attn_vectors = output_sent * sent_attn_norm.unsqueeze(2)
        sent_attn_vectors= sent_attn_vectors.sum(dim=1)
        # sent_attn_vectors = attention_mul2(output_sent, sent_attn_norm)

        return sent_attn_vectors, state_sent, sent_attn_norm



class AttentionWordRNNv2(nn.Module):

    def __init__(self,
                 num_tokens,
                 embed_size,
                 word_gru_hidden,
                 dropout,
                 bidirectional=True,
                 batch_first = True):

        super(AttentionWordRNNv2, self).__init__()

        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        self.num_dir = 2 if bidirectional else 1
        self.embed = nn.Embedding(num_tokens, embed_size)
        self.init_weights()

        self.word_rnn = nn.LSTM(embed_size,
                                word_gru_hidden,
                                dropout= dropout,
                                bidirectional=bidirectional,
                                batch_first=batch_first)
        self.word_project_fc = nn.Linear(self.num_dir * word_gru_hidden, self.num_dir * word_gru_hidden)
        self.word_context_fc = nn.Linear(self.num_dir * word_gru_hidden, 1, bias=False)


    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, embed, state_word=None):

        # embeddings
        embedded = self.embed(embed)
        # word level rnn
        self.word_rnn.flatten_parameters()
        # batch x seq_len -> batch x seq_len x hidden_len
        output_word, state_word = self.word_rnn(embedded, state_word)
        # print('output_word', output_word.shape)
        # batch x seq_len x hidden_len -> batch * seq_len x hidden_len
        # print('routput_word', output_word.shape)
        word_squish = torch.tanh(self.word_project_fc(output_word))

        # batch * seq_len x hidden_len -> batch * seq_len
        word_attn = self.word_context_fc(word_squish).squeeze(-1)

        # batch * seq_len -> batch x seq_len
        # word_attn = word_attn.view(batch, word_len)
        word_attn_norm = torch.softmax(word_attn,dim=1)

        word_attn_vectors = output_word * word_attn_norm.unsqueeze(2)
        word_attn_vectors= word_attn_vectors.sum(dim=1)

        return word_attn_vectors, state_word, word_attn_norm
