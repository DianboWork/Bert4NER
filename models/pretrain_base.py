from pytorch_transformers import BertModel
from bert_tagger.models.crf import CRF
import torch.nn as nn
import torch


class Token_Classification(nn.Module):
    def __init__(self, args, data):
        super(Token_Classification, self).__init__()

        self.pretrain_model = BertModel.from_pretrained(args.bert_file)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.to_crf = nn.Linear(768, data.label_alphabet.size() + 2)
        self.crf = CRF(data.label_alphabet.size(), args.use_gpu, args.average_batch)
        if args.use_gpu:
            self.to_crf = self.to_crf.cuda()
            self.pretrain_model = self.pretrain_model.cuda()
            self.gpu = True

    def forward(self, input_ids, attention_mask, crf_mask, scopes):
        crf_input = self.get_crf_input(input_ids, attention_mask, scopes)
        _, best_path = self.crf._viterbi_decode(crf_input, crf_mask)
        return best_path

    def neg_log_likelihood(self, input_ids, attention_mask, batch_label, crf_mask, scopes):
        crf_input = self.get_crf_input(input_ids, attention_mask, scopes)
        total_loss = self.crf.neg_log_likelihood_loss(crf_input, crf_mask, batch_label)
        _, best_path = self.crf._viterbi_decode(crf_input, crf_mask)
        return total_loss, best_path

    def get_crf_input(self, input_ids, attention_mask, scopes):
        pretrain_model_output = self.pretrain_model(input_ids, attention_mask=attention_mask)
        hidden_repr = self.to_crf(pretrain_model_output[0])
        max_len = max(map(len, scopes))
        repr_dim = hidden_repr.size()[-1]
        crf_input = []
        for scope, repr in zip(scopes, hidden_repr):
            c_repr = []
            for i in range(len(scope)):
                c_repr.append(torch.mean(repr[scope[i][0]: scope[i][1]], dim=0))
            c_repr = torch.stack(c_repr)
            if max_len - len(scope) > 0:
                if self.gpu:
                    pad_repr = torch.zeros(max_len - len(scope), repr_dim).cuda()
                else:
                    pad_repr = torch.zeros(max_len - len(scope), repr_dim)
                crf_input.append(torch.cat((c_repr, pad_repr), dim=0))
            else:
                crf_input.append(c_repr)
        return torch.stack(crf_input)