import torch


def batchify(input_batch_list, args, o_label_id):
    """
    :param input_batch_list:
    :param gpu:
    :return:
    """
    batch_size = len(input_batch_list)
    words_token = [sent[0] for sent in input_batch_list]
    labels = [sent[1] for sent in input_batch_list]
    label_ids = []
    scopes = []
    for ele in labels:
        label_ids.append([e[0] for e in ele])
        scopes.append([e[1] for e in ele])
    token_seq_lengths = list(map(len, words_token))
    sent_lengths = list(map(len, label_ids))
    max_seq_len = max(sent_lengths)
    max_token_len = max(token_seq_lengths)

    input_ids = torch.zeros((batch_size, max_token_len), requires_grad=False).long()
    attention_mask = torch.zeros((batch_size, max_token_len), requires_grad=False, dtype=torch.float32)

    crf_mask = torch.zeros((batch_size, max_seq_len), requires_grad=False, dtype=torch.bool)
    label_seq_tensor = o_label_id*torch.ones((batch_size, max_seq_len), requires_grad=False).long()
    loss_mask = torch.zeros((batch_size, max_seq_len), requires_grad=False).byte()
    for idx, (seq, label, tokenlen, seqlen) in enumerate(zip(words_token, label_ids, token_seq_lengths, sent_lengths)):
        input_ids[idx, :tokenlen] = torch.LongTensor(seq)
        attention_mask[idx, :tokenlen] = torch.FloatTensor([1] * tokenlen)
        crf_mask[idx, :seqlen] = torch.BoolTensor([1]*seqlen)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        loss_mask[idx, 1: seqlen-1] = torch.ByteTensor([1]*(seqlen-2))
    if args.use_gpu:
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        loss_mask = loss_mask.cuda()
        crf_mask = crf_mask.cuda()
    return input_ids, attention_mask, label_seq_tensor, loss_mask, crf_mask, scopes
