import codecs, re, emoji
from pytorch_transformers import BertTokenizer


def char_is_emoji(character):
    return character in emoji.UNICODE_EMOJI


# def read_instance(input_doc, max_sent_len, label_alphabet, bert_file):
#     berttokenizer = BertTokenizer.from_pretrained(bert_file, do_lower_case=False)
#     special_tokens_dict = {'additional_special_tokens': ["[URL]", "[emoji]"]}  #
#     berttokenizer.add_special_tokens(special_tokens_dict)
#     instance_ids = []
#     instance_texts = []
#     file = codecs.open(input_doc)
#     sent_id = []
#     sent_text = []
#     label_id = []
#     label_text = []
#     i = 0
#     while True:
#         line = file.readline()
#         if not line:
#             break
#         line = line.rstrip().split()
#         if len(line) == 1 and line[0][0:6] == 'IMGID:':
#             assert line[0][0:6] == 'IMGID:'
#             image = line[0][6:]
#         elif len(line) == 1 and line[0][0:6] != 'IMGID:':
#             pass
#         elif len(line) == 2:
#             if re.findall("http://", line[0]):
#                 token_id = berttokenizer.encode("[URL]")
#                 sent_id.extend(token_id)
#                 sent_text.append((line[0], [i, i + 1]))
#                 label_text.append((line[1], [i, i + 1]))
#                 i = i + 1
#                 label_id.append(label_alphabet.get_index(line[1]))
#             elif char_is_emoji(line[0]):
#                 token_id = berttokenizer.encode("[emoji]")
#                 sent_id.extend(token_id)
#                 sent_text.append((line[0], [i, i + 1]))
#                 label_text.append((line[1], [i, i + 1]))
#                 i = i + 1
#                 label_id.append(label_alphabet.get_index(line[1]))
#             else:
#                 token_id = berttokenizer.encode(line[0])
#                 if len(token_id) == 1:
#                     label_id.append(label_alphabet.get_index(line[1]))
#                 else:
#                     label = line[1][2:]
#                     p = line[1][0]
#                     if p == "I" or p == "O":
#                         label_id.extend([label_alphabet.get_index(line[1])] * len(token_id))
#                     else:
#                         label_list = ["I-" + label] * len(token_id)
#                         label_list[0] = line[1]
#                         label_id.extend([label_alphabet.get_index(ele) for ele in label_list])
#                 sent_id.extend(token_id)
#                 sent_text.append((line[0], [i, i + len(token_id)]))
#                 label_text.append((line[1], [i, i + len(token_id)]))
#                 i = i + len(token_id)
#         else:
#             assert len(line) == 0
#             assert len(sent_id) == len(label_id) == sent_text[-1][1][1] == label_text[-1][1][1]
#             assert len(label_text) == len(sent_text)
#             if len(sent_id) > max_sent_len:
#                 sent_id = sent_id[:max_sent_len]
#                 label_id = label_id[:max_sent_len]
#             if len(sent_id) != 0:
#                 sent_id = [berttokenizer.convert_tokens_to_ids("[CLS]")] + sent_id + [
#                     berttokenizer.convert_tokens_to_ids("[SEP]")]
#                 label_id = [label_alphabet.get_index("O")] + label_id + [label_alphabet.get_index("O")]
#                 instance_ids.append([sent_id, label_id, image])
#                 instance_texts.append([sent_text, label_text, image])
#             sent_id = []
#             label_id = []
#             sent_text = []
#             label_text = []
#             i = 0
#     return instance_ids, instance_texts

def read_instance(input_doc, label_alphabet, bert_file):
    berttokenizer = BertTokenizer.from_pretrained(bert_file, do_lower_case=False)
    special_tokens_dict = {'additional_special_tokens': ["[URL]", "[emoji]"]}  #
    berttokenizer.add_special_tokens(special_tokens_dict)
    instance_ids = []
    instance_texts = []
    file = codecs.open(input_doc)
    sent_id = []
    sent_text = []
    label_text = []
    i = 0
    while True:
        line = file.readline()
        if not line:
            break
        line = line.rstrip().split()
        if len(line) == 1 and line[0][0:6] == 'IMGID:':
            assert line[0][0:6] == 'IMGID:'
            image = line[0][6:]
            sent_text.append(("[CLS]", [i, i + 1]))
            label_text.append(("O", [i, i + 1]))
            i = i + 1
        elif len(line) == 1 and line[0][0:6] != 'IMGID:':
            pass
        elif len(line) == 2:
            if re.findall("http://", line[0]):
                token_id = berttokenizer.encode("[URL]")
                sent_id.extend(token_id)
                sent_text.append((line[0], [i, i + 1]))
                label_text.append((line[1], [i, i + 1]))
                i = i + 1
            elif char_is_emoji(line[0]):
                token_id = berttokenizer.encode("[emoji]")
                sent_id.extend(token_id)
                sent_text.append((line[0], [i, i + 1]))
                label_text.append((line[1], [i, i + 1]))
                i = i + 1
            else:
                token_id = berttokenizer.encode(line[0])
                if len(token_id) != 0:
                    sent_id.extend(token_id)
                    sent_text.append((line[0], [i, i + len(token_id)]))
                    label_text.append((line[1], [i, i + len(token_id)]))
                    i = i + len(token_id)
        else:
            label_id = [(label_alphabet.get_index(ele[0]), ele[1]) for ele in label_text]
            assert len(line) == 0
            assert len(label_text) == len(sent_text)
            if len(sent_id) != 0:
                sent_id = [berttokenizer.convert_tokens_to_ids("[CLS]")] + sent_id + [
                    berttokenizer.convert_tokens_to_ids("[SEP]")]
                sent_text.append(("[SEP]", [i, i + 1]))
                label_text.append(("O", [i, i + 1]))
                assert len(sent_id) == sent_text[-1][1][1] == label_text[-1][1][1]
                instance_ids.append([sent_id, label_id, image])
                instance_texts.append([sent_text, label_text, image])
            sent_id = []
            sent_text = []
            label_text = []
            i = 0
    return instance_ids, instance_texts