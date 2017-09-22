from conllu import parser
import nltk
import collections

file_list = ['id-ud-train.conllu', 'id-ud-dev.conllu']
tuple_key_list = ['lemma', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']

def raw_to_data(list_of_file_path):
    file_content = ''
    for file_path in list_of_file_path:
        with open(file_path) as file:
            for line in file.readlines():
                file_content += line
    return parser.parse(file_content)
    
def remove_unused_tuple_and_get_dict_of_tag(instances, list_of_tuple_key):
    dict_of_tag = dict()
    for instance in instances:
        for word in instance:
            for tuple_key in list_of_tuple_key:
                del word[tuple_key]
            if word['upostag'] in dict_of_tag:
                dict_of_tag[word['upostag']] += 1
            else:
                dict_of_tag[word['upostag']] = 1
    return instances, dict_of_tag

def calculate_probabilities(instances, list_of_tag):
    tag_probabilities = dict()
    tag_transition_probabilities = dict({'END': dict()})
    for tag in list_of_tag:
        tag_probabilities[tag] = dict()
        tag_transition_probabilities[tag] = dict({'START': 0})

    for instance in instances:
        prev_tag = 'START'
        for word in instance:
            curr_tag = word['upostag']
            if word['form'] in tag_probabilities[word['upostag']]:
                tag_probabilities[word['upostag']][word['form']] += 1
            else:
                tag_probabilities[word['upostag']][word['form']] = 1

            if prev_tag in tag_transition_probabilities[curr_tag]:
                tag_transition_probabilities[curr_tag][prev_tag] += 1
            else:
                tag_transition_probabilities[curr_tag][prev_tag] = 1
            prev_tag = curr_tag
        if prev_tag in tag_transition_probabilities['END']:
            tag_transition_probabilities['END'][prev_tag] += 1
        else:
            tag_transition_probabilities['END'][prev_tag] = 1

    for tag in tag_probabilities:
        for word in tag_probabilities[tag]:
            tag_probabilities[tag][word] /= float(list_of_tag[tag])
    
    for tag in tag_transition_probabilities:
        for prev_tag in tag_transition_probabilities[tag]:
            if prev_tag == 'START' or tag == 'END':
                tag_transition_probabilities[tag][prev_tag] /= float(len(instances))
            else:
                tag_transition_probabilities[tag][prev_tag] /= float(list_of_tag[tag])

    return tag_probabilities, tag_transition_probabilities


def tag(sentence, tag_prob, trans_prob):
    list_of_words = nltk.word_tokenize(sentence)
    prev_data = dict()
    words_tag = collections.OrderedDict()
    for word in list_of_words:
        words_tag[word] = dict()
        for tag in tag_prob:
            if word in tag_prob[tag]:
                words_tag[word][tag] = 0

    prev_data['START'] = dict({'prob': float(1)})
    for word in words_tag:
        temp_data = dict()
        for tag in words_tag[word]:
            max_prob = dict()
            max_prob['prob'] = 0
            for prev_tag in prev_data:
                max_prob['from'] = prev_tag
                try:
                    prob_calc = prev_data[prev_tag]['prob'] * tag_prob[tag][word] * trans_prob[tag][prev_tag]
                except KeyError as e:
                    prob_calc = 0
                if max_prob['prob'] < prob_calc:
                    max_prob['prob'] = prob_calc
                    max_prob['from'] = prev_tag
            words_tag[word][tag] = max_prob
            temp_data[tag] = words_tag[word][tag]
        prev_data = temp_data
    max_prob['prob'] = 0
    for prev_tag in prev_data:
        try:
            prob_calc = prev_data[prev_tag]['prob'] * trans_prob['END'][prev_tag]
        except KeyError as e:
            prob_calc = 0
        if max_prob['prob'] < prob_calc:
            max_prob['prob'] = prob_calc
            max_prob['from'] = prev_tag
    return words_tag, max_prob

def generate_tag(words, max_prob):
    curr_tag = max_prob['from']
    word_tag = collections.OrderedDict()
    reversed_words = reversed(words.items())
    for word, value in reversed_words:
        word_tag[word] = curr_tag
        curr_tag = value[curr_tag]['from']
    reversed_word_tag = collections.OrderedDict()
    for word, value in reversed(word_tag.items()):
        reversed_word_tag[word] = value
    return reversed_word_tag

a, b = remove_unused_tuple_and_get_dict_of_tag(raw_to_data(file_list), tuple_key_list)
c, d = calculate_probabilities(a,b)
e, f = tag('Sebuah serangan pengayauan biasanya terjadi di ladang atau dengan membakar sebuah rumah dan memenggal semua penghuninya ketika mereka melarikan diri.',c, d)
print generate_tag(e, f)