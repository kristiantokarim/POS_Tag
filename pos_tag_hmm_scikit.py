from conllu import parser
from nltk.tag import hmm
import nltk
from nltk.probability import LidstoneProbDist
from nltk.tag.perceptron import PerceptronTagger
file_list = ['id-ud-train.conllu', 'id-ud-dev.conllu']
test_file = ['test.conllu']
tuple_key_list = ['lemma', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']

def raw_to_data(list_of_file_path):
    file_content = ''
    for file_path in list_of_file_path:
        with open(file_path) as file:
            for line in file.readlines():
                file_content += line
    return parser.parse(file_content)

def remove_unused_attribute(instances, tuple_keys):
    for instance in instances:
        for word in instance:
            for tuple_key in tuple_keys:
                del word[tuple_key]
    data = list()
    return instances

def gen_data_for_nltk(instances):
    seq = list()
    symbol = set()
    state = set()
    for instance in instances:
        words = list()
        for word in instance:
            words.append((word['form'], word['upostag']))
            state.add(word['upostag'])
            symbol.add(word['form'])
        seq.append(words)
    return seq, list(state), list(symbol)

def get_correct_and_total(test_sequen, tagger):
    correct = 0
    total = 0
    for instance in test_sequen:
        list_of_words = list()
        for word in instance:
            w, tag = word
            list_of_words.append(w)
        results = tagger.tag(list_of_words)
        i = 0
        for word in instance:
            w, tag = word
            w2, tag2 = results[i]
            if tag == tag2:
                correct += 1
            total += 1
            i += 1
    return correct, total
    
instances = remove_unused_attribute(raw_to_data(file_list), tuple_key_list)
test_instances = instances[:1041]
instances = instances[1041:]
sequences, states, symbols = gen_data_for_nltk(instances)
test_seq, test_states, test_sym = gen_data_for_nltk(test_instances)

trainer = hmm.HiddenMarkovModelTrainer(states)
hmm_model = trainer.train_supervised(sequences, estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins))
print 'HMM'
print hmm_model.test(test_seq)

tagger = PerceptronTagger(states)
tagger.train(sequences)
print 'PERCEPTRON'
correct, total = get_correct_and_total(test_seq, tagger)
print correct
print total