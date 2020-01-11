import csv
import math
import re
import nltk as nltk

querys = []
# Reihen gespeichert in arrays
raw_querys = []
raw_query_class = []
evaluation_query_set = []
evaluation_class_set = []
all_querys_as_text = ""
# Vokabular mit allen jemals vorgekommenen Wörtern
vocabulary = dict()
tf_matrix = []
idf_matrix = []
tf_idf_matrix = []
pmi_matrix = []


def read_in_csv_file():
    file = open("chat_data.csv", "r")
    csv_reader = csv.reader(file, delimiter=";")
    for row in csv_reader:
        if get_converted_class(row[2]) != "no class":
           raw_querys.append(row[9])
           # Bsp: 1.1, 1.2, 1.3 -> 1
           class_name = get_converted_class(row[2])
           raw_query_class.append(class_name)
    file.close()
    # Das erste Element aus dem Array wird gelöscht, da es sich dabei um das Label der Reihe handelt
    raw_querys.pop(0)
    raw_query_class.pop(0)
    raw_querys.pop(len(raw_querys) - 1)
    raw_query_class.pop(len(raw_query_class) - 1)
    get_evaluation_set()
    #Dokumente erstellen
    for index in range(len(raw_querys)):
        if len(re.findall("1", raw_query_class[index])) > 0:
            querys[0] = querys[0] + " " + (raw_querys[index])
        if len(re.findall("2", raw_query_class[index])) > 0:
            querys[1] = querys[1] + " " + (raw_querys[index])
        if len(re.findall("3", raw_query_class[index])) > 0:
            querys[2] = querys[2] + " " + raw_querys[index]
        if len(re.findall("4", raw_query_class[index])) > 0:
            querys[3] = querys[3] + " " + raw_querys[index]
        if len(re.findall("5", raw_query_class[index])) > 0:
            querys[4] = querys[4] + " " + raw_querys[index]

def get_converted_class(class_name_raw):
    if bool(re.match("^1", class_name_raw)) is True:
        return "1"
    if bool(re.match("^2", class_name_raw)) is True:
        return "2"
    if bool(re.match("^3", class_name_raw)) is True:
        return "3"
    if bool(re.match("^4", class_name_raw)) is True:
        return "4"
    if bool(re.match("^5", class_name_raw)) is True:
        return "5"
    else:
        return "no class"

def get_evaluation_set():
    # 100 test documents
    for index in range(len(raw_querys)-100,len(raw_querys)):
        evaluation_query_set.append(raw_querys[index])
        evaluation_class_set.append(raw_query_class[index])
    for index in range(100):
        raw_querys.pop(len(raw_querys)-1)
        raw_query_class.pop(len(raw_querys)-1)

def init_classes():
    for index in range(5):
        querys.append("")


def init_vocabulary():
    global all_querys_as_text
    for query in raw_querys:
        all_querys_as_text = all_querys_as_text + " " + query
    words_all_querys_as_text = nltk.tokenize.word_tokenize(all_querys_as_text, language='german')
    # Vokabular initialisieren
    for word in words_all_querys_as_text:
        if len(word) > 1 and len(re.findall(r"\b[a-zA-ZÄÖÜäöüß]+\b", word)) > 0:
            if word in vocabulary:
                vocabulary[word] = vocabulary[word] + 1
            else:
                vocabulary[word] = 1


def create_tf_matrix():
    for word in vocabulary:
        matrix_line = []
        for index in range(5):
            tf = len(re.findall(word, querys[index]))
            matrix_line.append(tf)
        tf_matrix.append(matrix_line)


def create_idf_matrix():
    # N speichert die Anzahl an Dokumenten (5)
    N = 5
    for word_index in range(len(vocabulary)):
        df = 0
        for index in range(5):
            if tf_matrix[word_index][index] != 0:
                df = df + 1
        idf = math.log(N / df)
        idf_matrix.append(idf)


def create_tf_idf_matrix():
    for vocab_index in range(len(vocabulary)):
        idf = idf_matrix[vocab_index]
        matrix_line = []
        for index in range(5):
            tf = tf_matrix[vocab_index][index]
            tf_idf = idf * tf
            matrix_line.append(tf_idf)
        tf_idf_matrix.append(matrix_line)


def calculate_amount_of_tokens():
    amount = 0
    for vocab_index in range(len(vocabulary)):
        for index in range(5):
            amount = amount + tf_matrix[vocab_index][index]
    return amount


def create_pmi_matrix():
    N = calculate_amount_of_tokens()
    vocab_length = len(vocabulary.keys())
    for vocab_index in range(vocab_length):
        matrix_line = []
        for index in range(5):
            p_dependent = tf_matrix[vocab_index][index] / N
            p_word = 0
            p_class = 0
            for index_range in range(5):
                p_word = p_word + tf_matrix[vocab_index][index_range]
            p_word = p_word / N

            for index_range in range(len(vocabulary)):
                p_class = p_class + tf_matrix[index_range][index]
            p_class = p_class / N
            if p_dependent > 0 and p_class > 0 and p_word > 0:
                pmi = math.log2(p_dependent / (p_word * p_class))
                if pmi < 0:
                    pmi = 0
            else:
                pmi = 0
            matrix_line.append(pmi)
        pmi_matrix.append(matrix_line)


def print_query_documents():
    print(querys[0])
    print(querys[1])
    print(querys[2])
    print(querys[3])
    print(querys[4])


def a1_a2_matrix():
    index_count = 0
    for word in vocabulary:
        print("Inverse Document Frequenz: ", idf_matrix[index_count])
        for index in range(5):
            print("Wort:", word, "Klasse:", str(index + 1), "Term-Frequenz:", tf_matrix[index_count][index], "tf-idf:",
                  tf_idf_matrix[index_count][index], "pmi:", pmi_matrix[index_count][index])
        print("\n")
        index_count = index_count + 1

def create_classifier(example_sentence):
    example_sentence_tokens = nltk.tokenize.word_tokenize(example_sentence, language='german')
    vocabulary_array = vocabulary.keys()
    pmi_results = []
    tf_idf_results = []
    for index in range(5):
        pmi = 0
        tf_idf = 0
        for token in example_sentence_tokens:
            counter = 0
            for vocab in vocabulary_array:
                if vocab == token:
                    pmi = pmi + pmi_matrix[counter][index]
                    tf_idf = tf_idf + tf_idf_matrix[counter][index]
                counter = counter + 1
        pmi_results.append(pmi)
        tf_idf_results.append(tf_idf)
    max_value = max(tf_idf_results)
    selected_class_tf_idf = tf_idf_results.index(max_value) + 1
    max_value = max(pmi_results)
    selected_class_pmi = pmi_results.index(max_value) + 1
    selected_classes = []
    selected_classes.append(selected_class_tf_idf)
    selected_classes.append(selected_class_pmi)
    return selected_classes

def evaluation(pmi_tf_idf_index):
    classified = []
    assigned_correctly = 0
    #print("---------")
    for index in range(100):
        # für alle 100 Sätze wird eine klasse gesucht
        classified.append(create_classifier(evaluation_query_set[index]))
        true_class = evaluation_class_set[index]
        #print("true",true_class)
        #print("assigned",str(classified[index][pmi_tf_idf_index]))
        if bool(re.search(str(classified[index][pmi_tf_idf_index]),true_class)) is True:
            #print("assigned,positive")
            assigned_correctly = assigned_correctly + 1
    return assigned_correctly


init_classes()
read_in_csv_file()
init_vocabulary()
create_tf_matrix()
create_idf_matrix()
create_tf_idf_matrix()
create_pmi_matrix()
a1_a2_matrix()
# "0" means that we will use tf-idf to evaluate the classifier
print(evaluation(0))
print("Accuracy TF_IDF:", evaluation(0)/100)
print("Accuracy: Pointwise Mutual Information", evaluation(1)/100)
# "1" means that we will use pointwise mutual information to evaluate the classifier
print(evaluation(1))
