import json
import pandas as pd

def main():

    label_test_path = 'label-test.json'
    label_train_path = 'label-train.json'
    internal_path = 'internal.json'
    internal_dictionary_path = 'internal_dictionary.json'
    finetune_output_path = 'finetune_data/'
    
    files = [label_test_path, label_train_path, internal_path, \
            internal_dictionary_path, finetune_output_path]

    with open(label_test_path, 'r', encoding='utf-8') as f:
        label_test = json.load(f)
        print(f"Label test size: {len(label_test)}")
        f.close()
    with open(label_train_path, 'r', encoding='utf-8') as f:
        label_train = json.load(f)
        print(f"Label train size: {len(label_train)}")
        print(f"Example: {label_train[0]}")
        f.close()
    with open(internal_path, 'r', encoding='utf-8') as f:
        internal = json.load(f)
        print(f"Internal size: {len(internal)}")
        f.close()
    with open(internal_dictionary_path, 'r', encoding='utf-8') as f:
        internal_dictionary = json.load(f)
        print(f"Internal dictionary size: {len(internal_dictionary)}")
        f.close()
    #with open(finetune_output_path, 'r', encoding='utf-8') as f:
    #    finetune_output = json.laod(f)
    #    print(f"fine")
    #    f.close()

    #test if all  pairs are in internal dictionary
    for dic in label_train_path:
        print('no match') if dic['應匹配的內規1內容'] == dic['應匹配的內規1內容'] and str(dic['應匹配的內規1內容']) not in internal_dictionary else 0
        print('no match') if dic['應匹配的內規2內容'] == dic['應匹配的內規2內容'] and str(dic['應q匹配的內規2內容']) not in internal_dictionary else 0
        print('no match') if dic['應匹配的內規3內容'] == dic['應匹配的內規3內容'] and str(dic['應匹配的內規3內容']) not in internal_dictionary else 0





if __name__ == '__main__':
    main()
