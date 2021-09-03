import json
import numpy as np

from sklearn.model_selection import train_test_split

def main():
    with open('train.json', 'r') as f:
        data = json.load(f)
        f.close()
    data = np.array(data)
    train, test = train_test_split(data, test_size=0.05, shuffle=True)
    train=train.tolist()
    test=test.tolist()
    print(len(train), len(test))
    with open('train_data.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, indent=4,ensure_ascii=False)

    with open('test_data.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, indent=4, ensure_ascii=False)
if __name__ == '__main__':
    main()
