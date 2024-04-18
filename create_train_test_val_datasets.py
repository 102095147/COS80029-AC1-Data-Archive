import os
import json
from sklearn.model_selection import train_test_split


def get_dataset_from_files():
    directory = 'outputs'
    lines = []
    for filename in os.listdir(directory):
        if filename.endswith('_relations.txt'):
            with open(os.path.join(directory, filename), 'r') as f:
                lines.extend(f.readlines())
    return lines

def write_data_to_file(dataset, file_path):
    with open(file_path, 'w') as f:
        for line in dataset:
            f.write(line + '\n')
    print(f'Created file {file_path} with JSONL dataset')

def add_entity_ids_to_dataset(dataset):
    entities = {}
    updated_dataset = []
    for data in dataset:
        data_dict = json.loads(data)

        head_pos = data_dict['h']['pos']
        if head_pos[0] == -1 or head_pos[1] == -1:
            continue
        head = data_dict['h']['name']
        if head not in entities:
            entities[head] = 'Q' + str(len(entities))
        data_dict['h']['id'] = entities[head]

        tail_pos = data_dict['t']['pos']
        if tail_pos[0] == -1 or tail_pos[1] == -1:
            continue
        tail = data_dict['t']['name']
        if tail not in entities:
            entities[tail] = 'Q' + str(len(entities))
        data_dict['t']['id'] = entities[tail]

        updated_dataset.append(data_dict)

    return updated_dataset

def filter_unknown_relations(dataset_dict, relations):
    unknown_relations = {}
    updated_dataset_dict = []
    for data in dataset_dict:
        if (data['relation'] not in relations):
            if data['relation'] not in unknown_relations:
                unknown_relations[data['relation']] = 0
            unknown_relations[data['relation']] = unknown_relations[data['relation']] + 1
        else:
            updated_dataset_dict.append(json.dumps(data))
    
    if len(unknown_relations) > 0:
        print('\nUnknown relations found:\n')
        unknown_relations = dict(sorted(unknown_relations.items(), key=lambda x:x[1], reverse=True))
        [print(rel + ' ' + str(count)) for rel, count in unknown_relations.items()]

    return updated_dataset_dict 


def create_rel_2_id_file(relations):
    relation_ids_dict = {}

    with open('training_dataset/rel2id.json', 'w') as f:
        i = 0
        for relation in relations:
            relation_ids_dict[relation] = i
            i += 1
        f.write(json.dumps(relation_ids_dict))

    print('Created rel2id.json file from list of relations')

def main():
    print('Getting dataset from files')
    dataset = get_dataset_from_files()
    print('Adding entity ids to dataset')
    dataset_dict = add_entity_ids_to_dataset(dataset)

    relations = []
    with open('relations.txt', 'r') as f:
        relations = f.readlines()

    relations = [rel.strip() for rel in relations]

    dataset = filter_unknown_relations(dataset_dict, relations)

    #dataset = [json.dumps(data) for data in dataset_dict]

    train_ratio = 0.80
    val_ratio = 0.10
    test_ratio = 0.10

    print('Splitting dataset in training, validation and testing')
    train, test = train_test_split(dataset, test_size=1 - train_ratio)
    val, test = train_test_split(test, test_size=test_ratio/(test_ratio + val_ratio))

    print('Training dataset count: ', len(train))
    print('Validation dataset count: ', len(val))
    print('Testing dataset count: ', len(test))

    write_data_to_file(train, 'training_dataset/train.txt')
    write_data_to_file(val, 'training_dataset/val.txt')
    write_data_to_file(test, 'training_dataset/test.txt')
    
    create_rel_2_id_file(relations)

if __name__ == '__main__':
    main()