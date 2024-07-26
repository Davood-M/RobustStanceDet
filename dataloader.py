import sys
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_train(path, output_dir, batch_size=8):

    ## Single Dataset ##
    reader = pd.read_csv(path)
    print(reader.head())
    if "train_cf" in path:
      reader.rename(columns = {'original':'tweet'}, inplace = True)
    reader = reader.dropna(subset=['tweet'])

    input_ids = []
    attention_masks = []

    print('Number of training data: {:,}\n'.format(reader.shape[0]))
    text = reader.tweet.values
    if "vax_label" in reader.columns:
        labels = reader.vax_label.tolist()
    elif "stance" in reader.columns:
        labels = reader.stance.tolist()



    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    for t in text:
        encoded_dict = tokenizer.encode_plus(
            t,  # Data to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=32,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        input_ids.append(encoded_dict['input_ids'])

        # Attention mask :differentiates padding from non-padding
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels).type(torch.LongTensor).flatten()

    

    dataset = TensorDataset(input_ids, attention_masks, labels)

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )


    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )
    tokenizer.save_pretrained(output_dir)

    return train_dataloader, validation_dataloader







def CL_data_train(path, output_dir, batch_size):

    ## Single Dataset ##
    reader = pd.read_csv(path)
    print(reader.head())
    if "train_cf" in path:
      reader.rename(columns = {'original':'tweet'}, inplace = True)
    reader = reader.dropna(subset=['tweet'])

    input_ids = []
    attention_masks = []
    
    anchor_ids, anchor_attention_masks, anchor_labels, pos_ids, pos_attention_masks, pos_labels, neg_ids, neg_attention_masks, neg_labels = ([] for i in range(9))

    print('Number of training data: {:,}\n'.format(reader.shape[0]))
    text = reader.tweet.values
    if "vax_label" in reader.columns:
        labels = reader.vax_label.tolist()
    elif "stance" in reader.columns:
        labels = reader.stance.tolist()
    target = reader.target.tolist()

    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    min_num = min(len(anchor_ids), len(pos_labels), len(neg_labels))
    i = 0
    
    for t, l in zip(text, labels):
        encoded_dict = tokenizer.encode_plus(
            t, target,  # Data to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=32,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        if l == 1 & i%2==0:
            anchor_ids.append(encoded_dict['input_ids'])
            # Attention mask :differentiates padding from non-padding
            anchor_attention_masks.append(encoded_dict['attention_mask'])
            anchor_labels.append(l)
            i += 1
            
        elif l == 1 & i%2==1:
            pos_ids.append(encoded_dict['input_ids'])
            pos_attention_masks.append(encoded_dict['attention_mask'])
            pos_labels.append(l)
            i += 1
            
        elif l == 0:
            neg_ids.append(encoded_dict['input_ids'])
            neg_attention_masks.append(encoded_dict['attention_mask'])
            neg_labels.append(l)

    min_num = min(len(anchor_ids), len(pos_labels), len(neg_labels))

    anchor_ids = anchor_ids[:min_num]
    anchor_attention_masks = anchor_attention_masks[:min_num]
    anchor_labels = anchor_labels[:min_num]
    pos_ids = pos_ids[:min_num]
    pos_attention_masks = pos_attention_masks[:min_num]
    pos_labels = pos_labels[:min_num]    
    neg_ids = neg_ids[:min_num]
    neg_attention_masks = neg_attention_masks[:min_num]
    neg_labels = neg_labels[:min_num]

    # input_ids = torch.cat(input_ids, dim=0)
    # attention_masks = torch.cat(attention_masks, dim=0)
    # labels = torch.tensor(labels).type(torch.LongTensor).flatten()
    
    anchor_ids = torch.cat(anchor_ids, dim=0)
    anchor_attention_masks = torch.cat(anchor_attention_masks, dim=0)
    anchor_labels = torch.tensor(anchor_labels).type(torch.LongTensor).flatten()
    
    pos_ids = torch.cat(pos_ids, dim=0)
    pos_attention_masks = torch.cat(pos_attention_masks, dim=0)
    pos_labels = torch.tensor(pos_labels).type(torch.LongTensor).flatten()
    
    neg_ids = torch.cat(neg_ids, dim=0)
    neg_attention_masks = torch.cat(neg_attention_masks, dim=0)
    neg_labels = torch.tensor(neg_labels).type(torch.LongTensor).flatten()
    

    
    # dataset = TensorDataset(input_ids, attention_masks, labels)
    dataset = TensorDataset(anchor_ids, anchor_attention_masks, anchor_labels, pos_ids, pos_attention_masks, pos_labels, neg_ids, neg_attention_masks, neg_labels)

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )


    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )
    tokenizer.save_pretrained(output_dir)

    return train_dataloader, validation_dataloader










def data_test(test_df, model_path, batch_size=32):
    reader = test_df.sample(frac=1)
    reader = reader.dropna(subset=['tweet'])
    input_ids = []
    attention_masks = []
    # print('Number of test data: {:,}\n'.format(reader.shape[0]))
    text = reader.tweet.values
    if "vax_label" in reader.columns:
        labels = reader.vax_label.tolist()
    elif "stance" in reader.columns:
        labels = reader.stance.tolist()
    target = reader.target.tolist()

    # print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)

    for t in text:
        encoded_dict = tokenizer.encode_plus(
            t, target,  # Data to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=32,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        input_ids.append(encoded_dict['input_ids'])

        # Attention mask :differentiates padding from non-padding
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels).type(torch.LongTensor).flatten()

    test_dataset = TensorDataset(input_ids, attention_masks, labels)

    test_size = len(test_dataset)


    print('{:>5,} test samples'.format(test_size))

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.


    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size
    )

    return test_dataloader