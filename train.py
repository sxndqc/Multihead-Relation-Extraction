# import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, BertModel, GPT2Tokenizer, GPT2Model
from datasets import Dataset
import pickle
from torch.utils.data import DataLoader
from models import *
from torch import nn, optim

def load_data(filename):
    rel_dict = {}
    rel_cnt = 0
    train = {"labels":[], "sent": []}
    with open(filename, "r") as f:
        for line in f:
            l = line.strip()
            rel, e1, e2, sent = l.split("\t")

            # if rel[-4:]=="-Inv":
            #     swap = e1
            #     e1 = e2
            #     e2 = swap
            #     rel = rel[:-4]

            # # reduce the 19 labels to 10

            if rel not in rel_dict.keys():
                rel_dict[rel] = rel_cnt
                rel_cnt += 1
            train["labels"].append((rel_dict[rel], int(e1), int(e2)))
            
            # s = sent.strip().split(" ")
            # s[int(e1)] = "<e1>"
            # s[int(e2)] = "<e2>"

            # sent = " ".join(s)

            train["sent"].append(sent)
            # train.append({"label": rel_dict[rel], "sent": , "e1": int(e1), "e2": int(e2)})
    
    dataset = Dataset.from_dict(train)
    return dataset, rel_dict

def load_from_disk(filename):
    return Dataset.load_from_disk(filename)

def load_bert_vec(raw_dataset, transformer_name, train_batch_size= 64, dev_batch_size = 100):
    if transformer_name != "gpt2-large":
        tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    def tokenize_function(dataset):
        return tokenizer(dataset["sent"], max_length =512, truncation=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

    tokenized_dataset.set_format("torch")
    length = dataset.shape[1]
    tokenized_dataset = tokenized_dataset.remove_columns(["sent"]) # because DataCollatorWithPadding only accepts this

    splited = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True)

    train_dataloader = DataLoader(splited["train"], shuffle=True, batch_size = train_batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(splited["test"], batch_size=dev_batch_size, collate_fn=data_collator)

    return train_dataloader, eval_dataloader

def get_bert_embedding(model, encoded_input):
    # encoded_input is correct

    return model(encoded_input["input_ids"], attention_mask = encoded_input["attention_mask"])

def eval_model(attn_score, dev_data, bert_model):   
    """
    If the test is only to predict one relation in the test, then the result should be ranked
    """
    attn_score.eval() # attn_score is a model
    score = compute_result(dev_data, bert_model, attn_score)
    batches = torch.LongTensor(range(score.shape[0]))
    classes = dev_data["labels"][:, 0]
    e1s = dev_data["labels"][:, 1]
    e2s = dev_data["labels"][:, 2]
    predicted = torch.argmax(score[batches, :, e1s, e2s], dim=1)
    accuracy = torch.sum(predicted==classes) / score.shape[0]

    return accuracy

def eval_model_topk(attn_score, dev_data, bert_model, rel_dict_inv, k=3):   
    """
    If the test is only to predict one relation in the test, then the result should be ranked
    """
    attn_score.eval() # attn_score is a model
    score = compute_result(dev_data, bert_model, attn_score)
    batches = torch.LongTensor(range(score.shape[0]))
    classes = dev_data["labels"][:, 0]
    e1s = dev_data["labels"][:, 1]
    e2s = dev_data["labels"][:, 2]
    predicted = torch.topk(score[batches, :, e1s, e2s], k=k, dim=1).indices
    precision = (classes.unsqueeze(1).repeat_interleave(k, dim=1) == predicted).any(dim=1).sum(dim=0) / score.shape[0]

    for i, cate in enumerate(predicted): 
        print(f"Label {rel_dict_inv[classes[i].item()]} predicted {rel_dict_inv[predicted[i][0].item()]}, {rel_dict_inv[predicted[i][1].item()]} and {rel_dict_inv[predicted[i][2].item()]}.")


    return precision

def compute_result(data, bert_model, attn_score):

    k = get_bert_embedding(bert_model, data)
    CLS = k.pooler_output
    hidden_states = k.hidden_states
    e1_output = hidden_states[-2] + CLS.view(-1, 1, CLS.shape[-1])
    score = attn_score(hidden_states[-1], e1_output)

    return score

def load_test_data(test_name, transformer_name):
    raw_dataset, _ = load_data(test_name)
    
    if transformer_name != "gpt2-large":
        tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    def tokenize_function(dataset):
        return tokenizer(dataset["sent"], max_length =512, truncation=True)    
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_test_dataset = raw_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset.set_format("torch")
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(["sent"]) # because DataCollatorWithPadding only accepts this
    test_dataloader = DataLoader(tokenized_test_dataset, batch_size=50, collate_fn=data_collator)

    return test_dataloader

def test_infer(attn_score, test_data, bert_model):

    attn_score.eval() # attn_score is a model
    score = compute_result(test_data, bert_model, attn_score)
    batches = torch.LongTensor(range(score.shape[0]))
    classes = test_data["labels"][:, 0]
    e1s = test_data["labels"][:, 1]
    e2s = test_data["labels"][:, 2]
    predicted = torch.argmax(score[batches, :, e1s, e2s], dim=1).tolist()

    return predicted


def train(transformer_name,dataset, num_class, epoch, lr, bert_dim, rel_dict, opt="adamw", weight_decay=1, dr=0.2):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if transformer_name == "gpt2-large":
        bert_model = GPT2Model.from_pretrained("gpt2-large").to(device)
    else:
        bert_model = BertModel.from_pretrained(transformer_name, output_hidden_states=True, output_attentions = True).to(device)

    hidden_size = bert_model.config.hidden_size

    attn_score = MultiHeadAttention(input_size=hidden_size,
                                     output_size=num_class * hidden_size,
                                     num_heads=num_class,
                                     dr = dr)
    attn_score.to(device)

    for param in bert_model.parameters():
        param.requires_grad = False

    best_metric = 0
    global_step = 0
    warmup_step = 300

    params = attn_score.parameters()
    
    if opt == 'sgd':
        optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
    elif opt == 'adam':
        optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
    elif opt == 'adamw':
        optimizer = optim.AdamW(params, lr, betas=(0.9,0.999), weight_decay=weight_decay)

    for epoch in range(epoch):

        train_data, dev_data = load_bert_vec(raw_dataset = dataset, transformer_name = transformer_name)

        avg_loss = 0
        avg_acc = 0

        ckpt = './%s_%s_%s_%s_%s_%s_bert.th' % (
        transformer_name,
        "adamw",
        "sigmoid",
        "withCLS",
        "weightdecay"+str(weight_decay),
        bert_dim,)

        total_data_points = 0
        total_accuracy = 0

        for i, batch in enumerate(train_data):

            batch.to(device)

            attn_score.train()

            score = compute_result(batch, bert_model, attn_score)

            # train with this score, originally in sentenceRE, because subject_loss does not need to be calculated

            loss = loss_func(score, batch["labels"])

            avg_loss += loss.item()
     
            # warmup training

            if global_step < warmup_step:
                warmup_rate = float(global_step) / warmup_step
            else:
                warmup_rate = 1.0

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * warmup_rate
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1

            # eval of this batch ---------------
            batch_result = eval_model(attn_score, batch, bert_model)
            batch_size = batch["input_ids"].shape[0]
            total_data_points += batch_size
            total_accuracy += batch_result * batch_size

        avg_loss /= i
        result = total_accuracy / total_data_points

        print("=== Epoch %d val; loss: %.2f accuracy: %.4f===" % (epoch, avg_loss, result))
        
        if dev_data is not None:

            total_data_points = 0

            total_accuracy = 0

            for i, batch in enumerate(dev_data):

                batch.to(device)
                
                batch_result = eval_model(attn_score, batch, bert_model)
                # batch_result = eval_model_topk(attn_score, batch, bert_model, rel_dict, 3)

                batch_size = batch["input_ids"].shape[0]

                total_data_points += batch_size

                total_accuracy += batch_result * batch_size

            result = total_accuracy / total_data_points

            if result > best_metric:
                
                print("Best ckpt and saved.")

                torch.save(attn_score, ckpt)
                
                best_metric = result

            print("best_metric: %.4f; current : %.4f" % (best_metric, result))

    attn_score = torch.load(ckpt)

    return bert_model, attn_score, device


if __name__ == "__main__":
    
    transformer_name = "bert-base-cased"
    bert_dim=768
    # dataset, rel_dict = load_data("mini.train.tsv")
    # pickle.dump(rel_dict, open("rel_dict.pkl","wb"))
    # dataset.save_to_disk("mini.train.dataset")

    # rel_dict = pickle.load(open("rel_dict.pkl","rb"))
    # dataset = Dataset.load_from_disk("mini.train.dataset")

    dataset, rel_dict = load_data("semevalTrain.tsv")
    pickle.dump(rel_dict, open("rel_dict_whole.pkl","wb"))
    dataset.save_to_disk("semevalTrain.dataset")

    rel_dict_inv = {rel_dict[x]: x for x in rel_dict.keys()}

    # rel_dict = pickle.load(open("rel_dict.pkl","rb"))
    # dataset = Dataset.load_from_disk("mini.train.dataset")

    bert_model, attn_score, device = train(transformer_name, dataset, len(rel_dict), epoch = 25, lr = 8e-5, bert_dim=bert_dim, rel_dict = rel_dict_inv)

    test_dataloader = load_test_data("semevalTest_without_keys.tsv", transformer_name)

    output_f = open("test_result.txt","w")

    for batch in test_dataloader:
        batch.to(device)
        predicted = test_infer(attn_score, batch, bert_model)
        for i in predicted:
            output_f.write(rel_dict_inv[i]+"\n")

    output_f.close()




