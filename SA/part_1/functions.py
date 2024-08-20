from torch import nn
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model import ModelSA
import os
from copy import deepcopy
from collections import Counter
from utils import getDataLoaders, Lang

def loss_function(sentiments, sample, lang: Lang):
    # breakpoint()
    sent_count = Counter(sample['y_sents'].flatten().tolist())
    sent_weights = torch.tensor([1/(sent_count[x]+1) for x in lang.id2sent.keys()]).float().to(sentiments.device)
    criterion_sentiment = nn.CrossEntropyLoss(weight=sent_weights, ignore_index=lang.pad_token)

    # breakpoint()
    loss = criterion_sentiment(sentiments, sample['y_sents'])

    return loss
        
def train_loop(data, optimizer, model, lang, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        sentiments = model(sample['utterances'], sample['attention'], sample['mapping'])

        # breakpoint()
        loss = loss_function(sentiments, sample, lang)

        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def match_ts(gold_ts_sequence, pred_ts_sequence, lang):
    """
    calculate the number of correctly predicted targeted sentiment
    :param gold_ts_sequence: gold standard targeted sentiment sequence
    :param pred_ts_sequence: predicted targeted sentiment sequence
    :return:
    """
    # positive, negative and neutral
    n_classes = len(lang.sent2id)-1
    hit_count, gold_count, pred_count = np.zeros(n_classes), np.zeros(n_classes), np.zeros(n_classes)

    tag2tagid = lang.sent2id
    for i in range(len(gold_ts_sequence)):
        g_ts = gold_ts_sequence[i]
        p_ts = pred_ts_sequence[i]
        g_ts_id = tag2tagid[g_ts]-1
        p_ts_id = tag2tagid[p_ts]-1

        if g_ts == p_ts:
            hit_count[g_ts_id] += 1
        gold_count[g_ts_id] += 1
        pred_count[p_ts_id] += 1

    # breakpoint()
    return hit_count, gold_count, pred_count

SMALL_POSITIVE_CONST = 1e-4
def evaluate_ts(gold_ts, pred_ts, lang):
    """
    evaluate the model performance for the ts task
    :param gold_ts: gold standard ts tags
    :param pred_ts: predicted ts tags
    :return:
    """
    assert len(gold_ts) == len(pred_ts)

    n_classes = len(lang.sent2id)-1
    n_samples = len(gold_ts)
    # number of true postive, gold standard, predicted targeted sentiment
    n_tp_ts, n_gold_ts, n_pred_ts = np.zeros(n_classes), np.zeros(n_classes), np.zeros(n_classes)
    ts_precision, ts_recall, ts_f1 = np.zeros(n_classes), np.zeros(n_classes), np.zeros(n_classes)

    for i in range(n_samples):
        g_ts = gold_ts[i]
        p_ts = pred_ts[i]
        # g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=g_ts), tag2ts(ts_tag_sequence=p_ts)
        # breakpoint()
        hit_ts_count, gold_ts_count, pred_ts_count = match_ts(gold_ts_sequence=g_ts, pred_ts_sequence=p_ts, lang=lang)

        n_tp_ts += hit_ts_count
        n_gold_ts += gold_ts_count
        n_pred_ts += pred_ts_count
        # calculate macro-average scores for ts task
    for i in range(n_classes):
        n_ts = n_tp_ts[i]
        n_g_ts = n_gold_ts[i]
        n_p_ts = n_pred_ts[i]
        ts_precision[i] = float(n_ts) / float(n_p_ts + SMALL_POSITIVE_CONST)
        ts_recall[i] = float(n_ts) / float(n_g_ts + SMALL_POSITIVE_CONST)
        ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + SMALL_POSITIVE_CONST)

    ts_macro_f1 = ts_f1.mean()

    # calculate micro-average scores for ts task
    n_tp_total = sum(n_tp_ts)
    # total sum of TP and FN
    n_g_total = sum(n_gold_ts)
    # total sum of TP and FP
    n_p_total = sum(n_pred_ts)

    ts_micro_p = float(n_tp_total) / (n_p_total + SMALL_POSITIVE_CONST)
    ts_micro_r = float(n_tp_total) / (n_g_total + SMALL_POSITIVE_CONST)
    ts_micro_f1 = 2 * ts_micro_p * ts_micro_r / (ts_micro_p + ts_micro_r + SMALL_POSITIVE_CONST)
    ts_scores = {
        "macro f1": ts_macro_f1,
        "micro p": ts_micro_p,
        "micro r": ts_micro_r,
        "micro f1": ts_micro_f1,
    }
        
    return ts_scores

def sentiment_inference(sentiments, sample, lang: Lang):
    # Slot inference 
    ref_slots = []
    hyp_slots = []
    output_slots = torch.argmax(sentiments, dim=1)
    for id_seq, seq in enumerate(output_slots):
        length = sample['sent_len'][id_seq].tolist()
        # utterance = sample['sentence'][id_seq]

        gt_ids = sample['y_sents'][id_seq].tolist()
        gt_sents = [lang.id2sent[elem] for elem in gt_ids[:length]]

        ref_slots.append(gt_sents)    
    
        to_decode = seq[:length].tolist()

        # compute the hypothesis slots
        tmp_seq = []
        for id_el, elem in enumerate(to_decode):
            tmp_seq.append(lang.id2sent[elem])
        hyp_slots.append(tmp_seq)
    
    # breakpoint()
    return ref_slots, hyp_slots

def eval_loop(data, model, lang):
    model.eval()
    loss_array = []
    
    ref_sent = []
    hyp_sent = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            sentiments = model(sample['utterances'], sample['attention'], sample['mapping'])

            loss = loss_function(sentiments, sample, lang) 

            loss_array.append(loss.item())
            
            # Slot inference
            ref_slots_sample, hyp_slots_sample = sentiment_inference(sentiments, sample, lang)
            ref_sent.extend(ref_slots_sample)
            hyp_sent.extend(hyp_slots_sample)
    
    try:            
        results = evaluate_ts(ref_sent, hyp_sent, lang)
    except Exception as ex:
        print("\nWarning:", ex)
        ref_s = set([x[1] for x in ref_sent])
        hyp_s = set([x[1] for x in hyp_sent])
        print(hyp_s.difference(ref_s))
        results = { "macro f1": 0, "micro p": 0, "micro r": 0, "micro f1": 0 }

    # breakpoint()
    return results, loss_array


def runTest(test_name, device,
            bert_model, dropoutBertEmb, classification_layers,
            runs, n_epochs, lr, clip, patience, batchsize):
    print("Running test", test_name)

    train_loader, dev_loader, test_loader, lang = getDataLoaders(batchsize=batchsize, bert_model=bert_model)
    out_sents = len(lang.sent2id)

    macro_f1, micro_f1, precision, recall = [], [], [], []
    best_f1_runs = 0
    best_model_runs = None

    for x in tqdm(range(0, runs)):
        model = ModelSA(out_sents, bert_model, dropoutBertEmb, classification_layers).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        best_model = None
        pat = patience
        for x in tqdm(range(1,n_epochs)):
            loss = train_loop(train_loader, optimizer, model, clip=clip, lang=lang)
            
            if x % 5 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())

                results_dev, loss_dev = eval_loop(dev_loader, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())

                if results_dev['macro f1'] > best_f1:
                    # print(f"\nNew best F1: {f1}")
                    best_f1 = results_dev['macro f1']
                    best_model = deepcopy(model).to('cpu')
                else:
                    pat -= 1
                if pat <= 0: # Early stopping with patient
                    break # Not nice but it keeps the code clean
        # breakpoint()
        if best_model is None:
            print("No best model found?")
            # breakpoint()
            best_model = model
        results_test, _ = eval_loop(test_loader, best_model, lang)
        
        macro_f1.append(results_test['macro f1'])
        micro_f1.append(results_test['micro f1'])
        precision.append(results_test['micro p'])
        recall.append(results_test['micro r'])

        if results_test['macro f1'] > best_f1_runs:
            best_f1_runs = results_test['macro f1']
            best_model_runs = deepcopy(best_model)
            #show plot
            # plt.plot(sampled_epochs, losses_train, label='Train')
            # plt.plot(sampled_epochs, losses_dev, label='Dev')
            # plt.legend()
            # plt.title(f"Loss {test_name}")
            # plt.show()

    macro_f1 = np.asarray(macro_f1)
    micro_f1 = np.asarray(micro_f1)
    precision = np.asarray(precision)
    recall = np.asarray(recall)
    print('Macro F1', round(macro_f1.mean(),3), '+-', round(macro_f1.std(),3))
    print('Micro F1', round(micro_f1.mean(),3), '+-', round(micro_f1.std(),3))
    print('Precision', round(precision.mean(),3), '+-', round(precision.std(),3))
    print('Recall', round(recall.mean(),3), '+-', round(recall.std(),3))

    PATH = os.path.join("models", test_name+".pt")
    saving_object = {"epoch": x, 
                  "model": best_model_runs.state_dict(), 
                  "model_params": {
                        "out_sents": out_sents,
                        "bert_model": bert_model,
                        "dropoutBertEmb": dropoutBertEmb,
                        "classification_layers": classification_layers
                  },
                  "lang": lang,
                  "results": {
                        "macro_f1": macro_f1.mean(),
                        "macro_f1_std": macro_f1.std(),
                        "micro_f1": micro_f1.mean(),
                        "micro_f1_std": micro_f1.std(),
                        "precision": precision.mean(),
                        "precision_std": precision.std(),
                        "recall": recall.mean(),
                        "recall_std": recall.std()
                  },
                  }
    torch.save(saving_object, PATH)
