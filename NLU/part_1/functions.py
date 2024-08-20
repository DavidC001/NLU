from torch import nn
import torch
from conll import evaluate
from sklearn.metrics import classification_report
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model import ModelIAS
import os
from copy import deepcopy
from collections import Counter



def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def slot_inference(slots, sample, lang):
    # Slot inference 
    ref_slots = []
    hyp_slots = []
    output_slots = torch.argmax(slots, dim=1)
    for id_seq, seq in enumerate(output_slots):
        length = sample['slots_len'][id_seq].tolist()
        utt_ids = sample['utterance'][id_seq][:length].tolist()
        gt_ids = sample['y_slots'][id_seq].tolist()
        gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
        utterance = [lang.id2word[elem] for elem in utt_ids]
        to_decode = seq[:length].tolist()
        ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
        tmp_seq = []
        for id_el, elem in enumerate(to_decode):
            tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
        hyp_slots.append(tmp_seq)
    
    return ref_slots, hyp_slots

# Function to compute intent weight
def compute_intent_weight(intent, sample, lang):
    out_intents = torch.argmax(intent, dim=1).tolist()
    gt_intents = torch.tensor(sample['intents']).tolist()

    out_intents = [lang.id2intent[idx] for idx in out_intents]
    gt_intents = [lang.id2intent[idx] for idx in gt_intents]

    report_intent = classification_report(gt_intents, out_intents, zero_division=False, output_dict=True)
    # if report_intent['accuracy'] == 0:
    #     print("Warning: Intent accuracy is 0")
    accuracy = max(report_intent['accuracy'], 1e-5)
    return max(1 / accuracy, 1e-5)

# Function to compute slot weight
def compute_slot_weight(slots, sample, lang):
    ref_slots_sample, hyp_slots_sample = slot_inference(slots, sample, lang)
    try:
        results = evaluate(ref_slots_sample, hyp_slots_sample)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        results = {"total":{"f":-1}}
    f1 = max(results['total']['f'], 1e-5)
    return max(1 / f1, 1e-5)

def loss_function(intent, slots, sample, lang):
    #define positive weights for the loss function
    intent_count = Counter(sample['intents'].tolist())
    intent_weights = torch.tensor([1/(intent_count[x]+1) for x in lang.id2intent.keys()]).float().to(intent.device)
    criterion_intents = nn.CrossEntropyLoss(weight=intent_weights)

    # breakpoint()
    slot_count = Counter(sample['y_slots'].flatten().tolist())
    slot_weights = torch.tensor([1/(slot_count[x]+1) for x in lang.id2slot.keys()]).float().to(intent.device)
    criterion_slots = nn.CrossEntropyLoss(weight=slot_weights, ignore_index=lang.word2id['pad'])

    loss_intent = criterion_intents(intent, sample['intents'])
    loss_slot = criterion_slots(slots, sample['y_slots'])

    # print(f"Loss Intent: {loss_intent:.3f} Loss Slot: {loss_slot:.3f}\nFinal Loss: {loss_intent + loss_slot:.3f}")

    # compute f1 for slots and accuracy for intent to use as rescaling factor
    # only use 10% of the samples to compute the rescaling factor
    sample_idxs = np.random.choice(len(sample['intents']), int(len(sample['intents']) * 0.1), replace=False)
    samples = {key: [sample[key][i] for i in sample_idxs] for key in sample}
    weight_intent = compute_intent_weight(intent[sample_idxs], samples, lang)
    weight_slot = compute_slot_weight(slots[sample_idxs], samples, lang)

    loss = loss_intent * weight_intent + loss_slot * weight_slot
    return loss
        

def train_loop(data, optimizer, model, lang, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len'])

        loss = loss_function(intent, slots, sample, lang)

        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])

            loss = loss_function(intents, slots, sample, lang) 

            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference
            ref_slots_sample, hyp_slots_sample = slot_inference(slots, sample, lang)
            ref_slots.extend(ref_slots_sample)
            hyp_slots.extend(hyp_slots_sample)
    
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("\nWarning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array


def runTest(test_name, device,
            out_slot, out_int, vocab_len, pad_index, 
            hid_size, emb_size, n_layer, dropoutEmb, dropoutOut, bidirectional, combine, layerNorm,
            runs, n_epochs, lr, clip, patience, 
            lang, train_loader, dev_loader, test_loader):
    print("Running test", test_name)

    slot_f1s, intent_acc = [], []
    best_f1_runs = 0
    best_model_runs = None

    for x in tqdm(range(0, runs)):
        model = ModelIAS(hid_size=hid_size, out_slot=out_slot, out_int=out_int, 
                         emb_size=emb_size, vocab_len=vocab_len, n_layer=n_layer, layerNorm=layerNorm,
                         pad_index=pad_index, dropoutEmb=dropoutEmb, dropoutOut=dropoutOut, 
                         bidirectional=bidirectional, combine=combine).to(device)
        model.apply(init_weights)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        best_model = None
        pat = patience

        for e in tqdm(range(1,n_epochs)):
            loss = train_loop(train_loader, optimizer, model, clip=clip, lang=lang)
            if e % 5 == 0:
                sampled_epochs.append(e)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']

                if f1 > best_f1:
                    # print(f"\nNew best F1: {f1}")
                    best_f1 = f1
                    best_model = deepcopy(model)
                    pat = patience
                else:
                    pat -= 1
                if pat <= 0: # Early stopping with patient
                    break # Not nice but it keeps the code clean
        # breakpoint()
        if best_model is None:
            print("No best model found?")
            # breakpoint()
            best_model = model
        results_test, intent_test, _ = eval_loop(test_loader, best_model, lang)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])

        # print(f"Intent Acc: {intent_test['accuracy']:.3f} Slot F1: {results_test['total']['f']:.3f}")

        if results_test['total']['f'] > best_f1_runs:
            best_f1_runs = results_test['total']['f']
            best_model_runs = deepcopy(best_model)
            #show plot
            # plt.plot(sampled_epochs, losses_train, label='Train')
            # plt.plot(sampled_epochs, losses_dev, label='Dev')
            # plt.legend()
            # plt.title(f"Loss {test_name}")
            # plt.show()


    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))

    PATH = os.path.join("models", test_name+".pt")
    saving_object = { 
                  "model": best_model_runs.state_dict(), 
                  "model_params": {
                        "out_slot": out_slot,
                        "out_int": out_int,
                        "vocab_len": vocab_len,
                        "hid_size": hid_size,
                        "emb_size": emb_size,
                        "n_layer": n_layer,
                        "dropoutEmb": dropoutEmb,
                        "dropoutOut": dropoutOut,
                        "bidirectional": bidirectional,
                        "combine": combine,
                        "layerNorm": layerNorm,
                  },
                  "lang": lang,
                  "results": {
                                "slot_f1": slot_f1s.mean(), 
                                "slot_f1_std": slot_f1s.std(),
                                "intent_acc": intent_acc.mean(),
                                "intent_acc_std": intent_acc.std()
                    }
                  }
    torch.save(saving_object, PATH)
