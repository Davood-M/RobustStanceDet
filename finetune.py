import sys
import argparse
from transformers import BertForSequenceClassification, AdamW, BertPreTrainedModel, BertConfig, BertModel
from transformers import get_linear_schedule_with_warmup
import datetime
import random
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from dataloader import data_train, data_test, CL_data_train
import random
import transformers
from losses import SupConLoss
transformers.logging.set_verbosity_error()

pd.set_option("display.max_columns", None)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

configuration = BertConfig()


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", help="train or test or train2")
parser.add_argument("--dataset", type=str, default="fm+fa+sc+sh/train_100.csv")

parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--num_pair", type=int, default=50)
parser.add_argument("--num_label", type=int, default=2)

args = parser.parse_args()

mode = args.mode
src_data = args.dataset
lr = args.lr
batch_size = args.batch_size
total_epoch = args.epochs
alpha = args.alpha
beta = args.beta
num_cl = args.num_pair
num_label = args.num_label

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



class CustomBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config=configuration):
        super(CustomBertForSequenceClassification, self).__init__(config)
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_label, # 2 for covid-related data, 3 for semeval
            output_attentions=True,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

        self.model.to(device)


    def CL_train(self, train_path, output_dir):
        model = self.model
        train_dataloader, validation_dataloader = CL_data_train(train_path, output_dir, batch_size)

        optimizer = AdamW(self.model.parameters(),
                          lr=lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )

        epochs = total_epoch

        total_steps = len(train_dataloader) * epochs

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        training_stats = []
        total_t0 = time.time()
        all_losses = []

        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            model.train()  # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch

            calc_CL_steps = random.choices(np.arange(0, len(train_dataloader)), k=num_cl)
            for step, batch in enumerate(train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                a_input_ids = batch[0].to(device)
                a_input_mask = batch[1].to(device)
                a_labels = batch[2].to(device)
                p_input_ids = batch[3].to(device)
                p_input_mask = batch[4].to(device)
                p_labels = batch[5].to(device)
                n_input_ids = batch[6].to(device)
                n_input_mask = batch[7].to(device)
                n_labels = batch[8].to(device)

                model.zero_grad()

                all_ids = torch.cat((a_input_ids, p_input_ids, n_input_ids), 0)
                all_masks = torch.cat((a_input_mask, p_input_mask, n_input_mask), 0)
                all_labels = torch.cat((a_labels, p_labels, n_labels), 0)

                outputs = model(all_ids,
                                token_type_ids=None,
                                attention_mask=all_masks,
                                labels=all_labels)

                loss, logits = outputs[:2]
                
                
                ################
                ### ContLoss ###
                ################
                loss_func = SupConLoss(temperature=0.1)
                i = 0
                if (i < num_cl) & (step in calc_CL_steps):          
                    emb_model = BertModel.from_pretrained('bert-base-uncased',
                                    output_hidden_states = True, # Whether the model returns all hidden-states.
                                    )
                    emb_model.to(device)
                    emb_model.eval()
                    with torch.no_grad():
                        a_outputs = emb_model(a_input_ids, a_input_mask)
                        p_outputs = emb_model(p_input_ids, p_input_mask)
                        n_outputs = emb_model(n_input_ids, n_input_mask)

                    a_sentence_emb = a_outputs[0][0]
                    p_sentence_emb = p_outputs[0][0]
                    n_sentence_emb = n_outputs[0][0]
                    total_closs = loss_func(
                        torch.cat([a_sentence_emb.unsqueeze(1), p_sentence_emb.unsqueeze(1)], dim=1),
                        torch.zeros((a_sentence_emb.shape[0], 1)).to(device)
                    )
                    total_closs += loss_func(
                        torch.cat([a_sentence_emb.unsqueeze(1), n_sentence_emb.unsqueeze(1)], dim=1),
                        torch.ones((a_sentence_emb.shape[0], 1)).to(device)
                    )
                    

                #     cos = nn.CosineSimilarity(dim=1)
                #     pos_pair_closs = 1 - cos(a_sentence_emb, p_sentence_emb)
                #     neg_pair_closs = torch.clamp(cos(a_sentence_emb, n_sentence_emb), min=0)
                    
                #     i += 1
 
                # total_closs = pos_pair_closs + neg_pair_closs
                
                # total_train_loss += loss.item() + total_closs
                
                ################
                ### ContLoss ###
                ################


                
                # total_train_loss += loss.item() 
                # loss.backward()
                losses = alpha * torch.mean(total_closs) + (1 - alpha) * loss
                losses.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                all_losses.append([
                    losses.item(),
                    loss.item(),
                    torch.mean(total_closs).item()
                ])

            
            avg_train_loss = total_train_loss / len(train_dataloader)

            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {}".format(avg_train_loss))
            print("  Training epoch took: {}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode
            model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                with torch.no_grad():
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    outputs = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)

                loss, logits = outputs[:2]
                total_eval_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                total_eval_accuracy += flat_accuracy(logits, label_ids)

            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            avg_val_loss = total_eval_loss / len(validation_dataloader)

            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")
        
        loss_df = pd.DataFrame({
            'total_loss': [i[0] for i in all_losses],
            'loss_1': [i[1] for i in all_losses],
            'loss_2': [i[2] for i in all_losses],
        })
        loss_df.to_csv('./losses.csv')

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

        pd.set_option('display.precision', 4)
        df_stats = pd.DataFrame(data=training_stats)
        df_stats = df_stats.set_index('epoch')
        print(df_stats)

        print("Saving model to %s" % output_dir)
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        # torch.save(args, os.path.join(output_dir, 'training_args.bin'))






    # def train(self, train_path, output_dir):
    #     model = self.model
    #     train_dataloader, validation_dataloader = data_train(train_path, output_dir)

    #     optimizer = AdamW(self.model.parameters(),
    #                       lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
    #                       eps=1e-8  # args.adam_epsilon  - default is 1e-8.
    #                       )

    #     epochs = 2

    #     total_steps = len(train_dataloader) * epochs

    #     scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                                 num_warmup_steps=0,  # Default value in run_glue.py
    #                                                 num_training_steps=total_steps)

    #     seed_val = 42

    #     random.seed(seed_val)
    #     np.random.seed(seed_val)
    #     torch.manual_seed(seed_val)
    #     torch.cuda.manual_seed_all(seed_val)

    #     training_stats = []
    #     total_t0 = time.time()

    #     for epoch_i in range(0, epochs):

    #         # ========================================
    #         #               Training
    #         # ========================================

    #         print("")
    #         print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    #         print('Training...')

    #         t0 = time.time()

    #         # Reset the total loss for this epoch.
    #         total_train_loss = 0

    #         model.train()  # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch

            
    #         for step, batch in enumerate(train_dataloader):
    #             if step % 40 == 0 and not step == 0:
    #                 elapsed = format_time(time.time() - t0)
    #                 print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

    #             b_input_ids = batch[0].to(device)
    #             b_input_mask = batch[1].to(device)
    #             b_labels = batch[2].to(device)

    #             model.zero_grad()

    #             outputs = model(b_input_ids,
    #                             token_type_ids=None,
    #                             attention_mask=b_input_mask,
    #                             labels=b_labels)


    #             loss, logits = outputs[:2]
    #             total_train_loss += loss.item() 
    #             loss.backward()

    #             # Clip the norm of the gradients to 1.0.
    #             # This is to help prevent the "exploding gradients" problem.
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    #             optimizer.step()
    #             scheduler.step()

    #         avg_train_loss = total_train_loss / len(train_dataloader)

    #         training_time = format_time(time.time() - t0)

    #         print("")
    #         print("  Average training loss: {}".format(avg_train_loss))
    #         print("  Training epoch took: {}".format(training_time))

    #         # ========================================
    #         #               Validation
    #         # ========================================
    #         # After the completion of each training epoch, measure our performance on
    #         # our validation set.

    #         print("")
    #         print("Running Validation...")

    #         t0 = time.time()

    #         # Put the model in evaluation mode
    #         model.eval()

    #         # Tracking variables
    #         total_eval_accuracy = 0
    #         total_eval_loss = 0
    #         nb_eval_steps = 0

    #         for batch in validation_dataloader:
    #             b_input_ids = batch[0].to(device)
    #             b_input_mask = batch[1].to(device)
    #             b_labels = batch[2].to(device)

    #             with torch.no_grad():
    #                 # token_type_ids is the same as the "segment ids", which
    #                 # differentiates sentence 1 and 2 in 2-sentence tasks.
    #                 outputs = model(b_input_ids,
    #                                 token_type_ids=None,
    #                                 attention_mask=b_input_mask,
    #                                 labels=b_labels)

    #             loss, logits = outputs[:2]
    #             total_eval_loss += loss.item()

    #             logits = logits.detach().cpu().numpy()
    #             label_ids = b_labels.to('cpu').numpy()

    #             total_eval_accuracy += flat_accuracy(logits, label_ids)

    #         avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    #         print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    #         avg_val_loss = total_eval_loss / len(validation_dataloader)

    #         validation_time = format_time(time.time() - t0)

    #         print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    #         print("  Validation took: {:}".format(validation_time))

    #         training_stats.append(
    #             {
    #                 'epoch': epoch_i + 1,
    #                 'Training Loss': avg_train_loss,
    #                 'Valid. Loss': avg_val_loss,
    #                 'Valid. Accur.': avg_val_accuracy,
    #                 'Training Time': training_time,
    #                 'Validation Time': validation_time
    #             }
    #         )

    #     print("")
    #     print("Training complete!")

    #     print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    #     pd.set_option('display.precision', 4)
    #     df_stats = pd.DataFrame(data=training_stats)
    #     df_stats = df_stats.set_index('epoch')
    #     print(df_stats)

    #     print("Saving model to %s" % output_dir)
    #     model_to_save = model.module if hasattr(model,
    #                                             'module') else model  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(output_dir)
    #     # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        
        


    def predict(self, test_path, model_path, conf_option=False ):
        trained_model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_label, # 2 for covid-related data, 3 for semeval
            output_attentions=True,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        trained_model.to(device)
        if "csv" in test_path:
            test_df = pd.read_csv(test_path)
            test_dataloader = data_test(test_df, model_path)

        else:
            test_dataloader = test_path  # directly load dataloader

        # ========================================
        #               Test
        # ========================================
        # Measure our performance on our 1,000 test set.

        # print("")
        # print("Running Test...")

        t0 = time.time()

        # Put the model in evaluation mode
        trained_model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0

        all_label_ids = []
        all_logits = []

        for batch in test_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                outputs = trained_model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        labels=b_labels)

            loss, logits = outputs[:2]
            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            all_label_ids.extend(label_ids.flatten().tolist())
            all_logits.extend(np.argmax(logits, axis=1).flatten().tolist())

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
        # print(f"  Test Accuracy: {avg_val_accuracy:.4f}")

        avg_val_loss = total_eval_loss / len(test_dataloader)

        pred_flat = all_logits
        all_label_ids = np.array(all_label_ids)
        fpr, tpr, thresholds = roc_curve(all_label_ids, pred_flat, pos_label=1)

        test_time = format_time(time.time() - t0)

        # print(f"  Test Loss: {avg_val_loss:.4f}")
        # print(f"  Test F1-score: {f1_score(all_label_ids, pred_flat, average='macro'):.4f}")
        # print(f"  Test AUC: {auc(fpr, tpr)}")
        # print(f"  Test took: {test_time:}")
        # print(f"Total test took {test_time:} (h:mm:ss)")

        if conf_option == True:
            cm = confusion_matrix(all_label_ids, pred_flat)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.show()


        print("\n")
        return avg_val_accuracy, f1_score(all_label_ids, pred_flat, average='macro'), auc(fpr, tpr)


model = CustomBertForSequenceClassification()
data_dir = './MULTI-TARGET-CL/data/' + src_data
model_dir = './MULTI-TARGET-CL/' + src_data



def test(model_dir, topic):
    n = 5
    acc = f1 = auc = 0
    if topic == "covid":
        test_set = {'./MULTI-TARGET-CL/data/test_o2o.csv':"o2o", 
                    './MULTI-TARGET-CL/data/test_liv.csv':"liv", 
                    './MULTI-TARGET-CL/data/fm+fa+sc+sh/test_face masks.csv':"fm", 
                    './MULTI-TARGET-CL/data/fm+fa+sc+sh/test_Fauci.csv':"fa",
                    './MULTI-TARGET-CL/data/fm+fa+sc+sh/test_school closures.csv':"sc",
                    './MULTI-TARGET-CL/data/fm+fa+sc+sh/test_stay at home orders.csv':"sh"}
    elif topic == "trump":
        test_set = {'./MULTI-TARGET-CL/data/Semeval2016t6/test_Hillary Clinton.csv':"hilary", 
                    './MULTI-TARGET-CL/data/Semeval2016t6/test_Legalization of Abortion.csv':"leg"}
    
    else:
        print("Please select the domain")
        
    for i in test_set.keys():
        for j in range(n):
            acc1, f1, auc1 = model.predict(i, model_dir)
        
            acc += acc1
            f1 += f1
            auc += auc1
        
        acc = acc / n
        f1 = f1 / n
        auc = auc / n
    
        print(f"===== test_{test_set[i]} =====")
        print(f"AVG ACC : {acc:.3f}\nAVG F1 : {f1:.3f}\nAVG AUC : {auc:.3f}\n")
    


# # Training
if mode == 'train':
    # model.train('./MULTI-TARGET-CL/data/fm+fa/train30.csv', './MULTI-TARGET-CL/fm+fa/saved_model30/')
    model_name = model_dir.replace('.csv', '/')
    model.CL_train(data_dir, model_name)
    test(model_name, "covid")
    # model.CL_train('./MULTI-TARGET-CL/data/Semeval2016t6/train_100.csv', './MULTI-TARGET-CL/Semeval2016t6/saved_model100/')

elif mode == 'train2':
    model.train('./MULTI-TARGET-CL/data/Semeval2016t6/train_100.csv', './MULTI-TARGET-CL/Semeval2016t6/no_cl_saved_model100/')

    # model.train('./MULTI-TARGET-CL/data/face+fauci/train30.csv', './MULTI-TARGET-CL/face+fauci_saved_model30/')
    # model.train('./MULTI-TARGET-CL/data/fm+fa+sc+sh/train_100.csv', './MULTI-TARGET-CL/fm+fa+sc+sh/saved_model100/')


    # model.train('/content/drive/Shareddrives/Docogen/docogen/exp_100/train_cfs00.csv','/content/drive/Shareddrives/Docogen/docogen/exp_100/saved_model_cf00/' )
    # model.train('/content/drive/Shareddrives/Docogen/docogen/exp_100/train_cfs15.csv','/content/drive/Shareddrives/Docogen/docogen/exp_100/saved_model_cf15/' )
    # model.train('/content/drive/Shareddrives/Docogen/docogen/exp_100/train_cfs30.csv','/content/drive/Shareddrives/Docogen/docogen/exp_100/saved_model_cf30/' )
    #
    # model.train('./exp_30_epoch_10/combined00.csv','./exp_30_epoch_10/stance/saved_model_com00/' )
    # model.train('./exp_30_epoch_10/combined15.csv','./exp_30_epoch_10/stance/saved_model_com15/' )
    # model.train('./exp_30_epoch_10/combined30.csv','./exp_30_epoch_10/stance/saved_model_com30/' )
    pass

#  Test
elif mode == 'test':
    model_name = model_dir.replace('.csv', '/')
    test(model_name, "covid")
    

        
