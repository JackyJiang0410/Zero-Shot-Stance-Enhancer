import argparse
from tqdm import tqdm
import pickle
import faiss
from random import sample
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["WORLD_SIZE"] = "1"
import torch
import math
from torch import optim
import random
import numpy as np
from criterion import TraditionCriterion, Stance_loss, Target_loss
from torch.utils.data import RandomSampler, DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score
from data_utils import Tokenizer4Bert, ZSSDDataset, zeroshotDataset
from transformers import BertModel
from models.bert_spc import BERT_SPC
import pickle
from time import strftime, localtime

from sklearn.metrics import classification_report


# torch.cuda.set_device(gpu_id)
class Instructor(object):
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)
        print("using model: ", opt.model_name)
        print("running dataset: ", opt.dataset)
        print("output_dir: ", opt.output_dir)
        self.trainset = zeroshotDataset(data_dir=self.opt.train_dir, tokenizer=tokenizer, data_type='train',
                                        target=opt.target, opt=opt)
        self.valset = zeroshotDataset(data_dir=self.opt.test_dir, tokenizer=tokenizer, data_type='test',
                                      target=opt.target, opt=opt)
        self.testset = zeroshotDataset(data_dir=self.opt.test_dir, tokenizer=tokenizer, data_type='test',
                                       target=opt.target, opt=opt)

        self.criterion = TraditionCriterion(opt)

        params = ([p for p in self.model.parameters()])

        self.optimizer = self.opt.optim_class(params, lr=self.opt.lr)

    def run_tradition(self):
        best_acc, best_f1 = self.train_traditon()
        state_dict_dir = opt.output_dir + "/state_dict"
        print("\n\nReload the best model with best acc {} from path {}\n\n".format(best_acc, state_dict_dir))
        ckpt = torch.load(os.path.join(state_dict_dir, "best_acc_model.bin"))
        self.model.load_state_dict(ckpt)
        acc, f1, report, f1_ma = self.test_tradition()

        print("\n\nReload the best model with best f1 {} from path {}\n\n".format(best_f1, state_dict_dir))
        ckpt = torch.load(os.path.join(state_dict_dir, "best_f1_model.bin"))
        self.model.load_state_dict(ckpt)
        acc, f1, report, f1_ma = self.test_tradition()

        return acc, f1, report, f1_ma

    def train_traditon(self):
        sampler = RandomSampler(self.trainset)
        train_loader = DataLoader(self.trainset, batch_size=self.opt.batch_size, sampler=sampler)
        train_loader_prototype = DataLoader(self.trainset, batch_size=self.opt.batch_size, sampler=sampler)
        print("Train loader length: {}".format(len(train_loader)))
        optimizer = self.optimizer
        best_acc = 0
        best_f1 = 0
        cnt = 0

        for i_epoch in range(self.opt.epochs):
            print('>' * 20, 'epoch:{}'.format(i_epoch), '<' * 20)
            n_correct, n_total, loss_total = 0, 0, 0
            self.model.train()
            for i_batch, batch in enumerate(train_loader):

                input_features = [batch[feat_name].to(self.opt.device) for feat_name in self.opt.input_features]
                true_stance = batch['polarity']

                if opt.n_gpus > 0:
                    true_stance = true_stance.to(self.opt.device)

                logits = self.model(input_features)
                loss = self.criterion(logits, true_stance)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                n_correct += (torch.argmax(logits, -1) == true_stance).sum().item()
                n_total += len(logits)
                loss_total += loss.item() * len(logits)
                if cnt % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    print("Train step: {} acc:{} loss: {}".format(cnt, train_acc, train_loss))

                if cnt != 0 and cnt % self.opt.eval_steps == 0 and i_epoch > 0:
                    eval_acc, eval_f1 = self.dev_tradition()
                    if eval_acc > best_acc:
                        print('Better ACC! Saving model!')
                        best_acc = eval_acc
                        print("Saving model of best acc: {}".format(best_acc))
                        state_dict_dir = opt.output_dir + "/state_dict"
                        if not os.path.exists(state_dict_dir):
                            os.makedirs(state_dict_dir)
                        torch.save(self.model.state_dict(), os.path.join(state_dict_dir, "best_acc_model.bin"))
                    if eval_f1 > best_f1:
                        print('Better F1! Saving model!')
                        best_f1 = eval_f1
                        print("Saving model of best f1: {}".format(best_f1))
                        state_dict_dir = opt.output_dir + "/state_dict"
                        if not os.path.exists(state_dict_dir):
                            os.makedirs(state_dict_dir)
                        torch.save(self.model.state_dict(), os.path.join(state_dict_dir, "best_f1_model.bin"))
                cnt += 1
        print("Training finished.")
        return best_acc, best_f1

    def dev_tradition(self):
        self.model.eval()
        sampler = RandomSampler(self.valset)
        dev_loader = DataLoader(dataset=self.valset, batch_size=self.opt.eval_batch_size, sampler=sampler)
        all_labels = []
        all_logits = []
        eval_loss = 0
        cnt = 0
        for i_batch, batch in enumerate(dev_loader):
            input_features = [batch[feat_name].to(self.opt.device) for feat_name in self.opt.input_features]
            true_stance = batch['polarity']
            if opt.n_gpus > 0:
                true_stance = true_stance.to(self.opt.device)
            with torch.no_grad():
                # if 'scl' in self.opt.model_name:
                #     logits,_ = self.model(input_features+self.cluster_result)
                #     loss = self.logits_criterion(logits, true_stance)
                # else:
                logits = self.model(input_features)
                loss = self.criterion(logits, true_stance)
            if self.opt.n_gpus > 1:
                loss = loss.mean().item()
            else:
                loss = loss.item()
            eval_loss += loss
            labels = true_stance.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            all_labels.append(labels)
            all_logits.append(logits)
            cnt = cnt + 1
        all_labels = np.concatenate(all_labels, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        preds = all_logits.argmax(axis=1)

        acc = accuracy_score(y_true=all_labels, y_pred=preds)
        f1 = f1_score(all_labels, preds, average='macro')
        f1_ma = f1_score(all_labels, preds, labels=[0, 2], average='macro')
        # print('F1_ma:{}'.format(f1_ma))
        self.model.train()
        return acc, f1_ma

    def test_tradition(self):
        self.model.eval()
        sampler = RandomSampler(self.testset)
        test_loader = DataLoader(dataset=self.testset, batch_size=self.opt.eval_batch_size, sampler=sampler)
        all_labels = []
        all_logits = []
        eval_loss = 0
        cnt = 0
        for i_batch, batch in enumerate(test_loader):
            input_features = [batch[feat_name].to(self.opt.device) for feat_name in self.opt.input_features]
            true_stance = batch['polarity']
            if opt.n_gpus > 0:
                true_stance = true_stance.to(self.opt.device)
            with torch.no_grad():

                logits = self.model(input_features)
                loss = self.criterion(logits, true_stance)
            if self.opt.n_gpus > 1:
                loss = loss.mean().item()
            else:
                loss = loss.item()
            eval_loss += loss
            labels = true_stance.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            all_labels.append(labels)
            all_logits.append(logits)
            cnt = cnt + 1
        all_labels = np.concatenate(all_labels, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        preds = all_logits.argmax(axis=1)
        acc = accuracy_score(y_true=all_labels, y_pred=preds)
        f1 = f1_score(all_labels, preds, average='macro')
        f1_ma = f1_score(all_labels, preds, labels=[0, 2], average='macro')
        report = classification_report(all_labels, preds, target_names=['Con', 'Pro', 'None'], digits=4)
        print(classification_report(all_labels, preds, digits=6))
        print("Test Acc: {} F1_ma:{}".format(acc, f1_ma))
        self.model.train()
        return acc, f1, report, f1_ma

    def save_evaluation_result(self, f1, acc, score_dict, file_name):
        result_path = os.path.join(self.opt.output_dir, file_name)
        with open(result_path, 'w', encoding='utf-8') as out_file:
            out_file.write('Test Acc: {} F1: {}\nReport:\n'.format(acc, f1))
            for k in sorted(score_dict.keys()):
                scores = score_dict[k]
                for meas_name, value in scores.items():
                    out_file.write("{} {}: {}\n".format(k, meas_name, value))
                out_file.write("\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # config
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='bert-scl', type=str, required=False)
    parser.add_argument('--type', default=0, help='for VAST dataset 2 2 for all,0 for zero shot ,1 for few shot',
                        type=str, required=False)
    parser.add_argument('--dataset', default='covid', type=str, required=False)
    parser.add_argument('--target', default='stay at home orders',
                        help='Feminist_Movement,Hillary_Clinton,Legalization_of_Abortion', type=str, required=False)
    # for covid "stay at home orders","face masks","school closures","fauci"

    parser.add_argument('--output_par_dir', default='test_outputs', type=str)
    parser.add_argument('--polarities', default='sem16_naacl', nargs='+',
                        help="if just two polarity switch to ['positive', 'negtive']", required=False)
    parser.add_argument('--optimizer', default='adam', type=str, required=False)
    parser.add_argument('--temperature', default=0.07, type=float, required=False)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, required=False)
    parser.add_argument('--lr', default=3e-5, type=float, help='try 5e-5, 2e-5, 1e-3 for others', required=False)
    parser.add_argument('--dropout', default=0.1, type=float, required=False)
    parser.add_argument('--l2reg', default=1e-5, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False)
    parser.add_argument('--log_path', default="./log", type=str, required=False)
    parser.add_argument('--embed_dim', default=300, type=int, required=False)
    parser.add_argument('--hidden_dim', default=128, type=int, required=False, help="lstm encoder hidden size")
    parser.add_argument('--feature_dim', default=2 * 128, type=int, required=False,
                        help="feature dim after encoder depends on encoder")
    parser.add_argument('--output_dim', default=64, type=int, required=False)
    parser.add_argument('--relation_dim', default=100, type=int, required=False)
    parser.add_argument('--bert_dim', default=768, type=int, required=False)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str, required=False)
    parser.add_argument('--max_seq_len', default=512, type=int, required=False)
    parser.add_argument('--stance_loss_weight', default=1, type=float, required=False)
    parser.add_argument('--prototype_loss_weight', default=0.2, type=float, required=False)
    parser.add_argument('--alpha', default=0.8, type=float, required=False)
    parser.add_argument('--beta', default=1.2, type=float, required=False)

    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0', required=False)
    parser.add_argument('--seed', default=0, type=int, help='set seed for reproducibility')
    parser.add_argument("--batch_size", default=8, type=int, required=False)
    parser.add_argument("--eval_batch_size", default=8, type=int, required=False)
    parser.add_argument("--epochs", default=10, type=int, required=False)
    parser.add_argument("--eval_steps", default=50, type=int, required=False)

    # graph para
    parser.add_argument('--gnn_dims', default='192,192', type=str, required=False)
    parser.add_argument('--att_heads', default='4,4', type=str, required=False)
    parser.add_argument('--dp', default=0.1, type=float)

    opt = parser.parse_args()

    if opt.seed:
        set_seed(opt.seed)
    model_classes = {

        'bert-scl': BERT_SPC,
    }
    input_features = {
        'bert-scl': ['concat_bert_indices', 'concat_segments_indices'],
        'bert-scl-prototype-graph': ['concat_bert_indices', 'concat_segments_indices'],

    }

    dataset_files = {
        'semeval16': {
            'train': './datasets/semeval16_b/semeval16_background.csv',
            'test': './datasets/semeval16_b/semeval16_background.csv',
        },
        'covid': {
            'train': './datasets/COVID/covid19_background.csv',
            'test': './datasets/COVID/covid19_background.csv',
        }
    }

    polarities = {
        'semeval16': [0, 1, 2],  # 0 for Against, 1 for Favor, 2 for none
        'covid': [0, 1, 2],  # 0 for Against, 1 for Favor, 2 for none

    }
    optimizers = {
        'adam': optim.Adam,
    }

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.n_gpus = torch.cuda.device_count()
    opt.polarities = polarities[opt.dataset]
    opt.num_labels = len(opt.polarities)
    opt.model_class = model_classes[opt.model_name]
    opt.optim_class = optimizers[opt.optimizer]
    opt.input_features = input_features[opt.model_name]
    opt.dataset_files = dataset_files[opt.dataset]
    opt.train_dir = opt.dataset_files['train']
    opt.dev_dir = opt.dataset_files['test']
    opt.test_dir = opt.dataset_files['test']
    opt.output_dir = os.path.join(opt.output_par_dir, opt.model_name, opt.dataset,
                                  strftime("%Y-%m-%d %H-%M-%S", localtime()))  ##get output directory to save results
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    writer = SummaryWriter(opt.log_path)
    print(opt)
    ins = Instructor(opt)
    acc, f1, report, f1_ma = ins.run_tradition()
    print("Test dataset: {}".format(opt.dataset))
    print("Model: {}".format(opt.model_name))
    print('#' * 20, 'Final results:', '#' * 20)
    print('F1 avg (Con & Pro): {}'.format(f1_ma))
    print('acc: {}, f1 :{}'.format(acc, f1))
    print('classification report: \n{}'.format(report))
    writer.close()
