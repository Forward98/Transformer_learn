import argparse
import time
import torch
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import dill as pickle
import os

def train_model(model, opt):
    
    print("training model...")
    model.train()
    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()
                 
    for epoch in range(opt.epochs):

        total_loss = 0
        if opt.floyd is False:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        
        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')
                    
        for i, batch in enumerate(opt.train): 

            # src = batch.src.transpose(0, 1).to(opt.device)
            # trg = batch.trg.transpose(0, 1).to(opt.device)
            # src = torch.tensor(batch[0]).transpose(0, 1).to(opt.device)
            # trg = torch.tensor(batch[1]).transpose(0, 1).to(opt.device)
            src = batch[0].transpose(0, 1).to(opt.device)
            trg = batch[1].transpose(0, 1).to(opt.device)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, opt)
            src_mask.to(opt.device)
            trg_mask.to(opt.device)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True: 
                opt.sched.step()
            
            total_loss += loss.item()
            
            if (i + 1) % opt.printevery == 0:
                 p = int(100 * (i + 1) / opt.train_len)
                 avg_loss = total_loss/opt.printevery
                 if opt.floyd is False:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss), end='\r')
                 else:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss))
                 total_loss = 0
            
            if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()
   
   
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss))

def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-src_data', required=True)
    # parser.add_argument('-trg_data', required=True)
    # parser.add_argument('-src_lang', required=True)
    # parser.add_argument('-trg_lang', required=True)
    # parser.add_argument('-no_cuda', action='store_true')
    # parser.add_argument('-SGDR', action='store_true')
    # parser.add_argument('-epochs', type=int, default=2)
    # parser.add_argument('-d_model', type=int, default=512)
    # parser.add_argument('-n_layers', type=int, default=6)
    # parser.add_argument('-heads', type=int, default=8)
    # parser.add_argument('-dropout', type=int, default=0.1)
    # parser.add_argument('-batchsize', type=int, default=1500)
    # parser.add_argument('-printevery', type=int, default=100)
    # parser.add_argument('-lr', type=int, default=0.0001)
    # parser.add_argument('-load_weights')
    # parser.add_argument('-create_valset', action='store_true')
    # parser.add_argument('-max_strlen', type=int, default=80)
    # parser.add_argument('-floyd', action='store_true')
    # parser.add_argument('-checkpoint', type=int, default=0)

    # opt = parser.parse_args()

    # 指定参数值
    opt = type('Options', (object,), {})()  # 创建一个空对象来存储参数

    # 数据相关参数
    base_dir = r'E:\transformer_learn\Transformer_learn\data'
    opt.src_data = os.path.join(base_dir, 'english.txt')
    opt.trg_data = os.path.join(base_dir, 'french.txt')

    print("完整路径:", opt.src_data)
    print("文件是否存在:", os.path.exists(opt.src_data))

    opt.src_lang = 'en_core_web_sm'    # 源语言
    opt.trg_lang = 'fr_core_news_sm'   # 目标语言

    # 模型相关参数
    opt.d_model = 512                  # 嵌入维度
    opt.n_layers = 6                   # Transformer层数
    opt.heads = 8                      # 多头注意力头数
    opt.dropout = 0.1                  # Dropout概率

    # 训练相关参数
    opt.epochs = 2                     # 训练周期数
    opt.batchsize = 150               # 批量大小（以token数为单位）
    opt.printevery = 100               # 每隔多少次迭代打印一次损失
    opt.lr = 0.0001                    # 学习率
    opt.no_cuda = False                # 是否禁用CUDA
    opt.SGDR = False                   # 是否使用SGDR优化器，用途是周期性地重启学习率来加速收敛
    opt.load_weights = None            # 是否加载预训练权重
    opt.create_valset = False          # 是否创建验证集
    opt.max_strlen = 80                # 最大句子长度
    opt.floyd = False                  # 是否在FloydHub上运行
    opt.checkpoint = 0                 # 是否定期保存模型权重    

    opt.device = 'cuda' if opt.no_cuda is False else 'cpu'
    if opt.device == 'cuda':
        assert torch.cuda.is_available()

    read_data(opt)
    SRC, TRG = create_fields(opt)
    opt.train = create_dataset(opt, SRC, TRG)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))
    
    if opt.load_weights is not None and opt.floyd is not None:
        os.mkdir('weights')
        pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    
    train_model(model, opt)

    if opt.floyd is False:
        promptNextAction(model, opt, SRC, TRG)

def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt, SRC, TRG):

    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    
    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'

    while True:
        save = yesno(input('training complete, save results? [y/n] : '))
        if save == 'y':
            while True:
                if saved_once != 0:
                    res = yesno("save to same folder? [y/n] : ")
                    if res == 'y':
                        break
                dst = input('enter folder name to create for weights (no spaces) : ')
                if ' ' in dst or len(dst) < 1 or len(dst) > 30:
                    dst = input("name must not contain spaces and be between 1 and 30 characters length, enter again : ")
                else:
                    try:
                        os.mkdir(dst)
                    except:
                        res= yesno(input(dst + " already exists, use anyway? [y/n] : "))
                        if res == 'n':
                            continue
                    break
            
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights')
            if saved_once == 0:
                pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
                pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
                saved_once = 1
            
            print("weights and field pickles saved to " + dst)

        res = yesno(input("train for more epochs? [y/n] : "))
        if res == 'y':
            while True:
                epochs = input("type number of epochs to train for : ")
                try:
                    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
            opt.epochs = epochs
            train_model(model, opt)
        else:
            print("exiting program...")
            break

    # for asking about further training use while true loop, and return
if __name__ == "__main__":
    main()
