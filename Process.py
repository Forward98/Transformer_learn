# import pandas as pd
# import torchtext
# from torchtext.legacy import data
# from Tokenize import tokenize
# from Batch import MyIterator, batch_size_fn
# import os
# import dill as pickle

# def read_data(opt):
    
#     if opt.src_data is not None:
#         try:
#             opt.src_data = open(opt.src_data).read().strip().split('\n')
#         except:
#             print("error: '" + opt.src_data + "' file not found")
#             quit()
    
#     if opt.trg_data is not None:
#         try:
#             opt.trg_data = open(opt.trg_data).read().strip().split('\n')
#         except:
#             print("error: '" + opt.trg_data + "' file not found")
#             quit()

# def create_fields(opt):
    
#     spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it', 'nl']
#     if opt.src_lang not in spacy_langs:
#         print('invalid src language: ' + opt.src_lang + 'supported languages : ' + str(spacy_langs))
#     if opt.trg_lang not in spacy_langs:
#         print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + str(spacy_langs))
    
#     print("loading spacy tokenizers...")
    
#     t_src = tokenize(opt.src_lang)
#     t_trg = tokenize(opt.trg_lang)

#     TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
#     SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

#     if opt.load_weights is not None:
#         try:
#             print("loading presaved fields...")
#             SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
#             TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
#         except:
#             print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
#             quit()
        
#     return(SRC, TRG)

# def create_dataset(opt, SRC, TRG):

#     print("creating dataset and iterator... ")

#     raw_data = {'src' : [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
#     df = pd.DataFrame(raw_data, columns=["src", "trg"])
    
#     mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
#     df = df.loc[mask]

#     df.to_csv("translate_transformer_temp.csv", index=False)
    
#     data_fields = [('src', SRC), ('trg', TRG)]
#     train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

#     train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
#                         repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
#                         batch_size_fn=batch_size_fn, train=True, shuffle=True)
    
#     os.remove('translate_transformer_temp.csv')

#     if opt.load_weights is None:
#         SRC.build_vocab(train)
#         TRG.build_vocab(train)
#         if opt.checkpoint > 0:
#             try:
#                 os.mkdir("weights")
#             except:
#                 print("weights folder already exists, run program with -load_weights weights to load them")
#                 quit()
#             pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
#             pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

#     opt.src_pad = SRC.vocab.stoi['<pad>']
#     opt.trg_pad = TRG.vocab.stoi['<pad>']

#     opt.train_len = get_len(train_iter)

#     return train_iter

# def get_len(train):

#     for i, b in enumerate(train):
#         pass
    
#     return i

# import pandas as pd
# import torchtext
# from torchtext.datasets import IMDB
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
# from torch.utils.data import Dataset, DataLoader
# from Tokenize import tokenize
# from Batch import MyIterator, batch_size_fn
# import os
# import dill as pickle

# def read_data(opt):
#     # todo: debug看下传入的字符格式
#     if opt.src_data is not None:
#         try:
#             # # 打开源文件，读取内容，去除首尾空白字符，并按行分割成列表
#             # opt.src_data = open(opt.src_data).read().strip().split('\n')
#             with open(opt.src_data, 'r', encoding='utf-8') as f:
#                 opt.src_data = f.read().strip().split('\n')
#                 print("文件读取成功！")
#         except:
#             print("error: '" + opt.src_data + "' file not found")
#             quit()
    
#     if opt.trg_data is not None:
#         try:
#             # opt.trg_data = open(opt.trg_data).read().strip().split('\n')
#             """
#             read_data 函数中直接使用 open(opt.src_data)，但未指定文件模式或关闭文件句柄，
#             可能导致资源泄漏。使用 with 语句确保文件正确关闭
#             """
#             with open(opt.trg_data, 'r', encoding='utf-8') as f:
#                 opt.trg_data = f.read().strip().split('\n')
#                 print("文件读取成功！")            
#         except:
#             print("error: '" + opt.trg_data + "' file not found")
#             quit()

# def create_fields(opt):
#     # 定义支持的语言模型列表
#     spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it', 'nl']
#     # 检查源语言是否受支持
#     if opt.src_lang not in spacy_langs:
#         print('invalid src language: ' + opt.src_lang + 'supported languages : ' + str(spacy_langs))
#     # 检查目标语言是否受支持    
#     if opt.trg_lang not in spacy_langs:
#         print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + str(spacy_langs))
    
#     print("loading spacy tokenizers...")

#     # 加载源语言的分词器
#     t_src = tokenize(opt.src_lang)
#     # 加载目标语言的分词器
#     t_trg = tokenize(opt.trg_lang)

#     # 使用新的TextPipeline和LabelPipeline
#     SRC = torchtext.data.utils.TextPipeline(tokenizer=t_src.tokenizer, vocab=None)
#     TRG = torchtext.data.utils.TextPipeline(tokenizer=t_trg.tokenizer, vocab=None)
#     # todo: debug看下tokenizer后的字符格式
#     if opt.load_weights is not None:
#         try:
#             print("loading presaved fields...")
#             SRC.vocab = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
#             TRG.vocab = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
#         except:
#             print("error opening SRC.pkl and TRG.pkl field files, please ensure they are in " + opt.load_weights + "/")
#             quit()
        
#     return(SRC, TRG)

# def create_dataset(opt, SRC, TRG):

#     print("creating dataset and iterator... ")
#     # 创建一个包含源数据和目标数据的字典
#     raw_data = {'src' : [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
#     df = pd.DataFrame(raw_data, columns=["src", "trg"])
#     # 计算源数据和目标数据中每个句子的长度（以空格分隔的单词数）
#     # df['src'].str.count(' ')：计算源数据中每个句子的空格数量，这可以近似表示句子的长度
#     # df['trg'].str.count(' ')：计算目标数据中每个句子的空格数量
#     # < opt.max_strlen：判断句子长度是否小于最大允许长度
#     mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
#     # 使用掩码过滤数据框，只保留长度符合条件的句子
#     df = df.loc[mask]

#     """
#     假设我们有一个数据框df，其中包含以下数据：
#     import pandas as pd

#     data = {
#         'src': ['This is a short sentence.', 'This is a very long sentence that will be filtered out.', 'Another short sentence.'],
#         'trg': ['Ceci est une courte phrase.', 'Ceci est une très longue phrase qui sera filtrée.', 'Une autre courte phrase.']
#     }
#     df = pd.DataFrame(data)

#     opt.max_strlen = 10
#     mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
#     df = df.loc[mask]

#     过滤后的数据框df将只包含以下行：
#         src	                                trg
#     0	This is a short sentence.	        Ceci est une courte phrase.
#     2	Another short sentence.	        Une autre courte phrase.

#     """
#     # 将处理后的数据框保存为临时CSV文件
#     df.to_csv("translate_transformer_temp.csv", index=False)
    
#     class CustomDataset(Dataset):
#         def __init__(self, data):
#             self.data = data
        
#         def __len__(self):
#             return len(self.data)
        
#         def __getitem__(self, idx):
#             src = self.data.iloc[idx, 0]
#             trg = self.data.iloc[idx, 1]
#             return (src, trg)
    
#     dataset = CustomDataset(df)
    
#     if opt.load_weights is None:
#         # 构建词汇表
#         SRC.vocab = build_vocab_from_iterator([SRC.tokenizer(line) for line in df['src']])
#         TRG.vocab = build_vocab_from_iterator([TRG.tokenizer(line) for line in df['trg']])
#         if opt.checkpoint > 0:
#             try:
#                 os.mkdir("weights")
#             except:
#                 print("weights folder already exists, run program with -load_weights weights to load them")
#                 quit()
#             pickle.dump(SRC.vocab, open('weights/SRC.pkl', 'wb'))
#             pickle.dump(TRG.vocab, open('weights/TRG.pkl', 'wb'))

#     opt.src_pad = SRC.vocab['<pad>']
#     opt.trg_pad = TRG.vocab['<pad>']

#     def collate_batch(batch):
#         src_list, trg_list = [], []
#         for src, trg in batch:
#             src_list.append(SRC.vocab(SRC.tokenizer(src)))
#             trg_list.append(TRG.vocab(TRG.tokenizer(trg)))
#         return (src_list, trg_list)
    
#     train_iter = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, collate_fn=collate_batch)
    
#     os.remove('translate_transformer_temp.csv')

#     opt.train_len = get_len(train_iter)

#     return train_iter

# def get_len(train):

#     for i, b in enumerate(train):
#         pass
    
#     return i


import pandas as pd
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from Tokenize import tokenize
from Batch import MyIterator, batch_size_fn
import os
import dill as pickle
import torch
import functools

def read_data(opt):
    if opt.src_data is not None:
        try:
            with open(opt.src_data, 'r', encoding='utf-8') as f:
                opt.src_data = f.read().strip().split('\n')
                print("文件读取成功！")
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()
    
    if opt.trg_data is not None:
        try:
            with open(opt.trg_data, 'r', encoding='utf-8') as f:
                opt.trg_data = f.read().strip().split('\n')
                print("文件读取成功！")            
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()

def create_fields(opt):
    spacy_langs = ['en_core_web_sm', 'fr_core_news_sm', 'de', 'es', 'pt', 'it', 'nl']
    if opt.src_lang not in spacy_langs:
        print('invalid src language: ' + opt.src_lang + ' supported languages : ' + str(spacy_langs))
    if opt.trg_lang not in spacy_langs:
        print('invalid trg language: ' + opt.trg_lang + ' supported languages : ' + str(spacy_langs))
    
    print("loading spacy tokenizers...")
    t_src = tokenize(opt.src_lang)
    t_trg = tokenize(opt.trg_lang)

    # 使用 torchtext.data.Field 替代 TextPipeline
    SRC = torchtext.data.Field(
        tokenize=t_src.tokenizer,
        lower=True,
        include_lengths=True
    )
    TRG = torchtext.data.Field(
        tokenize=t_trg.tokenizer,
        lower=True,
        include_lengths=True
    )

    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            SRC.vocab = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG.vocab = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TRG.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()
        
    return SRC, TRG

def create_dataset(opt, SRC, TRG):
    print("creating dataset and iterator... ")
    raw_data = {'src' : [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]
    df.to_csv("translate_transformer_temp.csv", index=False)
    
    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            src = self.data.iloc[idx, 0]
            trg = self.data.iloc[idx, 1]
            return (src, trg)
    
    dataset = CustomDataset(df)
    
    if opt.load_weights is None:
        SRC.build_vocab(df['src'])
        TRG.build_vocab(df['trg'])
        if opt.checkpoint > 0:
            try:
                os.mkdir("weights")
            except:
                print("weights folder already exists, run program with -load_weights weights to load them")
                quit()
            pickle.dump(SRC.vocab, open('weights/SRC.pkl', 'wb'))
            pickle.dump(TRG.vocab, open('weights/TRG.pkl', 'wb'))
    
    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    def collate_batch(batch, SRC, TRG):
        src_list, trg_list = [], []
        for src, trg in batch:
            src_tokens = SRC.tokenize(src)
            trg_tokens = TRG.tokenize(trg)
            
            # 将分词后的文本转换为数值
            src_numerical = [SRC.vocab.stoi[token] for token in src_tokens]
            trg_numerical = [TRG.vocab.stoi[token] for token in trg_tokens]
            
            src_list.append(torch.tensor(src_numerical))
            trg_list.append(torch.tensor(trg_numerical))
        
        # 使用 pad_sequence 来处理不同长度的序列
        src_padded = torch.nn.utils.rnn.pad_sequence(src_list, padding_value=SRC.vocab.stoi['<pad>'])
        trg_padded = torch.nn.utils.rnn.pad_sequence(trg_list, padding_value=TRG.vocab.stoi['<pad>'])
        
        return (src_padded, trg_padded)
    
    # 使用 functools.partial 来传递 SRC 和 TRG
    train_iter = DataLoader(dataset, batch_size=opt.batchsize, shuffle=True, collate_fn=functools.partial(collate_batch, SRC=SRC, TRG=TRG))
    
    os.remove('translate_transformer_temp.csv')

    opt.train_len = get_len(train_iter)

    return train_iter

def get_len(train):
    for i, b in enumerate(train):
        pass
    return i

