# =======vid path :  /mnt/e/ydata-tvsum50-v1_1/video
#=======后续版本持续调优参数.
#  使用数值来模拟结果,收敛很慢. 改成多酚类试试. 变成多酚类.

import av
import torch
import numpy as np

from fenlei.transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download
device = torch.device("cuda" if torch.cuda.is_available()else "cpu") 
np.random.seed(0)
processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

#=============做数据集

import numpy as np

datap='video'
import pandas as pd
anno='ydata-tvsum50-anno.tsv'
info='ydata-tvsum50-info.tsv'

import os # ========做本地电脑和服务器的适配.
print(os.uname().nodename)
if os.uname().nodename=='PC-202003302200':
    #============做本地适配
    anno='/mnt/e/ydata-tvsum50-v1_1/ydata-tvsum50-anno.tsv'
    info='/mnt/e/ydata-tvsum50-v1_1/ydata-tvsum50-info.tsv'

    datap='/mnt/e/ydata-tvsum50-v1_1/video'
print()

all_video=[]
all_label=[]
chulitvsum=1
idex=0
info = pd.read_csv(info, sep="\t")
if chulitvsum:
 print()


    


print()

#================ferz


print('开始训练')
epoch=1
from torch import nn, optim
# optimizer = optim.Adam(model.parameters(), lr=0.001)
import torch.optim as optim

for p in model.parameters():
    p.requires_grad=False

for p in model.base_model.last.parameters():
    p.requires_grad=True
for p in model.base_model.last2.parameters():
    p.requires_grad=True
for p in model.base_model.last3.parameters():
    p.requires_grad=True
print()


optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
model=model.to(device)
for i in range(epoch):

#   for out3, out4 in zip(all_video,all_label):
#     #  getdata
  for idex in range(len(info)):
    # idex=0
    
    time=[]
    for i in range(len(info)):
        tmp=info.iloc[i].length
        time.append(int(tmp.split(':')[0])*60+int(tmp.split(':')[1]))

    import glob
    files=glob.glob(datap+'/*.mp4')


    print(1)
    dummy=datap+'/'+info.iloc[idex].video_id+'.mp4'  #取第一个做dummytest
    dummy_shipinchang=time[idex] # diyige shipinchnag .
    dummy_fps=1
    import av
    container = av.open(dummy)
    container.streams.video[0].average_rate
    time=container.streams.video[0].duration/10000
    time=dummy_shipinchang
    frames=container.streams.video[0].frames
    fps=frames/time
    print(1)

    indices=[]
    jiange=int(fps*2/8) # 每8个一组
    for i in range(0,frames,jiange):
        indices.append(i)

    print(1) 
    t=[]
    out=[]
    for i in indices:

        t.append(i)
        if len(t)==8:
                out.append(t)
                t=[]
    # if t:
    #     out.append(t)
    hout=out
    print(2)
    #-------每一组的平分.


    # import numpy as np 
    # df = pd.read_csv(anno, sep="\t")  # 用pd速度快.
    # df.iloc[0]
    with open(anno) as f:
        tmp=f.readlines()

    tmp=[''.join(i.strip().split('\t')[2:]).split(',') for i in tmp]
    tmp2=[]
    for i in range(len(tmp)):
        tmp2.append([int(i) for i in tmp[i]])
    tmp=tmp2 
    all=[]
#     for i in tmp:
#         print(len(i))
    for i in range(0,len(tmp),20):
        
        t=(np.array(tmp[i])+np.array(tmp[i+1])+np.array(tmp[i+2]))/3
#         print()
        all.append(t)
    #========改变out平分.
#     print()
    all2=[]
    frames2 = []
    container.seek(0)
    out=sum(out,[])
    for i, frame in enumerate(container.decode(video=0)):
        if i in out:
            frames2.append(frame)
#     print()
    t=[]
    out2=[]
    for i in frames2:

        t.append(i)
        if len(t)==8:
                out2.append(t)
                t=[]
    # if t:
    #     out2.append(t)
#     print(2)
    out3=[]
    for i in out2:

        out3.append(np.stack([x.to_ndarray(format="rgb24") for x in i]))
    print(len(out3),'numberofkeyframe')
    print()
    fenshu=all[idex]
    hout
    out4=[]
    for i in hout:
        ttt=fenshu[i]
        t=np.mean(ttt) #===============guiyihua
        out4.append(t)


    #-=----to int
    out4=[int(i+0.5)-1 for i in out4]
    print()
    # all_video.append(out3)
    # all_label.append(out4)













    bs=150
    model.train()
    import gc
    gc.collect()
    for i in range(0,len(out3),bs):
        
        tmp1=out3[i:i+bs]
        tmp2=out4[i:i+bs]


        for jj in range(20):
            optimizer.zero_grad()
            
            video=tmp1

            tmp2=torch.tensor(tmp2).to(device)

            # Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
            inputs = processor(
                text=[""],
                videos=list(video),
                return_tensors="pt",
                padding=True,
            )
            for i in inputs:
                inputs[i]=inputs[i].to(device)





            #=============收集所有的数据和标签.

        #     print(inputs,66666666666666666)



            # forward pass
            
            outputs,loss = model(**inputs,return_loss=True,label=tmp2,fenlei=1)


 
            loss.backward()
            print(loss.item(),'当前损失')
            print(optimizer.param_groups[0]["lr"],'当前学习率')
            print('当前学习的视频索引',idex)
            optimizer.step()
#============测试一下加这个.#一般在del 变量后面使用.
            # torch.cuda.empty_cache()



print('over_train')
torch.save(model, 'model.pt')
print('存完.')
# logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
# probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities
# print(probs)


