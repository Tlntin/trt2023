## 队伍名: 无声优化者（着）

## 大赛介绍 | [链接](https://tianchi.aliyun.com/competition/entrance/532108/information)
TensorRT 作为 NVIDIA 英伟达 GPU 上的 AI 推理加速库，在业界得到了广泛应用与部署。与此同时，TensorRT 开发团队也在持续提高产品的好用性：一方面让更多模型能顺利通过 ONNX 自动解析得到加速，另一方面对常见模型结构（如 MHA）的计算进行深度优化。这使得大部分模型不用经过手工优化，就能在 TensorRT 上跑起来，而且性能优秀。

过去的一年，是生成式 AI（或称“AI生成内容”） 井喷的一年。大量的图像和文本被计算机批量生产出来，有的甚至能媲美专业创作者的画工与文采。可以期待，未来会有更多的生成式AI模型大放异彩。在本届比赛中，我们选择生成式AI模型作为本次大赛的主题。

今年的 TensorRT Hackathon 是本系列的第三届比赛。跟往届一样，我们希望借助比赛的形式，提高选手开发 TensorRT 应用的能力，因此重视选手的学习过程以及选手与 NVIDIA 英伟达专家之间的沟通交流。我们期待选手们经过这场比赛，在 TensorRT 编程相关的知识和技能上有所收获。

## 准备工作
1. 安装nvidia-docker
2. 拉取docker镜像
```bash
docker pull registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2
```
3. 首次运行docker项目，方便拷贝代码到本地
```bash
docker run --gpus all \
  --name trt2023 \
  -it --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2
```
4. 另开一个终端，拷贝docker中的文件到本地（建议建一个仓库）
```bash
docker cp trt2023:/home/player/ControlNet .
mv ControlNet/* .
rm ControlNet -r
```

## 项目部署
1. 从网盘下载.pth模型，百度网盘[链接](https://pan.baidu.com/s/1FVk1wYBX32gosUxopEdBbw?pwd=uxmx) 。也可以参考准备工作的3、4步，提取models下面的.pth文件，然后放到`models`目录下。 
2. 正式运行docker, 顺便将本地代码目录覆盖docker中的代码（也更加方便文件拷贝）
- 对于使用本地GPU的用户
```bash
# 运行容器
docker run --gpus all \
  --name trt2023 \
  -u root \
  -d \
  --ipc=host \
  --ulimit memlock=-1 \
  --restart=always \
  --ulimit stack=67108864 \
  -v ${PWD}:/home/player/ControlNet/ \
  registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2 sleep 8640000

# 在Vscode中进行远程开发
# 打开Vscode, 选择最左侧的远程资源管理器，选择`开发容器`，然后选择`registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2(trt2023)`这个容器，选择在当前窗口附加即可。
# 然后选择`文件`，`打开文件夹`， 最后打开`/home/player/ControlNet/`目录即可
```
- 对于远程GPU的用户，可以通过暴露ssh端口（22）来实现远程操作（16785可以换成其他任意端口）。
```bash
# 运行docker容器
docker run --gpus all \
  --name trt2023 \
  -it \
  -p 16785:22 \
  -u root \
  --ipc=host \
  --ulimit memlock=-1 \
  --restart=always \
  --ulimit stack=67108864 \
  -v ${PWD}:/home/player/ControlNet/ \
  registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2 sleep 8640000

# 安装openssh
apt install openssh-server

# 启动openssh
service ssh start

# 创建密钥目录
mkdir ~/.ssh

# 将服务器上面的已授权ssh文件拷贝进docker，这样能连服务器的主机，自然能连docker的主机
docker cp ~/.ssh/authorized_keys  trt2023:/root/.ssh/

# 修改配置文件
vim /etc/ssh/sshd_config

将
#StrictModes yes
设置为
StrictModes no

将
#AuthorizedKeysFile .ssh/authorized_keys
设置为
AuthorizedKeysFile .ssh/authorized_keys

- 注释掉 PermitRootLogin prohibit-password 这一行 添加这一行 PermitRootLogin yes 

# 这样其他主机就可以通过ssh服务连上docker进行开发了。

# 也可以选择设置容器里面的root秘密来连接ssh
passwd

# 观察容器启动文件
docker inspect trt2023 | sed ":a;N;s/\n//g;ta" | grep -Poz "Entrypoint.*?]"

# 输出结果
Entrypoint": [ "/opt/nvidia/nvidia_entrypoint.sh"  ]

# 设置docker容器启动时，自动打开ssh服务
echo "service ssh start" >> /opt/nvidia/nvidia_entrypoint.sh 


# 在Vscode中进行远程开发
# 打开Vscode, 选择最左侧的远程资源管理器，选择`远程隧道（SSH)`，然后点击`+`，输入远程命令`ssh root@[容器所在输入机的ip] -p [刚刚自定义的映射端口]`，然后右键该服务，选择`在当前窗口中连接`即可。
# 然后选择`文件`，`打开文件夹`， 最后打开`/home/player/ControlNet/`目录即可
```
3. 运行测评代码，用于生成图片
```bash
python3 compute_score.py
```
4. 在容器中运行代码发现不用root情况下，容器没有写入项目的权限（造成的原因是因为宿主机和容器用户不一样，我的宿主机是普通用户），解决办法。
```bash
# 在宿主机（服务器）执行下面的代码获取当前用户id以及组用户id
id

# 我获得的结果是:用户id=1000(用户名），组id=1000

# 先输入下面的命令进入容器
docker exec -it trt2023 /bin/bash

# 获取当前路径
pwd

# 输出: `/home/player`， 说明用户是player
# 然后再docker中再输入下面的命令，将docker用户的id改成和宿主机一样的，再切换到docker用户就行了。
usermod -u 1000 player
# 创建新组
groupadd player
# 授权
usermod -g player player
# 改权限
chown -R player:player /home/player
# 改组id
groupmod -g 1000 player

# 切换到player用户
su player

# 在docker容器输入id看看
id

# 结果如下：
uid=1000(player) gid=1000(player) groups=1000(player),27(sudo)

# 可以看到player的组id已经改成了1000了

# 以后可以使用player用户权限进入容器
docker exec -u player -it trt2023 /bin/bash

# 对于vscode远程开发容器的用户
# 打开左侧栏`远程资源管理器`，选择`开发容器`，再第二个框框中，点击小齿轮图标，打开容器配置文件，然后再末尾加上一行"remoteUser": "player"，这样vscode就会自动用player权限去运行。1
# 我的改完后大概是这个效果
{
	"workspaceFolder": "/home/player/ControlNet",
	"extensions": [
		"MS-CEINTL.vscode-language-pack-zh-hans",
		"ms-python.python",
		"ms-python.vscode-pylance"
	],
	"remoteUser": "player"
}
# 关闭重开一下vscode,再启一个终端，你就能看到效果了。
# 此时终端长这样，可以看到是player用户
# player@335a6e153173:~/ControlNet$ 
```


### 其他

  

##### 本地运行代码

- 模型转onnx, onnx转trt, 检查精度，可以写到preprocess.sh

```bash

chomod  +x  preprocess.sh && ./preprocess.sh

```

  

- pytorch版模型生成图片。建议把项目提供的canny2image_TRT.py这个文件拷贝一份，重命名为`canny2image_torch.py`。然后将`compute_score.py`复制一份为`compute_score_old.py`，将第7行的`from canny2image_TRT import hackathon`换成`from canny2image_torch import hackathon`，然后将55行左右的`"bird_"+ str(i) + ".jpg"`改成`"bird_old_"+ str(i) + ".jpg"`，这样后续可以将TensorRT生成的图片和pytorch生成的图片做对比，计算得分。

```bash

python  compute_score_old.py

```

  

- TensorRT版模型生成图片，并计算PD_score。将`compute_score.py`复制一份为`compute_score_new.py`，然后将pytorch生成的图片路径`"bird_old_"+ str(i) + ".jpg"`和TensorRT生成的图片路径`"bird_"+ str(i) + ".jpg"`两个做对比。取消注释`score = PD(base_path, new_path)`，这样就可以得到PDscore分数了。然后用两个列表收集每张图片计算得到的pd_score和time_cost，计算平均分和最大分，最后打印即可。**注意: PDscore大于12是没有分数的，建议本地测试的时候控制PDScore在10以内，因为在线测评可能比本地高一些。**

```bash

python  compute_score_new.py

```

  

- TensorRT版模型生成图片，不计算PD_score(这是原版)，一般你就运行一下，确保没bug就行。

```bash

python  compute_score.py

```

  

##### docker容器本地模拟测试

- 主要测试自己的代码能在提交的时候可以运行

- 先拉一下最新容器，因为镜像可能有做更新

```bash

docker  pull  registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2

```

  

- 下载你的代码到/tmp/repo目录(其他目录也行，放/tmp方便开机后自动清空)。注意：这里将这个目录设置为777权限了，这里解释一下。linux分三个权限，组权限，用户权限，以及宾客权限。相当于你家人，你本人，路人。对你的本机来说，docker容器的player用户就是路人，路上是没有写入权限的，相当于你家的锁，钥匙只有你和你家人有。解决办法就是对你要执行的目录执行授权就行。你可以直接通过chmod 777路径给与权限，意思就是给你家人，你，路人都能写入、执行、读取这个目录，这样就没有测评容器无法在本地目录写入的问题了。

```bash

cd  /tmp

git  clone  xxxx/xxxx.git

mv  xxxxx  /tmp/repo

chmod  777  /tmp/repo

```

  

- 运行预处理，生成onnx和engine

```bash

docker  run  --rm  -t  --network  none  --gpus  '0'  --name  hackathon  -v  /tmp/repo/:/repo  registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2  bash  -c  "cd /repo && bash preprocess.sh"

```

- 跑一下pytorch版(仅限本地, 可选)

```bash

docker  run  --rm  -t  --network  none  --gpus  '0'  --name  hackathon  -v  /tmp/repo/:/repo  registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2  bash  -c  "cd /repo && python3 compute_score_old.py"

```

  

- 跑一下TRT版(仅限本地)，并计算PD_score, time_cost

```bash

docker  run  --rm  -t  --network  none  --gpus  '0'  --name  hackathon  -v  /tmp/repo/:/repo  registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2  bash  -c  "cd /repo && python3 compute_score_new.py"

```

  

- 跑一下TRT版(仅限本地)，测试一下测评代码是否ok

```bash

docker  run  --rm  -t  --network  none  --gpus  '0'  --name  hackathon  -v  /tmp/repo/:/repo  registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:v2  bash  -c  "cd /repo && python3 compute_score.py"

```

  

##### 评分规则猜想

- 根据我最近几次的提交记录，大概可以得出一个评分公式（可能不一定对）, 先说几个关键系数。

- 提分系数1：TRT优化倍数。你跑一下pytorch版，算一下平均推理时间，然后再跑一下TRT版，算一下TRT的推理平均分，两者相除，就是优化倍数。比如pytorch推理2700ms, trt推理 900ms，优化倍数就是3倍。如果优化倍数低于0.01，我猜可能就没分了。

- 提分系数2： PDScore优化倍数。最大容忍PDScore是12，你的TRT PDscore如果是6，那么这个精度就是1 / (1 - 6 / 12) = 2，如果你大于12，那就是0，甚至负分（群主大佬应该优化了，现在没有负分了，最低是0）。

- 最后，这个公式大概就是（pytorch推理时间/TRT推理时间）* （1 / (1 - 你的PDScore / 12)）* 某个常数

- 这个常数我估摸大概是300-400左右。上面的变量都是指的是平均值，当然还是老话PDScore大于12没分，我猜应该是单张图片没分。保险起见，最好控制max_PDscore < 12，建议max_PDScore在10以内，因为评测可能会高一些。

- 所以建议大家在线上测评前先估摸一下自己的分数大概是多少，如果有一定提升，再线上测评，这样也可以节省一些算力以及等待时间。

- 最后说一句：这个只是猜测，最终得分还是要以线上测评为准的。



### 优化记录

| 序号 | 提交时间            | 优化操作                                                     | 得分      | 提升幅度     | 其他备注                                                     | 最终是否应用 |
| ---- | ------------------- | ------------------------------------------------------------ | --------- | ------------ | ------------------------------------------------------------ | ------------ |
| 1    | 2023-07-18 22:50:19 | TensorRT fp16推理（除CLIP外）                                | 2351.6981 | 0->2351.6981 | CLIP fp16溢出，结果为nan，所以CLIP暂时用fp32                 | 是           |
| 2    | 2023-07-18 23:51:34 | 开启cuda graph                                               | 2414.6493 | 63分         |                                                              | 是           |
| 3    | 2023-07-19 22:46:34 | 双context+双stream+双profile优化self.apply_model             | 变化不大  | 10ms以内     | 基本没有提升，需要进一步优化                                 | 否           |
| 4    | 2023-07-20 01:00:51 | 将两次self.apply_model组batch后再计算一次，减少等待时间      | 2835.9280 | 421分        | 组batch后，上面的双context策略彻底废弃                       | 是           |
| 5    | 2023-07-20 15:49:55 | 设置编译时的优化等级系数，默认是3,改成最高的5。取消一个tqdm进度条显示（大概可以省3ms） | 3654.0500 | 818分        | 这个很早想过，但是想放在最后，不过第一名分数太高了，于是提前用上了这个手段，效果还不错。 | 是           |
| 6    | 2023-07-20 18:24:04 | 1. 将control_net和unet拼接成一个新模型，新模型为union_model,union模型可以减少中间结果的储存和拷贝，同时降低onnx导出和engine导出难度，并且可以让tensorRT进一步优化模型。2.同时将text_embedding部分也组batch，减少clip推理时间（大概1-2ms)。3.将所有run_engine的output clone操作取消（组batch后不需要clone了，不会有内存共享问题）。4.减少max_batch_size,从16改成4,opt_batch改成2,方便组batch后的union_model和clip更好的推理。5. 开启双stream,一个用于正常推理，一个用于cuda_graph。这个操作对于第一张图片有一定优化，但是后续图片只会用cuda_graph的stream。 | 4027.2095 | 373分        |                                                              | 是           |
| 7    |                     | 因为union_model和clip都是batch_size=2的操作了。所以可以进一步优化，将union_model和clip在onnx->trt的min/opt/max batch_size都写成2,vae固定为1，理论上应该会有一些加成。同时union_model和clip在onnx导出时，之前动态维度写的是B，现在我看可以试着写成2B了，代表2的倍数，理论上应该也会有一些优化。 |           |              |                                                              |              |
| 8    |                     | 逐层对比CLIP模型的TensorRT fp16和onnx cpu fp32的结果，找到溢出层，实现clip fp16正常导出 |           |              |                                                              |              |
|      |                     | plugin编写                                                   |           |              |                                                              |              |

