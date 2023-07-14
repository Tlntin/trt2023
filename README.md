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
2. 正式运行docker, 顺便将本地代码覆盖docker中的代码
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

