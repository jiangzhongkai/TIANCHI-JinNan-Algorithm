 ## TIANCHI-JinNan-Algorithm
 
#### 竞赛题目：
 `异烟酸用作医药中间体，主要用于制抗结核病药物异烟肼，也用于合成酰胺、酰肼、酯类等衍生物。烟酰胺生产过程包含水解脱色、结晶甩滤等过程。每个步骤会受到温度、时间、压强等各方面因素的影响，造成异烟酸收率的不稳定。为保证产品质量和提高生产效率，需要调整和优化生产过程中的参数。然而，根据传统经验的人工调整工艺参数费时费力。近年来，人工智能在工艺参数优化以及视频检测等领域取得了突飞猛进的成果。AI技术的发展有望助力原料药制造企业实现工艺生产革新，规范生产操作过程，从而达到提高产品的收率的目标。`
 
#### 竞赛网址： https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.6c464264psfibt&raceId=231695
 
#### 竞赛数据
`大赛包含有2000批次来自实际异烟酸生产中的各参数的监测指标和最终收率的数据。监测指标由两大工序数十个步骤构成。总生产步骤达30余项。我们将工序和步骤分别用字母和数字代号表示，比如A2，B5分别表示A工序第二步骤和B工序第五步骤。样例数据参考训练数据。`

#### 评估指标
#### 初赛评估指标
`选手提交结果与实际检测到的收率结果进行对比，以均方误差为评价指标，结果越小越好，均方误差计算公式如下：`
         
`其中m为总批次数，y'(i)为选手预测的第i批次的收率值，y(i)为第i批次的实际收率值。线上赛题分A，B，C三个阶段。其中A榜每天评测，B榜评测一次，C榜为隐式评测，即不提供评测数据，选手提交代码后，由工作人员统一运行代码生成结果。`
`初赛结果由B榜和C榜两次榜单最终排名序号简单相加，得到的最终值按照从小到大排序，并给出排名的序号，如序号相同，则C榜排名靠前的选手最终排名靠前。`

#### 复赛评估指标
`初赛最终排名前100的选手，需要提交线下材料，材料内容见提交说明：`
`复赛分为两个部分，每部分都会对选手进行排名：``
1、最优收率值：选手提供参数取值，使得收率值最高，要求除时间参数外的参数，取值范围必须在初赛所有数据集（含C榜，初赛结束后提供）的对应参数取值范围内。`

`2、复赛新数据预测效果：复赛提供新数据，评估代码预测收率的效果，采用均方误差作为评价指标。`

`复赛结果由上述两个部分选手最终排名序号简单相加，得到的最终值按照从小到大排序，并给出排名的序号，如序号相同，则最优收率值排名靠前的选手最终排名靠前。`
 
#### 现在线上测评：
`mse:0.00007622`
