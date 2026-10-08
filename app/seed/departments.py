"""科室种子数据。

``symptom_keywords`` 有两个用途：
1. **规则兜底分诊**：LLM 不可用时，按关键词命中数推荐科室，保证分诊功能
   不因 AI 故障而完全失效——这是降级链在业务层的延伸
2. **科室定向召回**：RAG 检索时可用科室过滤知识库，减少跨科室噪声

关键词刻意选了「患者会说出口的词」而非医学术语：患者说「拉肚子」
不会说「腹泻」，两者都要覆盖。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.auth import Department

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DepartmentSeed:
    """一条科室种子记录。"""

    code: str
    name: str
    description: str = ""
    symptom_keywords: tuple[str, ...] = field(default_factory=tuple)


# 覆盖常见病与急诊场景的科室集合。code 用于分诊 prompt 与前端路由，
# 保持稳定不随显示名调整。
DEPARTMENTS: tuple[DepartmentSeed, ...] = (
    DepartmentSeed(
        code="EMERGENCY",
        name="急诊科",
        description="处理急性病症、创伤与危急情况，24 小时接诊",
        symptom_keywords=(
            "昏迷", "意识不清", "意识改变", "抽搐", "呼吸困难", "喘不上气",
            "胸痛", "胸闷压榨", "大出血", "出血不止", "剧烈头痛", "突发头痛",
            "口眼歪斜", "言语不清", "肢体无力", "单侧麻木", "休克", "中毒",
            "烧伤", "烫伤", "外伤", "骨折", "车祸", "晕倒", "昏厥", "窒息",
            "心跳骤停", "剧烈腹痛", "呕血", "便血", "高热惊厥",
        ),
    ),
    DepartmentSeed(
        code="INTERNAL",
        name="内科",
        description="呼吸、消化、内分泌等非手术治疗科室，常见病首诊科室",
        symptom_keywords=(
            "发热", "发烧", "咳嗽", "咳痰", "感冒", "咽痛", "嗓子疼",
            "腹泻", "拉肚子", "便秘", "腹痛", "胃疼", "胃痛", "反酸",
            "恶心", "呕吐", "乏力", "没力气", "消瘦", "食欲不振",
            "血糖", "糖尿病", "甲状腺", "血压高", "血脂",
        ),
    ),
    DepartmentSeed(
        code="CARDIO",
        name="心血管内科",
        description="心脏与血管疾病：高血压、冠心病、心律失常、心衰",
        symptom_keywords=(
            "心慌", "心悸", "心跳快", "心跳乱", "胸闷", "气短",
            "活动后气喘", "下肢水肿", "脚肿", "高血压", "血压高",
            "冠心病", "心绞痛", "心律失常", "早搏", "房颤", "血脂高",
        ),
    ),
    DepartmentSeed(
        code="NEURO",
        name="神经内科",
        description="头痛、头晕、失眠、脑血管病、周围神经病变",
        symptom_keywords=(
            "头痛", "头疼", "头晕", "眩晕", "天旋地转", "失眠", "睡不着",
            "多梦", "记忆力下降", "健忘", "手麻", "脚麻", "麻木", "震颤",
            "手抖", "偏头痛", "神经痛", "面瘫", "癫痫", "脑梗", "中风",
        ),
    ),
    DepartmentSeed(
        code="RESP",
        name="呼吸内科",
        description="咳嗽、哮喘、慢阻肺、肺部感染等呼吸系统疾病",
        symptom_keywords=(
            "长期咳嗽", "慢性咳嗽", "咳血", "痰中带血", "哮喘", "喘息",
            "气促", "慢阻肺", "肺气肿", "肺炎", "支气管炎", "胸闷咳嗽",
            "夜间憋醒", "打鼾严重", "呼吸暂停",
        ),
    ),
    DepartmentSeed(
        code="GASTRO",
        name="消化内科",
        description="胃肠、肝胆胰疾病：胃炎、溃疡、肝炎、胰腺炎",
        symptom_keywords=(
            "胃胀", "烧心", "反酸水", "吞咽困难", "噎住", "黄疸",
            "皮肤发黄", "眼白发黄", "肝区疼", "右上腹痛", "肚胀",
            "便血", "黑便", "柏油样便", "长期腹泻", "便秘严重",
            "胃溃疡", "胃炎", "肝炎", "脂肪肝", "胆结石",
        ),
    ),
    DepartmentSeed(
        code="ENDO",
        name="内分泌科",
        description="糖尿病、甲状腺、肥胖、骨代谢等激素相关疾病",
        symptom_keywords=(
            "多饮", "多尿", "口渴厉害", "多食", "体重下降", "肥胖",
            "糖尿病", "血糖高", "血糖低", "低血糖", "甲亢", "甲减",
            "脖子粗", "甲状腺结节", "怕热", "怕冷", "多汗", "骨质疏松",
            "月经不调内分泌", "更年期",
        ),
    ),
    DepartmentSeed(
        code="SURG",
        name="外科",
        description="需手术评估的疾病：创伤、肿块、急腹症、疝气",
        symptom_keywords=(
            "肿块", "包块", "肿物", "疝气", "小肠气", "阑尾", "右下腹痛",
            "需要手术", "伤口", "缝针", "术后", "拆线", "肠梗阻",
            "扭伤", "拉伤", "关节脱位", "痔疮", "肛裂", "静脉曲张",
        ),
    ),
    DepartmentSeed(
        code="ORTHO",
        name="骨科",
        description="骨骼、关节、脊柱与运动损伤",
        symptom_keywords=(
            "腰痛", "腰疼", "颈椎痛", "脖子疼", "肩周炎", "肩膀疼",
            "膝盖疼", "关节痛", "关节肿", "腿疼", "骨折", "扭伤脚踝",
            "腰椎间盘突出", "坐骨神经痛", "脊柱侧弯", "骨质疏松骨折",
            "运动损伤", "不能负重", "活动受限",
        ),
    ),
    DepartmentSeed(
        code="PEDS",
        name="儿科",
        description="14 岁以下儿童的疾病诊治与生长发育评估",
        symptom_keywords=(
            "孩子发烧", "小孩咳嗽", "婴儿", "儿童", "宝宝", "幼儿",
            "新生儿", "小儿", "孩子腹泻", "孩子呕吐", "不长个",
            "发育迟缓", "挑食", "夜啼", "湿疹宝宝", "手足口", "腮腺炎",
        ),
    ),
    DepartmentSeed(
        code="OBGYN",
        name="妇产科",
        description="妇科疾病、妊娠与产前检查",
        symptom_keywords=(
            "月经不调", "月经推迟", "月经量多", "痛经", "闭经", "白带异常",
            "阴道出血", "外阴瘙痒", "怀孕", "妊娠", "产检", "孕吐",
            "胎动", "产后", "子宫肌瘤", "卵巢囊肿", "宫颈炎", "盆腔炎",
        ),
    ),
    DepartmentSeed(
        code="DERM",
        name="皮肤科",
        description="皮疹、瘙痒、脱发、色素异常与皮肤感染",
        symptom_keywords=(
            "皮疹", "起疹子", "瘙痒", "痒", "荨麻疹", "风团", "湿疹",
            "牛皮癣", "银屑病", "痤疮", "痘痘", "脱发", "掉头发",
            "白斑", "色素沉着", "痣变化", "脚气", "灰指甲", "疣",
            "皮肤过敏", "红肿热痛皮肤",
        ),
    ),
    DepartmentSeed(
        code="ENT",
        name="耳鼻喉科",
        description="耳、鼻、咽喉与头颈部疾病",
        symptom_keywords=(
            "耳鸣", "听力下降", "耳聋", "耳痛", "耳朵流脓", "鼻塞",
            "流鼻涕", "打喷嚏", "鼻炎", "鼻窦炎", "鼻出血", "流鼻血",
            "嗅觉下降", "咽干", "咽部异物感", "声音嘶哑", "喉咙痛",
            "扁桃体肿大", "打呼噜", "眩晕伴耳鸣",
        ),
    ),
    DepartmentSeed(
        code="OPHTH",
        name="眼科",
        description="视力、眼表与眼底疾病",
        symptom_keywords=(
            "视力下降", "看不清", "视物模糊", "眼睛干", "干眼", "眼痒",
            "眼红", "结膜炎", "麦粒肿", "飞蚊症", "眼前黑影", "闪光感",
            "视物变形", "视野缺损", "夜盲", "色觉异常", "眼外伤",
            "白内障", "青光眼", "眼底出血",
        ),
    ),
    DepartmentSeed(
        code="PSYCH",
        name="精神心理科",
        description="情绪障碍、焦虑抑郁、睡眠障碍与心理危机干预",
        symptom_keywords=(
            "情绪低落", "抑郁", "高兴不起来", "焦虑", "紧张", "恐慌",
            "惊恐发作", "烦躁", "易怒", "失眠严重", "早醒", "嗜睡",
            "注意力不集中", "记忆力减退伴情绪", "自卑", "无价值感",
            "不想活", "自杀念头", "自残", "幻觉", "妄想", "强迫",
            "社交恐惧", "创伤后",
        ),
    ),
    DepartmentSeed(
        code="URO",
        name="泌尿外科",
        description="泌尿系统与男性生殖系统疾病",
        symptom_keywords=(
            "尿频", "尿急", "尿痛", "排尿困难", "尿不尽", "夜尿多",
            "血尿", "尿血", "小便浑浊", "腰痛伴血尿", "肾结石",
            "输尿管结石", "前列腺", "前列腺炎", "前列腺增生",
            "阴囊肿痛", "男性功能", "尿路感染",
        ),
    ),
    DepartmentSeed(
        code="ONCO",
        name="肿瘤科",
        description="肿瘤的诊断、化疗与随访管理",
        symptom_keywords=(
            "肿瘤", "癌症", "化疗", "放疗", "靶向治疗", "免疫治疗",
            "肿块进行性增大", "无痛性肿块", "消瘦明显", "持续低热",
            "淋巴结肿大", "肿瘤标志物", "癌胚抗原", "复查随访",
        ),
    ),
    DepartmentSeed(
        code="HEALTH",
        name="体检中心",
        description="健康体检、入职体检与慢病筛查",
        symptom_keywords=(
            "体检", "健康检查", "入职体检", "年度体检", "筛查",
            "化验单解读", "体检报告", "指标异常无症状", "亚健康",
        ),
    ),
    DepartmentSeed(
        code="LAB",
        name="检验科",
        description="血液、生化、免疫与微生物检验，出具检查报告",
        symptom_keywords=(
            "抽血", "验血", "血常规", "尿常规", "生化", "肝功能",
            "肾功能", "血脂四项", "血糖检测", "糖化血红蛋白",
            "甲功五项", "肿瘤标志物检测", "凝血功能", "血型",
            "病原检测", "核酸检测", "化验",
        ),
    ),
    DepartmentSeed(
        code="PHARMACY",
        name="药剂科",
        description="处方审核、药品调配与用药指导",
        symptom_keywords=(
            "取药", "拿药", "配药", "药品咨询", "用药指导", "药物相互作用",
            "药品过敏", "处方审核", "发药", "药物剂量", "特殊药品",
        ),
    ),
)


async def seed_departments(db: AsyncSession) -> int:
    """写入科室种子数据，返回新建条数。

    幂等实现：按 ``code`` 查询已存在的科室，只插入缺失的。
    已存在的记录**不覆盖** —— 管理员可能已经改过描述或关键词，
    重跑种子脚本不应该冲掉人工维护的数据。
    """
    existing_codes = set(
        (await db.execute(select(Department.code))).scalars().all()
    )
    created = 0
    for seed in DEPARTMENTS:
        if seed.code in existing_codes:
            continue
        db.add(
            Department(
                code=seed.code,
                name=seed.name,
                description=seed.description or None,
                symptom_keywords=list(seed.symptom_keywords),
            )
        )
        created += 1
    await db.flush()
    logger.info("科室种子数据：新建 %d 条，已存在 %d 条", created, len(existing_codes))
    return created


__all__ = ["DepartmentSeed", "DEPARTMENTS", "seed_departments"]
