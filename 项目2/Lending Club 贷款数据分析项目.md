# Lending Club 贷款数据分析项目

## 项目概述

​        本项目基于Lending Club的"LoanStats_2017Q1"数据集，该数据集包含美国一家银行于2017年第一季度贷款记录。

## 分析目标

- 了解2017年Q1 Lending Club贷款数据的整体情况
- 识别影响贷款违约的关键因素
- 建立贷款风险评估模型
- 提供业务决策建议

##### 导入必要的库

~~~ python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')
import joblib

plt.style.use('seaborn-white')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print('所有库导入成功')
~~~

## 1.数据加载和初步探索

##### 加载数据

~~~ python
try:
    df = pd.read_csv(f'D:\下载\LoanStats\LoanStats_2017Q1.csv',skiprows=1,low_memory=False,encoding='gbk')
    print('导入数据成功')
except FileNotFoundError:
    print('找不到数据文件,请检查文件路径')

print('数据集基本信息:')
print(f'数据集形状:{df.shape}')
print(f'数据列数:{len(df.columns)}')
print('\n前5行数据:')
display(df.head())
~~~

## 2.数据清洗与预处理

##### 数据清洗

~~~ python
print('贷款状态分布:')
print(df['loan_status'].value_counts())

def create_target(status):
    if status in ['Charged Off','Default']:
        return 1
    elif status in ['Fully Paid','Current']:
        return 0
    else:
        return -1

df['target'] = df['loan_status'].apply(create_target)
df = df[df['target'] != -1]
print(f'\n处理后数据量:{len(df)}')
print(f"坏账率:{df['target'].mean():.3f}")

features = ['loan_amnt','term','int_rate','installment','grade','sub_grade','emp_length','home_ownership','annual_inc','verification_status','purpose','dti','delinq_2yrs','inq_last_6mths','mths_since_last_delinq','open_acc','pub_rec','revol_bal','revol_util','total_acc','initial_list_status','application_type']
analysis_df = df[features + ['target']].copy()

print(f'分析数据集形状:{analysis_df.shape}')
print(f'分析数据集数据类型:{analysis_df.dtypes}')

print("缺失值统计:")
missing_data = analysis_df.isnull().sum()
print(missing_data)

analysis_df['int_rate'] = analysis_df['int_rate'].str.replace('%','').astype(float)
analysis_df['revol_util'] = analysis_df['revol_util'].str.replace('%','').astype(float)

analysis_df['dti'] = analysis_df['dti'].fillna(analysis_df['dti'].mean())
analysis_df['emp_length'] = analysis_df['emp_length'].fillna('0 years')
analysis_df['revol_util'] = analysis_df['revol_util'].fillna(analysis_df['revol_util'].median())
analysis_df['mths_since_last_delinq'] = analysis_df['mths_since_last_delinq'].fillna(999)

def convert_emp_length(emp_str):
    if pd.isna(emp_str):
        return 0
    if emp_str == '< 1 year':
        return 0
    if emp_str == '10+ years':
        return 10
    try:
        return int(emp_str.split()[0])
    except:
        return 0

analysis_df['emp_length_num'] = analysis_df['emp_length'].apply(convert_emp_length)
grade_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
analysis_df['grade_num'] = analysis_df['grade'].map(grade_map)

analysis_df['loan_to_income'] = analysis_df['loan_amnt'] / analysis_df['annual_inc']

analysis_df.loc[analysis_df['loan_to_income'] > 1,'loan_to_income'] = 1

print(f'预处理后数据形状:{analysis_df.shape}')
print(analysis_df.info())
print(analysis_df.head())
~~~

## 3.探索性分析

### 3.1关键特征与违约率的关系

##### 关键特征与违约率的关系可视化

~~~ python
fig,axes = plt.subplots(2,3,figsize=(30,20))
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

grade_default = analysis_df.groupby('grade')['target'].mean().sort_index()
axes[0,0].bar(grade_default.index.to_numpy(),grade_default.values,color='b')
axes[0,0].set_title('各等级贷款违约率',fontsize=20,fontweight='bold')

purpose_default = analysis_df.groupby('purpose')['target'].mean().sort_values(ascending=False)
axes[0,1].bar(purpose_default.head(8).index.to_numpy(),purpose_default.head(8).values,color='g')
axes[0,1].set_title('主要贷款目的违约率',fontsize=20,fontweight='bold')
axes[0,1].tick_params(axis='x',rotation=45)

analysis_df['int_rate_bin'] = pd.cut(analysis_df['int_rate'],bins=10)
int_rate_default = analysis_df.groupby('int_rate_bin')['target'].mean()
axes[0,2].plot(range(len(int_rate_default)),int_rate_default.values,marker='o',linewidth=2)
axes[0,2].set_title('利率与违约率',fontsize=20,fontweight='bold')
axes[0,2].set_xticks(range(len(int_rate_default)))
axes[0,2].set_xticklabels([f'{interval.left:.1f}-{interval.right:.1f}'for interval in int_rate_default.index],rotation=45,fontsize=16)

emp_default = analysis_df.groupby('emp_length_num')['target'].mean()
axes[1,0].plot(emp_default.index.to_numpy(),emp_default.values,color='g',linewidth=2,marker='s')
axes[1,0].set_title('就业年限与违约率',fontsize=20,fontweight='bold')

axes[1,1].hist(analysis_df['dti'],bins=30,alpha=0.5,color='r')
axes[1,1].set_title('债务收入比分布',fontsize=20,fontweight='bold')
axes[1,1].set_xlabel('DTI',fontsize=16)
axes[1,1].set_yscale('log')

scatter = axes[1,2].scatter(analysis_df['loan_amnt'],analysis_df['int_rate'],c=analysis_df['target'],alpha=0.6,cmap='coolwarm')
axes[1,2].set_title('贷款金额vs利率(颜色=违约)',fontsize=20,fontweight='bold')
axes[1,2].set_xlabel('贷款金额',fontsize=16)
axes[1,2].set_ylabel('利率',fontsize=16)
plt.colorbar(scatter,ax=axes[1,2])

plt.tight_layout()
plt.show()
~~~

## 4.建模分析

### 4.1特征工程

##### 特征工程和预处理

~~~ python
final_features = ['loan_amnt','term','int_rate','installment','grade','sub_grade','emp_length','home_ownership','annual_inc','verification_status','purpose','dti','delinq_2yrs','inq_last_6mths','open_acc','pub_rec','revol_bal','revol_util','total_acc','loan_to_income','grade_num']

model_df = analysis_df[final_features + ['target']].copy()

categorical_cols = model_df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col != 'target']
print('需要编码的类别列:',categorical_cols)
model_encoded = pd.get_dummies(model_df,columns=categorical_cols,drop_first=True)

print(f'编码后特征数量:{model_encoded.shape[1] - 1}')
~~~

### 4.2预测模型比较

##### 分别训练Logistic Regression、Random Forest以及Gradient Boosting模型

~~~ python
X = model_encoded.drop('target',axis=1)
y = model_encoded['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

models = {
    'Logistic Regression':LogisticRegression(max_iter=1000,random_state=42),
    'Random Forest':RandomForestClassifier(n_estimators=200,random_state=42),
    'Gradient Boosting':GradientBoostingClassifier(n_estimators=200,random_state=42),
}
results = {}
plt.figure(figsize=(15,5))

for i,(name,model) in enumerate(models.items()):
    model.fit(X_train,y_train)

    y_pred_proba = model.predict_proba(X_test)[:,1]

    auc_score = roc_auc_score(y_test,y_pred_proba)
    cv_score = cross_val_score(model,X_train,y_train,cv=5,scoring='roc_auc')

    results[name] = {
        'model':model,
        'auc':auc_score,
        'cv_mean':cv_score.mean(),
        'cv_std':cv_score.std()
    }
    print(f'{name}:')
    print(f'测试集AUC:{auc_score:.4f}')
    print(f'交叉验证AUC:{cv_score.mean():.4f}(+/-){cv_score.std()*2:.4f}')

    fpr,tpr,thresholds = roc_curve(y_test,y_pred_proba)
    plt.subplot(1,3,i+1)
    plt.plot(fpr,tpr,label=f'{name}(AUC={auc_score:.3f})',linewidth=2)
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} - ROC曲线')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()
~~~

### 4.3预测模型调优

##### 随机森林调优

~~~ python
param_grid = {
    'n_estimators':[100,200],
    'max_depth':[10,15,20],
    'min_samples_split':[8,10,12],
    'min_samples_leaf':[3,4,5],
    'class_weight':['balanced',None]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf,param_grid=param_grid,cv=5,scoring='roc_auc',n_jobs=-1)
grid_search.fit(X_train,y_train)

print('最佳参数:',grid_search.best_params_)
print('最佳分数:',grid_search.best_score_)

best_rf1 = grid_search.best_estimator_
y_pred = best_rf1.predict(X_test)
y_pred_proba = best_rf1.predict_proba(X_test)[:,1]

print('\n随机森林详细评估')
print(f'测试集AUC:{roc_auc_score(y_test,y_pred_proba):.4f}')
print('\n分类报告:')
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['好帐','坏账'],yticklabels=['好帐','坏账'])
plt.title('混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.show()

best_rf2 = RandomForestClassifier(random_state=42,class_weight='balanced',max_depth=10,min_samples_leaf=4,min_samples_split=8,n_estimators=200)
best_rf2.fit(X_train,y_train)
y_pred = best_rf2.predict(X_test)
y_pred_proba = best_rf2.predict_proba(X_test)[:,1]

print('\n随机森林详细评估')
print(f'测试集AUC:{roc_auc_score(y_test,y_pred_proba):.4f}')
print('\n分类报告:')
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['好帐','坏账'],yticklabels=['好帐','坏账'])
plt.title('混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.show()
~~~

### 4.4特征重要性排名

##### 特征重要性排名

~~~ python
feature_importance = pd.DataFrame({
    'feature':X.columns,
    'importance':best_rf2.feature_importances_
}).sort_values('importance',ascending=False)

plt.figure(figsize=(12,8))

sns.barplot(data=feature_importance.head(15),x='feature',y='importance')
plt.title('TOP15特征重要性',fontsize=20,fontweight='bold')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

print('最重要的10个特征:')
for i,row in feature_importance.head(10).iterrows():
    print(f"{i+1:2d}.{row['feature']}:{row['importance']:.4f}")
~~~

## 5.商业洞察和建议

### 5.1商业洞察

##### 关键业务指标总结

~~~ python
print('关键业务总结:')
print('='*50)

grade_risk = analysis_df.groupby('grade').agg({
    'target':['mean','count'],
    'int_rate':'mean',
}).round(4)
grade_risk.columns = ['违约率','贷款数量','平均利率']
print('\n1.各等级风险分析:')
print(grade_risk)

print('\n2.关键阈值分析:')
print(f"利率>15%的违约率:{analysis_df[analysis_df['int_rate']>15]['target'].mean():.3f}")
print(f"DTI>20的违约率:{analysis_df[analysis_df['dti']>20]['target'].mean():.3f}")
print(f"循环信用使用率>70%的违约率:{analysis_df[analysis_df['revol_util']>70]['target'].mean():.3f}")

print('\n3.高风险组合分析:')
high_risk_conditions = []
if 'int_rate' in analysis_df.columns: high_risk_conditions.append(analysis_df['int_rate'] > 15)
if 'dti' in analysis_df.columns: high_risk_conditions.append(analysis_df['dti'] > 20)
if 'grade_num' in analysis_df.columns: high_risk_conditions.append(analysis_df['grade_num'] >= 5)

if high_risk_conditions:
    high_risk_combo = analysis_df[np.all(high_risk_conditions, axis=0)]
    if len(high_risk_combo) > 0:
        print(f"高风险组合违约率: {high_risk_combo['target'].mean():.3f}")
        print(f"高风险组合数量: {len(high_risk_combo)}")
~~~

## 5.2商业建议

##### 基于以上分析，提出以下建议：

##### 1.风险定价策略

- 利用贷款等级和子等级作为信用风险的代理指标

- 对高等级贷款（E、F、G）提高风险溢价")

##### 2.审批优化

- 重点关注高利率（>15%）和高DTI（>20%）的申请
- 对环信用使用率高的申请人加强审核

##### 3.催收策略

-  对同时具备高利率、高DTI和低等级的贷款建立早期预警
-  重点关注就业年限短的借款人

## 6.模型部署准备

##### 模型部署

~~~ python
joblib.dump(best_rf2,'lending_club_model_no_fico.pkl')

model_info = {
    'features':X.columns.tolist(),
    'feature_importance':feature_importance.to_dict(),
    'performance':{
        'auc':roc_auc_score(y_test,y_pred_proba),
        'best_params':grid_search.best_params_
    }
}
print('模型已保存,可用于生产环境')
print(f'最终模型性能:AUC = {roc_auc_score(y_test,y_pred_proba):.4f}')
~~~

## 7.结论

### 7.1 项目总结

​        本项目通过系统性的数据分析，成功构建了Lending Club贷款违约预测模型并对模型进行调优，取得了良好的预测效果（AUC=0.6830 提升至 AUC=0.7045）。分析揭示了影响贷款违约的关键因素，并为业务决策提供了数据支持。

### 7.2 主要发现

1. 贷款等级和利率是最重要的风险指标
2. 借款人财务状况对违约风险有显著影响
3. 信用使用行为提供了重要的风险信号
4. 多重风险因素叠加会显著提高违约概率

### 7.3 局限性

1. 数据集中缺少FICO信用分数等传统信用评估指标
2. 样本时间范围有限，可能无法捕捉经济周期的影响
3. 模型在极端情况下的预测能力有待验证

### 7.4 后续工作建议

1. 模型持续优化：引入更多特征，尝试深度学习等先进算法
2. 实时预测系统：开发在线风险评估API，支持实时审批决策
3. 行为数据分析：结合还款行为数据，构建动态风险评估模型
4. 外部数据整合：引入外部经济指标和信用数据，提升模型准确性

##### 保存清洗后的数据和模型结果

~~~ python
analysis_df.to_csv('Lending_Club_Analysis_Data.csv')
model_df.to_csv('Lending_Club_Model_Data.csv')
model_encoded.to_csv('Lending_Club_Model_Encoded_Data.csv')
feature_importance.to_csv('Lending_Club_Feature_Importance_Data.csv')
predictions_df = pd.DataFrame({
    'actual':y_test,
    'predicted':y_pred,
    'Predicted_Probability':y_pred_proba
})
predictions_df.to_csv('Lending_Club_Predictions_Data.csv')

print('数据已保存为csv文件')
print('项目完成')
~~~

