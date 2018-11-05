'''
载入相关库
'''
# from sklearn.linear_model import LogisticRegression
# import xgboost as xgb
# from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import numpy as np
class grade_pred:
    '''
    输入数据类型：dict
    变量的含义：
    {
    "features":
    ['problem_count_grade7', 'problem_count_grade8', 'problem_count_grade9',
              'problem_count_other', 'video_count_grade7', 'video_count_grade8', 'video_count_grade9',
              'video_avg_progress_grade7', 'video_avg_progress_grade8', 'video_avg_progress_grade9',
              'video_avg_progress_show'
              ]
              （'一个月做7年级习题计数'，'一个月做8年级习题计数'，'一个月做9年级习题计数'，
              '一个月做其他年级习题计数'，'一个月中观看7年级视频计数'，'一个月中观看8年级视频计数'，'一个月中观看9年级视频计数'
              '一个月中观看7年级视频的平均进度'，'一个月中观看8年级视频的平均进度'，'一个月中观看9年级视频的平均进度'，
              ，'一个月中观看真人秀的平均进度'）
    'registered': 7
    }
    '''
    def isvalid_data(self,data):
        '''
        判断输入的数据是否为合适的用户样本
        判断依据：data["features"]是否为空,data["features"]里为0的变量数是否大于10,若是，则为不活跃用户，不适合用来预
        测
        :param data: dict
        :return: boolean
        '''
        if not data['features']:
            return False
        num_zero=len(list(filter(lambda x:x==0,data['features'])))
        if num_zero>10:
            print('用户活跃度过低')
            return False
        #为了保证测试样本和建模时使用的样本一致，如果输出得到的有效预测过少，可注释掉样本注册年级的判定
        if data['registered'] not in [7,8,9]:
            return False
        return True


    def loadmodel(self,path,modelname):
        '''
        载入模型
        :param path:路径 str
        :param modelname: xgb->xgboost，onehot->one-hot编码，lr->线性回归 str 模型名
        :return: obj
        '''
        modeldic={'xgb':"xgb_model.m",'onehot':"enc_onehot.m",'lr':"lr_model.m"}
        try:
            model=joblib.load(path+ modeldic[modelname])
            return model
        except FileNotFoundError as e:
            print('path of model error !')

    def deldata(self,data):
        '''
        特征工程，生成交叉特征：data['features']里为0的变量数,各年级下的观看视频数和习题数除以总观看/做题数
        :param data: dict
        :return: list  自变量[[]]
        '''
        var=data['features']
        eps=0.000001#防止分母为0出现计算错误
        count0=len(list(filter(lambda x:x==0,var)))
        var.append(count0)
        for i in range(4):
            var.append(var[i]/(sum(var[:4])+eps))
        for i in range(4,7):
            var.append(var[i] /(sum(var[4:7])+eps))
        return [var]
    def model_predict(self,model_path,var,predict_proba=True):
        '''
        流程：根据变量，经过xgboost训练提取叶子节点信息，进行one-hot编码形成稀疏矩阵，
        用线性回归进行预测
        :param model_path: str 模型路径
        :param var: list 自变量
        :param predict_proba:boolean #确定是否输出三个年级的概率 (若是，则输出列表的索引0,1,2对应的值分别为7，8，9
        三个年级概率值)
        :return: list
        '''
        xgb =self.loadmodel( model_path, 'xgb')
        onehot = self.loadmodel( model_path, 'onehot')
        lr = self.loadmodel(model_path, 'lr')
        feature = xgb.apply(var)
        feature_onehot=onehot.transform(feature)
        if predict_proba ==True:
           return lr.predict_proba(feature_onehot).tolist()
        else:
             tmp=lr.predict(feature_onehot).tolist()
             return [tmp[0]+7]



if __name__ == '__main__':
    #测试范例
    g=grade_pred()
    user_test={
    "features": [
        0.0,
        0.0,
        98.0,
        0.0,
        0.0,
        0.0,
        15.0,
        0.0,
        0.0,
        0.95,
        0.0
    ],
    'registered': 7
    }
    if g.isvalid_data(user_test):
        var=g.deldata(user_test)
        pred_list=g.model_predict('', var, predict_proba=True)
        print(pred_list)
        for i in range(3):
            print(str(i+7)+'年级概率：',str(pred_list[0][i]))
        print('用户年级为:',g.model_predict('', var, predict_proba=False))