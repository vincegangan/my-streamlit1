#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd  
import numpy as np  
import pickle  
import streamlit as st  
from PIL import Image  


# In[3]:


# Load the trained model  
pickle_in1 = open('VAPRF.pkl', 'rb')  
rf1= pickle.load(pickle_in1)  


# In[ ]:


import streamlit as st
import pandas as pd
import pickle

def main():
    st.title('请输入以下信息，预测ICU老年患者呼吸机相关性肺炎的发生概率')

    # 插入图片
    st.image('呼吸机.jpg')  # 替换'image_path.jpg'为你的图片路径

    # 添加说明性文本
    st.markdown('### 连续型变量')
    albumin = st.number_input('白蛋白(g/dL)', value=0.0)
    age = st.number_input('年龄 (岁)', value=0)
    icustay_time = st.number_input('ICU住院时间 (天)', value=0)
    mechanical_ventilation_days = st.number_input('机械通气时间(天)', value=0)

    st.markdown('### 分类变量')
    sex = st.selectbox('性别', options=[0, 1], format_func=lambda x: '女性' if x == 0 else '男性')
    intubation_times = st.selectbox('插管次数，3代表三次插管及以上', options=[0, 1, 2, 3])
    whether_tracheotomy = st.selectbox('是否气管切开', options=[0, 1], format_func=lambda x: '否' if x == 0 else '是')
    conscious_state = st.selectbox('意识状态', options=[0, 1], format_func=lambda x: '清醒' if x == 0 else '昏迷')
    long_term_combined_use_of_antibiotics = st.selectbox('长期联合使用抗生素 (≥2种并且≥7天)', options=[0, 1], format_func=lambda x: '否' if x == 0 else '是')



    if st.button('预测'):
        input_data = pd.DataFrame([[
            albumin, sex, age, icustay_time, intubation_times, whether_tracheotomy, mechanical_ventilation_days, conscious_state, long_term_combined_use_of_antibiotics
        ]], columns=[
            'albumin', 'sex', 'age', 'icustay.time', 'intubation.times', 'Whether.tracheotomy', 'Mechanical.ventilation.days.', 'conscious.state', 'Long.term.combined.use.of.antibiotics'
        ])
        
    
         # 使用模型进行预测概率
        prediction_proba = rf1.predict_proba(input_data)
        
        # 显示预测概率结果
        st.write(f'预测结果：该ICU老年患者呼吸机相关性肺炎发生的概率为：{prediction_proba[0][1]:.2%}')

if __name__ == "__main__":
    main()

