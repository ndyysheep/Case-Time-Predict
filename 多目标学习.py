import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, Concatenate, Flatten, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib
from tensorflow.keras.layers import Dropout

class PauseTrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if os.path.exists('pause_training.txt'):
            print(f"Pausing training at epoch {epoch+1}")
            self.model.stop_training = True
            self.model.save('paused_model.keras')
            save_encoders()

class PrintSavedValue(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print()
        print("最好的模型为：")
        print(f"Epoch {epoch+1}: val_total_time_output_root_mean_squared_error = {logs['val_total_time_output_root_mean_squared_error']}")

def save_encoders():
    for key, encoder in encoders.items():
        joblib.dump(encoder, f'MPKL/le_{key}.pkl')
    joblib.dump(node_to_index, 'MPKL/node_to_index.pkl')
    joblib.dump(scaler, 'MPKL/scaler.pkl')
    joblib.dump(maxlen, 'MPKL/maxlen.pkl')

# 检查是否使用GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("GPU is available")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("GPU is not available")

# 从 Excel 文件加载数据
df = pd.read_excel('train.xlsx')

# 拆分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=5140)
train_df.to_excel('train_M.xlsx', index=False)
test_df.to_excel('test_M.xlsx', index=False)

# 分别加载训练集和测试集数据
df = pd.read_excel('train_M.xlsx')
test_df = pd.read_excel('test_M.xlsx')

# 数据预处理
node_sequences = df['处置部门'].str.split(';')
event_sources = df['事件来源']
streets = df['所属街道']
categories = df['一级分类']
work_order_types = df['工单类型']
secondary_features = df['二级分类']
administrative_divisions = df['行政区划']
weekdays = df['星期']
holidays = df['是否为节假日']
hours = df['小时']
months = df['月份']
delays = df['延期情况']
event_levels = df['事件等级']
processing_times = df['处置时间间隔/小时'].str.split(';').apply(lambda x: list(map(float, x)))
total_times = df['总消耗时间/小时']

# 创建节点到索引的映射
all_nodes = sorted(set(sum(node_sequences, [])))
node_to_index = {node: idx for idx, node in enumerate(all_nodes)}
num_classes = len(all_nodes)

# 使用 OneHotEncoder 对所有类别进行独热编码
encoders = {
    'event_source': OneHotEncoder(sparse_output=False),
    'street': OneHotEncoder(sparse_output=False),
    'category': OneHotEncoder(sparse_output=False),
    'work_order_type': OneHotEncoder(sparse_output=False),
    'secondary_feature': OneHotEncoder(sparse_output=False),
    'administrative_division': OneHotEncoder(sparse_output=False),
    'delay': OneHotEncoder(sparse_output=False),
    'event_level': OneHotEncoder(sparse_output=False),
}

event_sources_encoded = encoders['event_source'].fit_transform(event_sources.values.reshape(-1, 1))
streets_encoded = encoders['street'].fit_transform(streets.values.reshape(-1, 1))
categories_encoded = encoders['category'].fit_transform(categories.values.reshape(-1, 1))
work_order_types_encoded = encoders['work_order_type'].fit_transform(work_order_types.values.reshape(-1, 1))
secondary_features_encoded = encoders['secondary_feature'].fit_transform(secondary_features.values.reshape(-1, 1))
administrative_divisions_encoded = encoders['administrative_division'].fit_transform(administrative_divisions.values.reshape(-1, 1))
delays_encoded = encoders['delay'].fit_transform(delays.values.reshape(-1, 1))
event_levels_encoded = encoders['event_level'].fit_transform(event_levels.values.reshape(-1, 1))

# 对星期和月份进行循环编码
weekdays_sin = np.sin(2 * np.pi * weekdays / 7)
weekdays_cos = np.cos(2 * np.pi * weekdays / 7)
months_sin = np.sin(2 * np.pi * months / 12)
months_cos = np.cos(2 * np.pi * months / 12)

# 将节点序列转换为索引序列
sequences = [[node_to_index[node] for node in seq if node in node_to_index] for seq in node_sequences]

# 确保所有序列的长度一致
lengths = list(map(len, sequences))
consistent_length_indices = [i for i, length in enumerate(lengths) if length == len(sequences[i])]

sequences = [sequences[i] for i in consistent_length_indices]
event_sources_encoded = event_sources_encoded[consistent_length_indices]
streets_encoded = streets_encoded[consistent_length_indices]
categories_encoded = categories_encoded[consistent_length_indices]
work_order_types_encoded = work_order_types_encoded[consistent_length_indices]
secondary_features_encoded = secondary_features_encoded[consistent_length_indices]
administrative_divisions_encoded = administrative_divisions_encoded[consistent_length_indices]
weekdays_sin = weekdays_sin[consistent_length_indices]
weekdays_cos = weekdays_cos[consistent_length_indices]
holidays = holidays[consistent_length_indices]
hours = hours[consistent_length_indices]
months_sin = months_sin[consistent_length_indices]
months_cos = months_cos[consistent_length_indices]
delays_encoded = delays_encoded[consistent_length_indices]
event_levels_encoded = event_levels_encoded[consistent_length_indices]
processing_times = processing_times[consistent_length_indices]
total_times = total_times[consistent_length_indices]

# 生成训练数据
X_nodes, X_event_sources, X_streets, X_categories, X_work_order_types, X_secondary_features, X_administrative_divisions, X_weekdays_sin, X_weekdays_cos, X_holidays, X_hours, X_months_sin, X_months_cos, X_delays, X_event_levels, y_nodes, y_processing_times, y_total_times = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for seq, es, street, category, wo_type, sec_feat, admin_div, wd_sin, wd_cos, holiday, hour, month_sin, month_cos, delay, event_level, proc_times, total_time in zip(sequences, event_sources_encoded, streets_encoded, categories_encoded, work_order_types_encoded, secondary_features_encoded, administrative_divisions_encoded, weekdays_sin, weekdays_cos, holidays, hours, months_sin, months_cos, delays_encoded, event_levels_encoded, processing_times, total_times):
    for i in range(1, len(seq)):
        X_nodes.append(seq[:i])
        X_event_sources.append(es)
        X_streets.append(street)
        X_categories.append(category)
        X_work_order_types.append(wo_type)
        X_secondary_features.append(sec_feat)
        X_administrative_divisions.append(admin_div)
        X_weekdays_sin.append(wd_sin)
        X_weekdays_cos.append(wd_cos)
        X_holidays.append(holiday)
        X_hours.append(hour)
        X_months_sin.append(month_sin)
        X_months_cos.append(month_cos)
        X_delays.append(delay)
        X_event_levels.append(event_level)
        y_nodes.append(seq[i])
        y_processing_times.append(proc_times[i])
        y_total_times.append(total_time)

# 填充序列
maxlen = max(map(len, X_nodes))
X_nodes = pad_sequences(X_nodes, maxlen=maxlen, padding='pre')
X_event_sources = np.array(X_event_sources)
X_streets = np.array(X_streets)
X_categories = np.array(X_categories)
X_work_order_types = np.array(X_work_order_types)
X_secondary_features = np.array(X_secondary_features)
X_administrative_divisions = np.array(X_administrative_divisions)
X_weekdays_sin = np.array(X_weekdays_sin)
X_weekdays_cos = np.array(X_weekdays_cos)
X_holidays = np.array(X_holidays)
X_hours = np.array(X_hours)
X_months_sin = np.array(X_months_sin)
X_months_cos = np.array(X_months_cos)
X_delays = np.array(X_delays)
X_event_levels = np.array(X_event_levels)
y_nodes = np.array(y_nodes)
y_processing_times = np.array(y_processing_times).reshape(-1, 1)
y_total_times = np.array(y_total_times).reshape(-1, 1)

# 归一化处理时间
scaler = MinMaxScaler()
y_processing_times = scaler.fit_transform(y_processing_times)
y_total_times = scaler.fit_transform(y_total_times)

# 将 y_nodes 进行独热编码
y_nodes = np.eye(len(all_nodes))[y_nodes]

# 拆分训练和测试数据
X_train_nodes, X_test_nodes, X_train_es, X_test_es, X_train_streets, X_test_streets, X_train_categories, X_test_categories, X_train_wo_types, X_test_wo_types, X_train_sec_feats, X_test_sec_feats, X_train_admin_divs, X_test_admin_divs, X_train_weekdays_sin, X_test_weekdays_sin, X_train_weekdays_cos, X_test_weekdays_cos, X_train_holidays, X_test_holidays, X_train_hours, X_test_hours, X_train_months_sin, X_test_months_sin, X_train_months_cos, X_test_months_cos, X_train_delays, X_test_delays, X_train_event_levels, X_test_event_levels, y_train_nodes, y_test_nodes, y_train_proc_times, y_test_proc_times, y_train_total_times, y_test_total_times = train_test_split(
    X_nodes, X_event_sources, X_streets, X_categories, X_work_order_types, X_secondary_features, X_administrative_divisions, X_weekdays_sin, X_weekdays_cos, X_holidays, X_hours, X_months_sin, X_months_cos, X_delays, X_event_levels, y_nodes, y_processing_times, y_total_times, test_size=0.2, random_state=5140)

# 定义模型参数
embedding_dim = 64
hidden_units = 128
learning_rate = 0.001
dropout_rate = 0.2

# 构建混合模型
with tf.device('/GPU:0'):
    node_input = Input(shape=(maxlen,))
    event_source_input = Input(shape=(event_sources_encoded.shape[1],))
    street_input = Input(shape=(streets_encoded.shape[1],))
    category_input = Input(shape=(categories_encoded.shape[1],))
    work_order_type_input = Input(shape=(work_order_types_encoded.shape[1],))
    secondary_feature_input = Input(shape=(secondary_features_encoded.shape[1],))
    administrative_division_input = Input(shape=(administrative_divisions_encoded.shape[1],))
    weekday_sin_input = Input(shape=(1,))
    weekday_cos_input = Input(shape=(1,))
    holiday_input = Input(shape=(1,))
    hour_input = Input(shape=(1,))
    month_sin_input = Input(shape=(1,))
    month_cos_input = Input(shape=(1,))
    delay_input = Input(shape=(delays_encoded.shape[1],))
    event_level_input = Input(shape=(event_levels_encoded.shape[1],))

    # 嵌入层
    node_embedding = Embedding(input_dim=num_classes, output_dim=embedding_dim)(node_input)

    # 自注意力层
    attention = MultiHeadAttention(num_heads=8, key_dim=embedding_dim)(node_embedding, node_embedding)
    attention = Add()([node_embedding, attention])
    attention = LayerNormalization(epsilon=1e-6)(attention)

    # 合并自注意力层的输出与其他特征
    additional_features = Concatenate()([
        event_source_input,
        street_input,
        category_input,
        work_order_type_input,
        secondary_feature_input,
        administrative_division_input,
        weekday_sin_input,
        weekday_cos_input,
        holiday_input,
        hour_input,
        month_sin_input,
        month_cos_input,
        delay_input,
        event_level_input
    ])
    additional_features = tf.expand_dims(additional_features, axis=1)
    additional_features = tf.tile(additional_features, [1, maxlen, 1])

    combined_features = Concatenate()([attention, additional_features])

    # 第一层LSTM，不在时间步之间使用Dropout
    lstm_out = LSTM(hidden_units, return_sequences=True)(combined_features)
    # 在每个时间步之后使用Dropout
    lstm_out = TimeDistributed(Dropout(dropout_rate))(lstm_out)

    # 第二层LSTM
    lstm_out = LSTM(hidden_units, return_sequences=False)(lstm_out)
    # 只在最终输出使用Dropout
    lstm_out = Dropout(dropout_rate)(lstm_out)

    # 多任务输出层
    node_output = Dense(num_classes, activation='softmax', name='node_output')(lstm_out)
    time_output = Dense(1, activation='linear', name='time_output')(lstm_out)
    total_time_output = Dense(1, activation='linear', name='total_time_output')(lstm_out)

    model = Model(inputs=[
        node_input,
        event_source_input,
        street_input,
        category_input,
        work_order_type_input,
        secondary_feature_input,
        administrative_division_input,
        weekday_sin_input,
        weekday_cos_input,
        holiday_input,
        hour_input,
        month_sin_input,
        month_cos_input,
        delay_input,
        event_level_input
    ], outputs=[node_output, time_output, total_time_output])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss={
            'node_output': 'categorical_crossentropy',
            'time_output': 'mse',
            'total_time_output': 'mse'
        },
        loss_weights={
            'node_output': 0.5,
            'time_output': 0.5,
            'total_time_output': 1.0
        },
        optimizer=optimizer,
        metrics={
            'node_output': 'accuracy',
            'time_output': 'mse',
            'total_time_output': tf.keras.metrics.RootMeanSquaredError(name='root_mean_squared_error')
        }
    )

    # 定义 ModelCheckpoint 和 EarlyStopping 回调函数
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_total_time_output_root_mean_squared_error', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_total_time_output_root_mean_squared_error', patience=10, restore_best_weights=True)

    # 训练模型
    history = model.fit(
        [X_train_nodes, X_train_es, X_train_streets, X_train_categories, X_train_wo_types, X_train_sec_feats,
         X_train_admin_divs, X_train_weekdays_sin, X_train_weekdays_cos, X_train_holidays, X_train_hours,
         X_train_months_sin, X_train_months_cos, X_train_delays, X_train_event_levels],
        {'node_output': y_train_nodes, 'time_output': y_train_proc_times, 'total_time_output': y_train_total_times},
        epochs=300,
        batch_size=32,
        validation_data=([X_test_nodes, X_test_es, X_test_streets, X_test_categories, X_test_wo_types, X_test_sec_feats,
                          X_test_admin_divs, X_test_weekdays_sin, X_test_weekdays_cos, X_test_holidays, X_test_hours,
                          X_test_months_sin, X_test_months_cos, X_test_delays, X_test_event_levels],
                         {'node_output': y_test_nodes, 'time_output': y_test_proc_times, 'total_time_output': y_test_total_times}),
        callbacks=[checkpoint, early_stopping, PrintSavedValue(), PauseTrainingCallback()]
    )

# 保存模型和编码器
save_encoders()
