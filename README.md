# 악성 채팅 분류 모델

## 주제 선정 이유
- 악성 채팅 문제가 최근 온라인 상에서 많은 문제가 되고 있음
- 채팅 검열 시스템의 잘못된 판단으로 정상 유저들을 악성 유저로 판단하는 문제가 종종 발생
- 딥러닝을 통하여 악성채팅을 정확하게 분류하고자 함


## 프로젝트 방법
### 1. Data Preprocessing
#### 1-1. Data Labeling, Sampling
- 분류 모델 사용을 위해 Target값을 0.5를 기준으로 Toxic label 할당 및 Sampling을 통해 데이터 불균형 문제 해소
```python
data_df = data[['id','comment_text','target']]

# set index
data_df.set_index('id', inplace=True)

# y_label
data_df['label'] = np.where(data_df['target'] >= 0.5, 1, 0) # Label 1 >= 0.5 / Label 0 < 0.5
#train_df.drop(['target'], axis=1, inplace=True)
```
```python
# 언더샘플링 이용해서 불균형 문제 해소

# toxic nontoxic 분류
toxic = data_df[data_df['label']==1]
nontoxic = data_df[data_df['label']==0]

# nontoxic 에서 toxic 만큼 샘플링
nontoxic = nontoxic.sample(n=len(toxic),random_state =1018)
```
#### 1-2. Remove Stopwords, Punctuations, Emojis
- 모델 학습 속도와 정확도를 위해 Stopword(are, is ...)와 Punctuation(!, ?, ..) 이모지 제거함수 정의
```python
## Clean Punctuation & Stopwords
class clean_text:
	def __init__(self, text):
		self.text = text
	
	# Remove Punctuation
	def rm_punct(text):
		punct = set([p for p in "…/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'])
		text = [t for t in text if t not in punct]
			
		return "".join(text)

	# Remove Stopwords
	def rm_stopwords(text):
		word_tokens = word_tokenize(text)   
		result = [w for w in word_tokens if w not in stop_words]
				
		return " ".join(result)
## Clean Emoji
# 이모지 제거 함수 정의
import re

text_test = u'This dog \U0001f602'
print(text_test) # with emoji

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
```
![image](https://github.com/ShinWooHyeon/Unintended_Bias_in_Toxicity_Classification/assets/118239192/340dc481-e1ca-4317-bec0-ee7f023063d6)
![image](https://github.com/ShinWooHyeon/Unintended_Bias_in_Toxicity_Classification/assets/118239192/705df1d5-62e0-495d-947f-d30b05ec4b84)

#### 1-3. Data Tokenizing, Sequencing, Padding
- 모델 학습을 위해선 문장을 단어별로 Tokenizing 한 후 Sequence Data로 만들어야 
- 각 Sequence Data가 같은 길이를 가져야 학습 가능하므로 Padding해서 길이를 맞춰준다
```python
## tokenize
max_words = 100000
tokenizer = text.Tokenizer(num_words=max_words) # Tokenizer 객체생성
tokenizer.fit_on_texts(X_train)	# 토큰 별 word index 생성

# texts_to_sequences
sequences_text_train = tokenizer.texts_to_sequences(X_train)
sequences_text_test = tokenizer.texts_to_sequences(X_test)

# padding
max_len = max(len(l) for l in sequences_text_train)
pad_train = pad_sequences(sequences_text_train, maxlen=max_len)
pad_test = pad_sequences(sequences_text_test, maxlen=max_len)
```
- Padding 까지 한 후의 Data는 다음과 같이 sequence 형태의 길이가 같은 형태를 가진다
![image](https://github.com/ShinWooHyeon/Unintended_Bias_in_Toxicity_Classification/assets/118239192/23b655a3-53e8-4d36-bf9c-502fd8a39442)

### 2. Modeling
- 양방향 학습을 하는 Bidirectional LSTM Layer를 사용하여 성능을 일반 LSTM Layer 보다 높임
- Max Pooling과 Average Pooling 모두 사용
- Batch Normalization Layer를 사용하여 학습속도를 높임
- Overfitting 막기 위해 ReduceLROnPlateau, Early Stopping 사용
```python
def Embedding_CuDNNLSTM_model(max_words, max_len):
	sequence_input = layers.Input(shape=(None, ))
	x = layers.Embedding(max_words, 128, input_length=max_len)(sequence_input)
	x = layers.SpatialDropout1D(0.4)(x)
	x = layers.Bidirectional(layers.CuDNNLSTM(64, return_sequences=True))(x)
	x = layers.Bidirectional(layers.CuDNNLSTM(64, return_sequences=True))(x)
	
	avg_pool1d = layers.GlobalAveragePooling1D()(x)
	max_pool1d = layers.GlobalMaxPool1D()(x)
	
	x = layers.concatenate([avg_pool1d, max_pool1d])
	x = layers.Dense(32, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	output = layers.Dense(1)(x)
	
	model = models.Model(sequence_input, output)
	
	return model
## embedding_lstm models 
model = Embedding_CuDNNLSTM_model(max_words, max_len)
# model compile
adam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam,
			 loss='mse')
```
```python
# keras.callbacks
callbacks_list = [
		ReduceLROnPlateau(
			monitor='val_loss', patience=2, factor=0.1, mode='max'),	# val_loss가 patience동안 향상되지 않으면 학습률을 0.1만큼 감소 (new_lr = lr * factor)
		EarlyStopping(
			patience=5, monitor='val_loss', mode='max', restore_best_weights=True),
		ModelCheckpoint(
			filepath='/content/drive/MyDrive/2022-2학기/딥러닝/jigsaw classfication/best_embedding_lstm_model_ver1.h5', monitor='val_loss', mode='max', save_best_only=True)
]

# model fit & save
model_path = '/content/drive/MyDrive/2022-2학기/딥러닝/jigsaw classfication/best_embedding_lstm_model_ver1.h5'
if os.path.exists(model_path):
	model.load_weights(model_path)
else:
	history = model.fit(pad_train, Y_train,
						epochs=7, batch_size=256,
						callbacks=callbacks_list, 
						validation_split=0.3, verbose=1)
```

### 3. Hyper Parmeter Tuning
- 각 Hyper Parameter의 후보군들에 대해적용해 최적의 결과값을 내는 모델 선택
![image](https://github.com/ShinWooHyeon/Unintended_Bias_in_Toxicity_Classification/assets/118239192/9a1d8b03-eb6f-4b6e-8888-6effae76d689)

### 4. Result
![image](https://github.com/ShinWooHyeon/Unintended_Bias_in_Toxicity_Classification/assets/118239192/a07d9592-446d-4a84-8225-3a584b2a0ed6)
- 다음과 같은 학습 그래프를 보였으 5 epoch의 weight를 선택

![image](https://github.com/ShinWooHyeon/Unintended_Bias_in_Toxicity_Classification/assets/118239192/5d3ff40f-7b96-4800-903b-7ed50b8ad1fe)
- 0.874의 AUROC 성능지표를 가지는 최종 채팅 분류 모델 생성
- Hyper Parameter 튜닝 등을 통해 효과적으로 모델 성능을 향상 시킬 수 있었음

## Data Source
Kaggle: https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data
 
