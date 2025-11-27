from spin import get_n_qubit_op, build_hamiltonian, qrc_step, obs
import numpy as np
from sklearn.model_selection import train_test_split
from qutip import Qobj, basis, tensor, identity, sigmax, sigmaz, mesolve, qzero, expect
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

N = 3 

xt = np.load('xt.npy')
array_thermal = np.load('array_thermal.npy')
array_coherent = np.load('array_coherent.npy')
array_thermal_g2 = np.load('array_thermal_g2.npy')
array_coherent_g2 = np.load('array_coherent_g2.npy')

final_encoding_thermal = np.zeros((len(array_coherent_g2[0]),N*len(xt)))
final_encoding_coherent = np.zeros((len(array_coherent_g2[0]),N*len(xt)))


# final_encoding_thermal = np.zeros((len(array_coherent_g2[0]),N))
# final_encoding_coherent = np.zeros((len(array_coherent_g2[0]),N))

coherent_label = np.ones([len(array_coherent_g2[0]),1])
thermal_label = np.zeros([len(array_coherent_g2[0]),1])

# coherent_encoding = obs()

for i in range(len(array_coherent_g2[0])):
    final_encoding_thermal[i,:] = obs(array_thermal[i,:])
    final_encoding_coherent[i,:] = obs(array_coherent[i,:])
    
    
    
combined_x = np.concatenate((final_encoding_thermal, final_encoding_coherent), axis=0)    
combined_y = np.concatenate((thermal_label, coherent_label), axis=0)    



X_train, X_test, y_train, y_test = train_test_split(
    combined_x, combined_y, 
    test_size=0.2,       
    random_state=42,    
    stratify=combined_y        
)


model = Sequential([
    # Dense Layer가 곧 출력층입니다.
    # units=1: 이진 분류이므로 출력 노드는 1개입니다.
    # activation='sigmoid': 출력을 0과 1 사이의 확률로 변환합니다. (이진 분류의 핵심)
    # input_shape: 입력 데이터의 특징 개수를 지정합니다.
    Dense(units=1, activation='sigmoid', input_shape=(np.shape(combined_x)[1],))
])

# --- 3. 모델 컴파일 ---
model.compile(
    optimizer='adam',
    # 손실 함수: 이진 분류에 필수적인 Binary Crossentropy를 사용합니다.
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 요약 확인 (총 파라미터 수가 (10 * 1 + 1) = 11개임을 확인)
# model.summary()

# --- 4. 모델 학습 ---
print("\n--- 선형 모델 학습 시작 ---")
history = model.fit(
    X_train, y_train,
    epochs=5000,  # 단순 모델이므로 충분한 Epochs를 부여합니다.
    batch_size=32,
    validation_split=0.1, # 학습 데이터 중 10%를 검증에 사용
    verbose=1 #학습경과 출력
)

# --- 5. 모델 평가 및 예측 ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n테스트 정확도: {accuracy*100:.2f}%")

# 예측 (0~1 사이의 확률 값)
predictions_proba = model.predict(X_test, verbose=0)