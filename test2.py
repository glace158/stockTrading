import torch
import torch.nn as nn
import numpy as np
from gym import spaces

# 1. 입력 NumPy 배열 정의
# 신경망은 보통 float 타입의 입력을 사용하므로 dtype=np.float32로 지정합니다.
input_np = np.array([1, 2, 3, 4], dtype=np.float32)
print(f"원본 NumPy 배열: {input_np}, 타입: {input_np.dtype}")

# 2. NumPy 배열을 PyTorch 텐서로 변환
# PyTorch 신경망은 PyTorch 텐서를 입력으로 받습니다.
input_tensor = torch.from_numpy(input_np)
print(f"입력 PyTorch 텐서: {input_tensor}, 형태: {input_tensor.shape}")

# 3. "아무것도 안 하는" 신경망 정의
# nn.Identity()는 입력을 그대로 반환하는 계층입니다.
# nn.Sequential을 사용하여 여러 계층을 순차적으로 구성할 수 있으며,
# 여기서는 Identity 계층 하나만 포함합니다.
class IdentityNetwork(nn.Module):
    """
        입력한 데이터를 아무 처리도 하지않고 반환 
    """
    def __init__(self, observation_space: spaces.Box):
        super().__init__()
        self.identity_layer = nn.Identity()

        # 전체 요소 수 계산
        self._features_dim = int(np.prod(observation_space.shape))
        
    def forward(self, x):
        return self.identity_layer(x)
    
    @property
    def features_dim(self):
        return self._features_dim

# 모델 인스턴스 생성
model = IdentityNetwork(input_np)

# 더 간단하게는 이렇게도 가능합니다:
# model = nn.Sequential(
#     nn.Identity()
# )

print("\n모델 구조:")
print(model)
print(model.features_dim)

# 4. 모델에 텐서 입력하여 결과 얻기
# model.eval()은 추론 모드로 설정합니다 (학습 시에는 model.train()).
# 여기서는 학습이 없으므로 큰 차이는 없지만 좋은 습관입니다.
model.eval()
# torch.no_grad()는 기울기 계산을 비활성화하여 메모리 사용량을 줄이고 계산 속도를 높입니다.
# 추론 시에는 항상 사용하는 것이 좋습니다.
with torch.no_grad():
    output_tensor = model(input_tensor)

print(f"\n모델 출력 (PyTorch 텐서): {output_tensor}, 형태: {output_tensor.shape}")

# 5. 결과 PyTorch 텐서를 NumPy 배열로 변환
output_np = output_tensor.numpy()
print(f"최종 NumPy 배열 출력: {output_np}, 타입: {output_np.dtype}")

# 6. 입력과 출력이 동일한지 확인
if np.array_equal(input_np, output_np):
    print("\n성공: 입력 NumPy 배열과 최종 출력 NumPy 배열이 동일합니다!")
else:
    print("\n실패: 입력과 출력이 다릅니다.")