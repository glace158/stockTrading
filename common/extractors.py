import torch
import torch.nn as nn
from gym import spaces
from typing import Union, Dict, List
import numpy as np

class CnnExtractor(nn.Module):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256): # SB3 CnnExtractor의 기본 features_dim은 512
        super().__init__()

        observation_space = self._set_image_dimensions(observation_space) # 이미지 데이터 형식 바꾸기 (H,W) -> (C,H,W)

        n_input_channels = observation_space.shape[0] # (C, H, W) 가정
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=1, padding=2), # Out: (B, 32, 60, 60)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Out: (B, 32, 30, 30)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Out: (B, 64, 30, 30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Out: (B, 64, 15, 15)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Out: (B, 128, 15, 15)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Out: (B, 128, 7, 7)
            nn.AdaptiveAvgPool2d((1, 1)), # Out: (B, 128, 1, 1)
            nn.Flatten(), # Out: (B, 128)
        )

        # CNN 출력 크기 계산
        with torch.no_grad():
            dummy_input = torch.as_tensor(observation_space.sample()[None]).float()
            cnn_out_dim = self.cnn(dummy_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(cnn_out_dim, features_dim), nn.ReLU())
        self._features_dim = features_dim

    def _set_image_dimensions(self, image_data : spaces.Box): 
        "이미지 데이터 형식 바꾸기 (H,W) -> (1,H,W)"
        if isinstance(image_data, spaces.Box) and len(image_data.shape) >= 2: # 이미지 또는 2D 이상 데이터
            
            # 만약 입력이 (H, W, C)라면, forward에서 transpose 필요 또는 여기서 shape 조정
            if len(image_data.shape) == 3 and image_data.shape[0] > image_data.shape[2] and image_data.shape[2] <=4 : # H,W,C일 가능성
                print(f"Warning: CnnExtractor input shape {image_data.shape} might be HWC. Assuming CHW for Conv2d.")
                # 실제로는 환경 래퍼에서 CHW로 바꾸는 것이 좋음
            
            # (H,W) -> (1,H,W) 처리
            if len(image_data.shape) == 2:
                img_obs_space = spaces.Box(low=np.expand_dims(image_data.low, axis=0),
                                            high=np.expand_dims(image_data.high, axis=0),
                                            shape=(1, *image_data.shape),
                                            dtype=image_data.dtype)
                return img_obs_space
            else: # (C,H,W)
                return image_data
        else:       
            raise ValueError(f"Unsupported Box observation space shape: {image_data.shape}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

    @property
    def features_dim(self):
        return self._features_dim

class MlpExtractor(nn.Module):
    """단일 벡터 입력을 위한 간단한 MLP 특징 추출기"""
    def __init__(self, observation_space: spaces.Box, features_dim: int = 32):
        super().__init__()
        self.flatten = nn.Flatten()
        input_dim = np.prod(observation_space.shape)
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, features_dim),
            nn.ReLU()
        )
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.flatten(observations))

    @property
    def features_dim(self):
        return self._features_dim

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

    
class CombinedFeaturesExtractor(nn.Module):
    def __init__(self, observation_space: spaces.Dict, cnn_features_dim: int = 256, mlp_features_dim: int = 32):
        super().__init__()
        extractors = {}
        total_features_dim = 0

        for key, subspace in observation_space.spaces.items(): # 상태 딕셔너리
            if isinstance(subspace, spaces.Box) and len(subspace.shape) >= 2: # 보통 이미지 (C,H,W) or (H,W) 등
                extractors[key] = CnnExtractor(subspace, features_dim=cnn_features_dim) # CNN 신경망 설정
                total_features_dim += extractors[key].features_dim

            elif isinstance(subspace, spaces.Box) and len(subspace.shape) == 1: # 수치형 벡터
                if mlp_features_dim != 0:
                    extractors[key] = MlpExtractor(subspace, features_dim=mlp_features_dim) # MLP 신경망 설정
                    total_features_dim += extractors[key].features_dim
                else:
                    total_features_dim += subspace.shape[0]

            else: # 기타 (예: Discrete) - 여기서는 Flatten 후 사용
                extractors[key] = nn.Flatten() 
                total_features_dim += np.prod(subspace.shape) # subspace 크기의 곱

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_features_dim
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded_tensor_list = []
        
        for key, observation in observations.items():
            #print(f"[{key}] observation shape: {observation.shape}")
            if key in self.extractors.keys():

                if observation.dim() == 3 and key == "img":  # (C,H,W)인 이미지에 배치 차원 추가
                    observation = observation.unsqueeze(0)
                elif observation.dim() == 1:  # 벡터 형태는 배치 차원 추가
                    observation = observation.unsqueeze(0)
                
                #print(f"[{key}] extractor: {self.extractors[key].__class__.__name__}")
                encoded = self.extractors[key](observation)
                #print(f"[{key}] encoded shape: {encoded.shape}")
                encoded_tensor_list.append(encoded)
            else:
                if observation.dim() == 1 and self.extractors.keys():  # (C,H,W)인 이미지에 배치 차원 추가
                    observation = observation.unsqueeze(0)
                    #print(f"[{key}] encoded shape: {observation.shape}")
                    encoded_tensor_list.append(observation)
                else:
                    encoded_tensor_list.append(observation)
                
                
        for t in encoded_tensor_list:
            #print(f"Encoded tensor shape for concat: {t.shape}")
            assert t.dim() == 2, "All tensors must be 2D for concat(dim=1)"
            
        return torch.cat(encoded_tensor_list, dim=1)

    @property
    def features_dim(self):
        return self._features_dim