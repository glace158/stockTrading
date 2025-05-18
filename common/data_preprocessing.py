import numpy as np

class VectorL2Normalizer:
    def get_preprocessing(self, data: np.ndarray) -> np.ndarray:
        """
        입력 데이터(벡터)를 L2 norm으로 정규화합니다.
        data: 1D numpy 배열
        """
        norm = np.linalg.norm(data)
        if norm == 0:
            return data
        
        return data / norm