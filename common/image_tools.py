import numpy as np
import pandas as pd
from typing import List
from common.data_visualization import *

def get_multiple_time_series_images(image_format: str, target_column_list : List[str], df : pd.DataFrame):
    """
        한 번에 N개의 칼럼을 받아 (N, H, W) 형식으로 시계열 이미지를 변환해줍니다.
        시계열 이미지 형식(GASF, GADF, RP, MTF, STFT, CWT)을 입력으로 받아
        해당하는 형식의 이미지 데이터들을 반환받습니다.

    Args:
        image_format(str): 변환할 이미지 데이터 형식입니다.
                                    GASF, GADF, RP, MTF, STFT, CWT
        target_column_list(List[str]): 변환할 칼럼들을 지정해 줍니다.

        df(pd.DataFrame): 변환에 사용할 데이터를 지정해줍니다.

    Returns:
        np.ndarray: image_format에 해당하는 이미지 데이터입니다.

    Raises:
        ValueError: 잘못된 이미지 형식을 입력하는 경우 발생합니다.
    """
    time_series_image_list = []
    for target_column in target_column_list:
        
        time_series_image = get_time_series_image(image_format, target_column, df)
        time_series_image = add_batch_channel_dim(time_series_image) # 
        time_series_image_list.append(time_series_image)

    combine_image = combine_multiple_images(time_series_image_list)
    return combine_image

def get_time_series_image(image_format: str, target_column : str, df : pd.DataFrame):
    """
        시계열 이미지 형식(GASF, GADF, RP, MTF, STFT, CWT)을 입력으로 받아
        해당하는 형식의 이미지 데이터를 반환받습니다.

    Args:
        image_format(str): 변환할 이미지 데이터 형식입니다.
                                    GASF, GADF, RP, MTF, STFT, CWT
        target_column(str): 변환할 칼럼을 지정해줍니다.

        df(pd.DataFrame): 변환에 사용할 데이터를 지정해줍니다.

    Returns:
        np.ndarray: image_format에 해당하는 이미지 데이터입니다.

    Raises:
        ValueError: 잘못된 이미지 형식을 입력하는 경우 발생합니다.
    """
    
    data_processor = TimeSeriesDataGenerator(feature_range=(0, 1))
    # DataFrame과 컬럼 이름을 전달하여 처리
    processed_series, np_ts_pyts, np_ts_1d, current_time_steps = data_processor.process_dataframe_column(
        df, target_column
    )

    if "GASF":
        gaf_converter = GramianAngularFieldConverter(image_size=current_time_steps)
        gasf_image, gadf_image = gaf_converter.transform(np_ts_pyts)
        #print(f"GASF image shape: {gasf_image.shape}") # (1, image_size, image_size)
        return gadf_image
    elif "GADF":
        gaf_converter = GramianAngularFieldConverter(image_size=current_time_steps)
        gasf_image, gadf_image = gaf_converter.transform(np_ts_pyts)
        #print(f"GADF image shape: {gadf_image.shape}") # (1, image_size, image_size)
        return gadf_image
    elif "RP":
        rp_converter = RecurrencePlotConverter(threshold=0.15)
        rp_image = rp_converter.transform(np_ts_pyts)
        #print(f"Recurrence Plot image shape: {rp_image.shape}") # (1, time_steps, time_steps)
        return rp_image
    elif "MTF":
        mtf_converter = MarkovTransitionFieldConverter(n_bins=10)
        mtf_image = mtf_converter.transform(np_ts_pyts)
        #print(f"Markov Transition Field image shape: {mtf_image.shape}") # (1, n_bins, n_bins)
        return mtf_image
    elif "STFT":
        spectrogram_converter = SpectrogramConverter(fs=1.0, nperseg_ratio=0.25)
        spec_frequencies, spec_times, spec_db_image = spectrogram_converter.transform(np_ts_1d)
        #print(f"Spectrogram (dB) image shape: {spec_db_image.shape}") # (num_frequencies, num_time_bins)
        return spec_db_image
    elif "CWT":
        scalogram_converter = ScalogramConverter(wavelet_name='morl', max_scale_ratio=0.5, fs=1.0)
        scalogram_mag_image, scal_scales_arr, scal_freq_arr, scal_w_name = scalogram_converter.transform(np_ts_1d)
        #print(f"Scalogram (magnitude) image shape: {scalogram_mag_image.shape}") # (num_scales, time_steps)
        return scalogram_mag_image
    else:
        raise ValueError(f"잘못된 포맷입니다. 사용가능한 포맷 : GASF, GADF, RP, MTF, STFT, CWT / 현재 포맷: {image_format}")

def combine_multiple_images(image_list: List[np.ndarray]) -> np.ndarray:
    """
    Shape (1, H, W)를 갖는 여러 이미지 배열들을 리스트로 받아, 
    첫 번째 축을 따라 결합하여 Shape (N, H, W)의 단일 이미지 배열로 만듭니다.
    N은 리스트에 있는 이미지의 개수입니다.

    Args:
        image_list (List[np.ndarray]): Shape (1, H, W)를 갖는 이미지 배열들의 리스트.
                                      리스트 내 모든 이미지의 H와 W는 동일해야 합니다.

    Returns:
        np.ndarray: 결합된 이미지 배열. Shape는 (N, H, W)입니다.

    Raises:
        ValueError: 입력 이미지 리스트가 비어 있거나, 이미지들의 shape 또는
                    H, W 크기가 적절하지 않은 경우 발생합니다.
    """
    if not image_list:
        raise ValueError("입력 이미지 리스트가 비어 있습니다.")

    # 첫 번째 이미지를 기준으로 shape 검증 및 H, W 크기 저장
    first_image_shape = image_list[0].shape
    if not (len(first_image_shape) == 3 and first_image_shape[0] == 1):
        raise ValueError(f"리스트의 첫 번째 이미지 shape는 (1, H, W)여야 합니다. 현재 shape: {first_image_shape}")
    
    expected_h = first_image_shape[1]
    expected_w = first_image_shape[2]

    # 리스트 내 다른 이미지들의 shape 및 H, W 크기 검증
    for i, image in enumerate(image_list):
        if not (image.ndim == 3 and image.shape[0] == 1):
            raise ValueError(f"리스트의 {i+1}번째 이미지 shape는 (1, H, W)여야 합니다. 현재 shape: {image.shape}")
        if image.shape[1] != expected_h or image.shape[2] != expected_w:
            raise ValueError(f"리스트의 모든 이미지의 높이(H)와 너비(W)가 일치해야 합니다. "
                             f"첫 번째 이미지 HxW: {expected_h}x{expected_w}, "
                             f"{i+1}번째 이미지 HxW: {image.shape[1]}x{image.shape[2]}")

    # 첫 번째 축(axis=0)을 따라 모든 이미지를 연결합니다.
    # image_list에 있는 배열들이 concatenate의 입력으로 바로 사용됩니다.
    
    squeezed_images = [img.squeeze(axis=0) for img in image_list] # 각 이미지를 (H,W)로 만듦
    combined_image = np.stack(squeezed_images, axis=0)
    #combined_image = np.concatenate(image_list, axis=0)
    
    return combined_image

def add_batch_channel_dim(image_2d: np.ndarray) -> np.ndarray:
    """
    Shape (H, W)를 갖는 2D 이미지 배열에 첫 번째 축을 추가하여
    Shape (1, H, W)의 3D 이미지 배열로 변환합니다.
    이는 단일 이미지를 배치 크기 1 또는 채널 수 1로 간주할 때 사용됩니다.

    Args:
        image_2d (np.ndarray): 2D 이미지 배열. Shape는 (H, W)여야 합니다.

    Returns:
        np.ndarray: 변환된 3D 이미지 배열. Shape는 (1, H, W)입니다.
    """
    if image_2d.ndim == 2:
        image_3d = np.expand_dims(image_2d, axis=0)
        return image_3d
    elif image_2d.ndim == 3: # 이미 3D 이미지 배열이면 
        return image_2d # 그대로 반환
    else:
        raise ValueError(f"입력 이미지의 차원이 잘못되었습니다. 현재 차원: {image_2d.ndim}")
    
if __name__ == '__main__':
    # 테스트용 샘플 이미지 생성
    num_images_to_combine = 5
    H, W = 128, 128
    
    list_of_images = []
    for i in range(num_images_to_combine):
        # 각 이미지는 (1, H, W) shape를 가집니다.
        # 값은 예시를 위해 간단히 i 값으로 채웁니다.
        img = np.full((1, H, W), fill_value=i, dtype=np.uint8)
        list_of_images.append(img)
        print(f"Shape of img_{i+1}: {img.shape}")

