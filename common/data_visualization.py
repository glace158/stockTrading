import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField, RecurrencePlot, MarkovTransitionField
from scipy.signal import stft
import pywt

class TimeSeriesDataGenerator:
    """
    외부 Pandas DataFrame과 컬럼 이름을 입력받아 해당 컬럼을 시계열 데이터로 처리하고,
    관련 속성들을 설정한 후 그 값들을 반환하는 클래스.
    """
    def __init__(self, feature_range=(0, 1)):
        """
        :param feature_range: 스케일링 시 사용할 값의 범위
        """
        self.feature_range = feature_range
        self.original_series_scaled = None # 스케일링된 Pandas Series
        self.X_ts_pyts = None             # pyts 라이브러리 입력용 NumPy 배열 (N, L)
        self.X_ts_1d = None               # scipy, pywt 입력용 NumPy 배열 (L,)
        self.time_steps = 0               # 처리된 시계열의 길이 (L)
        self.processed_column_name = None # 처리된 컬럼의 이름

    def process_dataframe_column(self, dataframe: pd.DataFrame, column_name: str):
        """
        입력 DataFrame에서 지정된 컬럼을 추출하고 스케일링한 후, 클래스 속성을 설정하고 값들을 반환합니다.
        
        :param dataframe: 입력 Pandas DataFrame
        :param column_name: 사용할 시계열 데이터가 있는 컬럼 이름
        :return: Tuple (self.original_series_scaled, self.X_ts_pyts, self.X_ts_1d, self.time_steps)
                 또는 오류 발생 시 (None, None, None, 0)
        :raises ValueError: 컬럼이 존재하지 않거나 비어있는 경우
        """
        if column_name not in dataframe.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

        # 지정된 컬럼을 Pandas Series로 추출
        original_series = dataframe[column_name].copy()
        
        if original_series.empty:
            raise ValueError(f"The selected column '{column_name}' is empty.")

        self.time_steps = len(original_series)
        self.processed_column_name = column_name

        scaler = MinMaxScaler(feature_range=self.feature_range)
        # MinMaxScaler는 NumPy 배열을 반환하므로, 다시 Series로 만듭니다.
        # Series의 인덱스를 유지합니다.
        scaled_values = scaler.fit_transform(original_series.values.reshape(-1, 1)).flatten()
        self.original_series_scaled = pd.Series(scaled_values, index=original_series.index, name=f"Scaled_{column_name}")

        # NumPy 배열 형태도 준비
        # pyts는 (n_samples, n_timesteps)를 기대하므로, 단일 시계열이라도 2D로 만듦
        self.X_ts_pyts = self.original_series_scaled.values.reshape(1, -1) 
        self.X_ts_1d = self.original_series_scaled.values # 1D 형태
        
        return self.original_series_scaled, self.X_ts_pyts, self.X_ts_1d, self.time_steps

class GramianAngularFieldConverter:
    """Gramian Angular Fields (GASF, GADF) 이미지를 생성하는 클래스."""
    def __init__(self, image_size):
        self.image_size = image_size

    def transform(self, time_series_pyts_input):
        gasf_transformer = GramianAngularField(image_size=self.image_size, method='summation')
        X_gasf = gasf_transformer.fit_transform(time_series_pyts_input)
        gadf_transformer = GramianAngularField(image_size=self.image_size, method='difference')
        X_gadf = gadf_transformer.fit_transform(time_series_pyts_input)
        return X_gasf, X_gadf

class RecurrencePlotConverter:
    """Recurrence Plot (RP) 이미지를 생성하는 클래스."""
    def __init__(self, threshold=0.15):
        self.threshold = threshold

    def transform(self, time_series_pyts_input):
        rp_transformer = RecurrencePlot(threshold=self.threshold)
        X_rp = rp_transformer.fit_transform(time_series_pyts_input)
        return X_rp

class MarkovTransitionFieldConverter:
    """Markov Transition Fields (MTF) 이미지를 생성하는 클래스."""
    def __init__(self, n_bins=10, strategy='quantile'):
        self.n_bins = n_bins
        self.strategy = strategy

    def transform(self, time_series_pyts_input):
        mtf_transformer = MarkovTransitionField(n_bins=self.n_bins, strategy=self.strategy)
        X_mtf = mtf_transformer.fit_transform(time_series_pyts_input)
        return X_mtf

class SpectrogramConverter:
    """STFT를 사용하여 스펙트로그램을 생성하는 클래스."""
    def __init__(self, fs=1.0, nperseg_ratio=0.25, noverlap_ratio=0.5):
        self.fs = fs
        self.nperseg_ratio = nperseg_ratio
        self.noverlap_ratio = noverlap_ratio

    def transform(self, time_series_1d_input):
        if len(time_series_1d_input) == 0:
            return np.array([]), np.array([]), np.array([])
            
        nperseg = int(len(time_series_1d_input) * self.nperseg_ratio)
        if nperseg == 0: nperseg = max(1, int(len(time_series_1d_input) * 0.1)) 
        if nperseg > len(time_series_1d_input): nperseg = len(time_series_1d_input)
        
        noverlap = int(nperseg * self.noverlap_ratio)
        if noverlap >= nperseg and nperseg > 0 : noverlap = nperseg -1

        nfft = nperseg

        if len(time_series_1d_input) < nperseg:
             nperseg = len(time_series_1d_input)
             noverlap = 0 

        frequencies, times, Zxx = stft(time_series_1d_input, fs=self.fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        spectrogram_magnitude = np.abs(Zxx)
        spectrogram_db = 20 * np.log10(spectrogram_magnitude + 1e-9)
        return frequencies, times, spectrogram_db

class ScalogramConverter:
    """CWT를 사용하여 웨이블릿 스케일로그램을 생성하는 클래스."""
    def __init__(self, wavelet_name='morl', max_scale_ratio=0.5, fs=1.0):
        self.wavelet_name = wavelet_name
        self.max_scale_ratio = max_scale_ratio
        self.fs = fs

    def transform(self, time_series_1d_input):
        time_steps_len = len(time_series_1d_input)
        if time_steps_len == 0:
            return np.array([]), np.array([]), np.array([]), self.wavelet_name

        max_scale = int(time_steps_len * self.max_scale_ratio)
        if max_scale < 1: max_scale = 1
        scales = np.arange(1, max_scale + 1)
        
        coefficients, frequencies = pywt.cwt(time_series_1d_input, scales, self.wavelet_name, sampling_period=1.0/self.fs)
        scalogram_magnitude = np.abs(coefficients)
        return scalogram_magnitude, scales, frequencies, self.wavelet_name

class TimeSeriesVisualizer:
    """시계열 데이터와 변환된 이미지들을 시각화하는 클래스."""
    def plot_all(self, original_series, 
                   time_steps,
                   column_name_for_title: str, # 컬럼 이름 추가
                   gasf_img, gadf_img, rp_img, mtf_img, mtf_n_bins,
                   spec_data, scal_data):
        fig, axs = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle(f"Various Time Series Imaging Techniques for Column: '{column_name_for_title}'", fontsize=16)

        # 1. Original Time Series
        if original_series is not None and not original_series.empty:
            original_series.plot(ax=axs[0, 0], grid=True)
            axs[0, 0].set_title(f"Original Time Series ('{original_series.name}', L={time_steps})")
        else:
            axs[0,0].text(0.5, 0.5, "No data", ha='center', va='center')
            axs[0,0].set_title("Original Time Series")

        # 2. GASF
        if gasf_img is not None and gasf_img.size > 0:
            img_gasf = axs[0, 1].imshow(gasf_img[0], cmap='rainbow', origin='lower')
            axs[0, 1].set_title(f"GASF Image {gasf_img.shape}")
            fig.colorbar(img_gasf, ax=axs[0, 1], fraction=0.046, pad=0.04)
        else:
            axs[0,1].text(0.5, 0.5, "No GASF data", ha='center', va='center')
            axs[0,1].set_title("GASF Image")
        # ... (이하 다른 이미지 플롯들도 유사하게 None 및 size 체크 추가)
        # 3. GADF
        if gadf_img is not None and gadf_img.size > 0:
            img_gadf = axs[0, 2].imshow(gadf_img[0], cmap='rainbow', origin='lower')
            axs[0, 2].set_title(f"GADF Image {gadf_img.shape}")
            fig.colorbar(img_gadf, ax=axs[0, 2], fraction=0.046, pad=0.04)
        else:
            axs[0,2].text(0.5, 0.5, "No GADF data", ha='center', va='center')
            axs[0,2].set_title("GADF Image")

        # 4. Recurrence Plot
        if rp_img is not None and rp_img.size > 0:
            img_rp = axs[1, 0].imshow(rp_img[0], cmap='binary', origin='lower')
            axs[1, 0].set_title(f"Recurrence Plot {rp_img.shape}")
            fig.colorbar(img_rp, ax=axs[1, 0], fraction=0.046, pad=0.04)
        else:
            axs[1,0].text(0.5, 0.5, "No RP data", ha='center', va='center')
            axs[1,0].set_title("Recurrence Plot")

        # 5. Markov Transition Field
        if mtf_img is not None and mtf_img.size > 0:
            img_mtf = axs[1, 1].imshow(mtf_img[0], cmap='viridis', origin='lower', extent=(0, mtf_n_bins, 0, mtf_n_bins))
            axs[1, 1].set_title(f"MTF ({mtf_n_bins}x{mtf_n_bins} bins) {mtf_img.shape}")
            axs[1, 1].set_xlabel("Quantile bin at t+1")
            axs[1, 1].set_ylabel("Quantile bin at t")
            fig.colorbar(img_mtf, ax=axs[1, 1], fraction=0.046, pad=0.04)
        else:
            axs[1,1].text(0.5, 0.5, "No MTF data", ha='center', va='center')
            axs[1,1].set_title("Markov Transition Field")

        # 6. Spectrogram (STFT)
        spec_freq, spec_time, spec_db = spec_data
        if spec_db is not None and spec_db.size > 0:
            img_spec = axs[1, 2].pcolormesh(spec_time, spec_freq, spec_db, shading='gouraud', cmap='viridis')
            axs[1, 2].set_ylabel('Frequency [Hz]')
            axs[1, 2].set_xlabel('Time [sec]')
            axs[1, 2].set_title(f'Spectrogram {spec_db.shape}')
            fig.colorbar(img_spec, ax=axs[1, 2], fraction=0.046, pad=0.04, label='Intensity [dB]')
        else:
            axs[1,2].text(0.5, 0.5, "No Spectrogram data", ha='center', va='center')
            axs[1,2].set_title("Spectrogram (STFT)")

        # 7. Wavelet Scalogram (CWT)
        scal_mag, scal_scales, _, scal_wavelet_name = scal_data 
        if scal_mag is not None and scal_mag.size > 0 and scal_scales is not None and scal_scales.size > 0:
            max_scale_val = scal_scales[-1]
            img_scal = axs[2, 0].imshow(scal_mag, extent=[0, time_steps, max_scale_val, 1], cmap='viridis', aspect='auto', interpolation='bilinear')
            if len(scal_scales) > 0:
                 axs[2,0].set_yticks(np.linspace(1, max_scale_val, min(5, len(scal_scales)), dtype=int if max_scale_val >=1 else float))
            axs[2, 0].set_ylabel('Scale (decreasing frequency)')
            axs[2, 0].set_xlabel('Time [samples]')
            axs[2, 0].set_title(f'Scalogram ({scal_wavelet_name}) {scal_mag.shape}')
            fig.colorbar(img_scal, ax=axs[2, 0], fraction=0.046, pad=0.04, label='Magnitude')
        else:
            axs[2,0].text(0.5, 0.5, "No Scalogram data", ha='center', va='center')
            axs[2,0].set_title("Wavelet Scalogram (CWT)")

        axs[2, 1].axis('off')
        axs[2, 2].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

if __name__ == '__main__':
    # --- 샘플 DataFrame 생성 ---
    num_points = 150
    date_rng = pd.date_range(start='2023-01-01', periods=num_points, freq='D')
    sample_df = pd.DataFrame(date_rng, columns=['Date'])
    sample_df['Open'] = np.random.rand(num_points) * 50 + 100
    t_main = np.linspace(0, 20, num_points)
    sample_df['Close'] = np.sin(t_main * 1.5) + np.cos(t_main * 0.8) + np.random.normal(0, 1.5, num_points) + 110
    sample_df.set_index('Date', inplace=True)

    print("--- Sample DataFrame Head ---")
    print(sample_df.head())
    print("\n")

    # --- 데이터 처리기 인스턴스화 ---
    data_processor = TimeSeriesDataGenerator(feature_range=(0, 1))
    target_column_to_process = 'Close' # 처리할 단일 컬럼 지정

    try:
        # DataFrame과 컬럼 이름을 전달하여 처리
        processed_series, np_ts_pyts, np_ts_1d, current_time_steps = data_processor.process_dataframe_column(
            sample_df, target_column_to_process
        )

        print(f"--- Processed Data for Column: '{data_processor.processed_column_name}' ---")
        print(f"Original Scaled Pandas Series shape (length): {processed_series.shape[0]}")
        print(f"NumPy array for pyts (X_ts_pyts) shape: {np_ts_pyts.shape}")
        print(f"NumPy 1D array (X_ts_1d) shape: {np_ts_1d.shape}")
        print(f"Time steps: {current_time_steps}")
        print("-" * 30)

        # --- 이미지 변환 ---
        # 클래스 속성에 저장된 값 또는 반환된 값을 사용할 수 있음
        # 여기서는 반환된 값을 사용
        print("--- Transformed Image Shapes ---")
        gasf_image, gadf_image, rp_image, mtf_image = None, None, None, None
        spectrogram_data = (np.array([]), np.array([]), np.array([]))
        scalogram_data = (np.array([]), np.array([]), np.array([]), '')

        if current_time_steps >= 20: # 최소 길이 체크
            gaf_converter = GramianAngularFieldConverter(image_size=current_time_steps)
            gasf_image, gadf_image = gaf_converter.transform(np_ts_pyts)
            print(f"GASF image shape: {gasf_image.shape}")
            print(f"GADF image shape: {gadf_image.shape}")

            rp_converter = RecurrencePlotConverter(threshold=0.15)
            rp_image = rp_converter.transform(np_ts_pyts)
            print(f"Recurrence Plot image shape: {rp_image.shape}")

            mtf_converter = MarkovTransitionFieldConverter(n_bins=10)
            mtf_image = mtf_converter.transform(np_ts_pyts)
            print(f"Markov Transition Field image shape: {mtf_image.shape}")

            spectrogram_converter = SpectrogramConverter(fs=1.0, nperseg_ratio=0.25)
            spec_frequencies, spec_times, spec_db_image = spectrogram_converter.transform(np_ts_1d)
            print(f"Spectrogram (dB) image shape: {spec_db_image.shape}")
            spectrogram_data = (spec_frequencies, spec_times, spec_db_image)

            scalogram_converter = ScalogramConverter(wavelet_name='morl', max_scale_ratio=0.5, fs=1.0)
            scalogram_mag_image, scal_scales_arr, scal_freq_arr, scal_w_name = scalogram_converter.transform(np_ts_1d)
            print(f"Scalogram (magnitude) image shape: {scalogram_mag_image.shape}")
            scalogram_data = (scalogram_mag_image, scal_scales_arr, scal_freq_arr, scal_w_name)
        else:
            print(f"Warning: Time series length ({current_time_steps}) is too short for some transformations.")
        print("-" * 30)
        
        # --- 시각화 ---
        visualizer = TimeSeriesVisualizer()
        visualizer.plot_all(
            original_series=processed_series, # 또는 data_processor.original_series_scaled
            time_steps=current_time_steps,    # 또는 data_processor.time_steps
            column_name_for_title=data_processor.processed_column_name, # 클래스 속성 사용
            gasf_img=gasf_image,
            gadf_img=gadf_image,
            rp_img=rp_image,
            mtf_img=mtf_image,
            mtf_n_bins=mtf_converter.n_bins if 'mtf_converter' in locals() and current_time_steps >=20 else 10,
            spec_data=spectrogram_data,
            scal_data=scalogram_data
        )

    except ValueError as e:
        print(f"Error during data processing: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()