import pandas as pd
import numpy as np
import yaml
import logging
import pandas_ta as ta

class FeatureEngineer:
    def __init__(self, config_path):
        # load configuration.
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)
            self.config = full_config.get('feature_engineering', {}) # Get FE specific section
            if not self.config: logging.warning(f"FeatureEngineer settings not found in {config_path}. Using defaults or hardcoded logic.")
        self.scaler_params = {}
        self.col_names_for_scaler = None
        self.indicator_configs = self.config.get('indicators', [])
        logging.info("FeatureEngineer initialized.")

    def calculate_native_features(self, df: pd.DataFrame, timeframe_suffix: str):
        if df.empty: return df
        df_feat = df.copy()

        # essential OHLCV columns
        used_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in used_cols:
            if col not in df_feat.columns:
                logging.warning(f"Column {col} not found in DataFrame for {timeframe_suffix}. Imputing with NaN.")
                df_feat[col] = np.nan 
            df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce')
            
            if col == 'Volume': # handle inf in Volume before dropna
                if df_feat[col].hasnans or np.isinf(df_feat[col].values).any():
                    df_feat[col].replace([np.inf, -np.inf], np.nan, inplace=True)

        df_feat.dropna(subset=used_cols, how='any', inplace=True)
        if df_feat.empty:
            logging.warning(f"df for {timeframe_suffix} is empty after dropping NaNs in OHLCV, return empty df.")
            return df_feat

        df_feat[f'SMA_10_{timeframe_suffix}'] = df_feat.ta.sma(length=10, close='Close', append=False)
        df_feat[f'SMA_30_{timeframe_suffix}'] = df_feat.ta.sma(length=30, close='Close', append=False)
        df_feat[f'EMA_10_{timeframe_suffix}'] = df_feat.ta.ema(length=10, close='Close', append=False)
        df_feat[f'EMA_30_{timeframe_suffix}'] = df_feat.ta.ema(length=30, close='Close', append=False)
        df_feat[f'RSI_14_{timeframe_suffix}'] = df_feat.ta.rsi(length=14, close='Close', append=False)
        df_feat[f'ATR_14_{timeframe_suffix}'] = df_feat.ta.atr(length=14, high='High', low='Low', close='Close', append=False)
        
        bbands = df_feat.ta.bbands(length=20, close='Close', append=False)
        if bbands is not None and not bbands.empty and bbands.shape[1] >= 5:
            # .iloc[:, 4] for BBP (%B) and .iloc[:, 3] for BBB (Bandwidth) assumes fixed column order from pandas-ta
            df_feat[f'BBP_20_{timeframe_suffix}'] = bbands.iloc[:, 4] 
            df_feat[f'BBW_20_{timeframe_suffix}'] = bbands.iloc[:, 3] 
        else: # just in case calculation failed
            df_feat[f'BBP_20_{timeframe_suffix}'] = np.nan
            df_feat[f'BBW_20_{timeframe_suffix}'] = np.nan
        
        if 'Volume' in df_feat.columns and not df_feat['Volume'].empty:
            df_feat[f'Vol_SMA_20_{timeframe_suffix}'] = df_feat.ta.sma(length=20, close='Volume', append=False)
            vol_change_col = f'Vol_Change_1_{timeframe_suffix}'

            # (1) pct_change will produce NaN for the first row or if previous Volume was NaN (after cleaning).
            # fix(2) 'Volume' related indicators as some results in inf as Volume = 0
            df_feat[vol_change_col] = df_feat['Volume'].pct_change(periods=1)
            if df_feat[vol_change_col].hasnans or np.isinf(df_feat[vol_change_col].values).any():
                df_feat[vol_change_col] = df_feat[vol_change_col].replace([np.inf, -np.inf], np.nan)
        else:
            df_feat[vol_change_col] = np.nan
        
        # rate of change
        df_feat[f'ROC_1_{timeframe_suffix}'] = df_feat.ta.roc(length=1, close='Close', append=False)
        df_feat[f'ROC_5_{timeframe_suffix}'] = df_feat.ta.roc(length=5, close='Close', append=False)
        
        # candle-based features
        df_feat[f'CandleRange_{timeframe_suffix}'] = df_feat['High'] - df_feat['Low']
        df_feat[f'BodySize_{timeframe_suffix}'] = (df_feat['Close'] - df_feat['Open']).abs()
        
        candle_range_col_name = f'CandleRange_{timeframe_suffix}'
        close_pos_col_name = f'ClosePosInRange_{timeframe_suffix}'
        if candle_range_col_name in df_feat and not df_feat[candle_range_col_name].isnull().all():
            # replace 0 to avoid division by zero
            df_feat[close_pos_col_name] = (df_feat['Close'] - df_feat['Low']) / (df_feat[candle_range_col_name].replace(0, np.nan))
            # fill the NaN with .5
            df_feat[close_pos_col_name] = df_feat[close_pos_col_name].fillna(.5)
        else:
            df_feat[close_pos_col_name] = .5
            
        feature_cols = [col for col in df_feat.columns if timeframe_suffix in col]
        # keep:
        # 1. original OHLCV columns
        # 2. newly made features with timeframe suffix {5m, 15m, 1h}
        cols_to_keep = used_cols + feature_cols
        
        if 'Timestamp' in df_feat.columns:
            cols_to_keep = ['Timestamp'] + cols_to_keep
        elif isinstance(df_feat.index, pd.DatetimeIndex) and df_feat.index.name == 'Timestamp':
            df_feat = df_feat.reset_index() 
            if 'Timestamp' not in df_feat.columns and 'index' in df_feat.columns: 
                 df_feat.rename(columns={'index':'Timestamp'}, inplace=True)
            cols_to_keep = ['Timestamp'] + cols_to_keep
        elif isinstance(df_feat.index, pd.DatetimeIndex): 
            df_feat = df_feat.reset_index()
            if 'Timestamp' not in df_feat.columns and 'index' in df_feat.columns:
                 df_feat.rename(columns={'index':'Timestamp'}, inplace=True)
            cols_to_keep = ['Timestamp'] + cols_to_keep
        
        # final df & col
        final_cols = [col for col in list(set(cols_to_keep)) if col in df_feat.columns]
        df_output = df_feat[final_cols].copy()
        return df_output

    def align_and_create_final_dataset(self, df_primary_featured: pd.DataFrame, 
                                       supporting_tf_dfs_featured: dict, 
                                       primary_tf_str: str) -> tuple[pd.DataFrame, list]:
        # align supporting timeframe {5m, 1h} to primary timeframe {15m} as one big df
        if df_primary_featured.empty:
            raise ValueError("Primary timeframe DataFrame (df_primary_featured) is empty for alignment.")
        df_final = df_primary_featured.copy()
        timestamp_col = 'Timestamp'
        if timestamp_col not in df_final.columns: 
            raise ValueError(f"Primary DF missing '{timestamp_col}' column for alignment.")
            
        df_final[timestamp_col] = pd.to_datetime(df_final[timestamp_col], errors='coerce', utc=True)
        df_final.dropna(subset=[timestamp_col], inplace=True)
        if df_final.empty:
            logging.error("Primary DataFrame became empty after Timestamp processing in align_and_create_final_dataset.")
            return pd.DataFrame(), []
        df_final = df_final.set_index(timestamp_col).sort_index()

        # define prefixes for basic OHLCV and candle calculations to exclude from supporting TFs
        ohlcv_candle_calc_prefixes_to_exclude = [
            'Open_', 'High_', 'Low_', 'Close_', 'Volume_', 
            'CandleRange_', 'BodySize_', 'ClosePosInRange_'
        ]

        for support_tf_str_key, df_support_orig in supporting_tf_dfs_featured.items():
            if df_support_orig.empty:
                logging.debug(f"Supporting featured DataFrame for {support_tf_str_key} is empty. Skipping merge.")
                continue
            df_support = df_support_orig.copy()
            if timestamp_col not in df_support.columns:
                logging.warning(f"Supporting DF {support_tf_str_key} missing '{timestamp_col}'. Skipping merge.")
                continue
            df_support[timestamp_col] = pd.to_datetime(df_support[timestamp_col], errors='coerce', utc=True)
            df_support.dropna(subset=[timestamp_col], inplace=True)
            if df_support.empty:
                logging.warning(f"Supporting DF {support_tf_str_key} became empty after Timestamp processing. Skipping merge.")
                continue
            df_support = df_support.set_index(timestamp_col).sort_index()

            # select only calculated indicators from support df
            feature_cols_from_support = []
            for col in df_support.columns:
                if f"_{support_tf_str_key}" in col: # MUST have the correct suffix
                    is_basic_ohlcv_derivative = False
                    for prefix in ohlcv_candle_calc_prefixes_to_exclude:
                        # check if the column name starts with one of these prefixes
                        if col.upper().startswith(prefix.upper()): 
                            is_basic_ohlcv_derivative = True
                            break
                    if not is_basic_ohlcv_derivative: # if not, we want'em
                        feature_cols_from_support.append(col)
            
            if not feature_cols_from_support:
                logging.debug(f"No distinct indicator features found in supporting DataFrame for {support_tf_str_key} to merge.")
                continue

            # safety check in case it became empty
            if df_final.empty: 
                logging.error("df_final became empty before merging with supporting timeframes. Aborting further merges.")
                break 
            
            # merge them
            try:
                df_final = pd.merge_asof(left=df_final, right=df_support[feature_cols_from_support], 
                                         left_index=True, right_index=True, direction='backward')
                logging.debug(f"aligned and merged {len(feature_cols_from_support)} indicator features from {support_tf_str_key}.")
            except Exception as e:
                logging.error(f"error during merge_asof for {support_tf_str_key}: {e}, skipping this timeframe.")
        
        if df_final.empty: 
            logging.error("df_final is empty after all merge operations.")
            return pd.DataFrame(), []
        df_final.reset_index(inplace=True)


        # ---------- define final features for RL training ----------
        # goal is to include only indicator features from ALL timeframes
        # exclude 'Timestamp', 'datetime(just incase)', and all suffixed OHLCV & basic candle calcs from primary df
        # environment will use specific price column like 'Close' or something from df_final
        cols_to_exclude_from_agent_features = [timestamp_col, 'datetime']

        # list of primary timeframe's OHLCV and basic candle calculations (suffixed ofc) to exclude
        primary_ohlcv_candle_calc_suffixed = [f'{prefix}{primary_tf_str}' for prefix in ohlcv_candle_calc_prefixes_to_exclude]
        cols_to_exclude_from_agent_features.extend(primary_ohlcv_candle_calc_suffixed)
        cols_to_exclude_from_agent_features = list(set(cols_to_exclude_from_agent_features)) # ensure it's unique
        
        all_tf_suffixes_in_data = [primary_tf_str] + list(supporting_tf_dfs_featured.keys())
        agent_features_list = []
        for col in df_final.columns:
            # skip
            if col in cols_to_exclude_from_agent_features: continue
            
            is_agent_feature = False
            for tf_sfx in all_tf_suffixes_in_data:
                if f"_{tf_sfx}" in col: # if suffix feature, prob the one we created
                    # if it's from the primary timeframe, it's already passed the primary OHLCV exclusion.
                    # if it's from a supporting timeframe, we need to ensure it's not an OHLCV-like derivative
                    # (this check is kinda redundant if feature_cols_from_support was perfect, but just in case)
                    is_support_ohlcv_like = False
                    if tf_sfx != primary_tf_str: # it's a suffixed column from a supporting timeframe
                        for prefix in ohlcv_candle_calc_prefixes_to_exclude: # use the same exclusion list
                            if col.upper().startswith(prefix.upper()):
                                is_support_ohlcv_like = True; break
                    
                    if not is_support_ohlcv_like: 
                        is_agent_feature = True; break 
            
            # the selected ones
            if is_agent_feature:
                agent_features_list.append(col)

        final_agent_features_list = sorted(list(set(agent_features_list)))
        
        essential_primary_cols_for_env = ['Open', 'High', 'Low', 'Close', 'Volume'] + \
                                         [f'{col}_{primary_tf_str}' for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
        nan_check_cols = final_agent_features_list + [col for col in essential_primary_cols_for_env if col in df_final.columns]
        actual_nan_check_cols = [col for col in list(set(nan_check_cols)) if col in df_final.columns]

        original_len = len(df_final)
        if actual_nan_check_cols: 
            df_final.dropna(subset=actual_nan_check_cols, inplace=True)
            if original_len > 0 and len(df_final) < original_len:
                 logging.info(f"align_and_create_final_dataset: Dropped {original_len - len(df_final)} rows due to NaNs in critical columns.")
        
        df_final.reset_index(drop=True, inplace=True)
        if df_final.empty and original_len > 0: 
            logging.error("Final processed DataFrame is empty after final NaN drop.")
        logging.info(f"Final aligned dataset created. Shape: {df_final.shape}. Agent features: {len(final_agent_features_list)}")
        return df_final, final_agent_features_list
    
    def fit_transform(self, data_array: np.ndarray, col_names_for_scaling: list = None) -> np.ndarray:
        """
        functions:
        -> fits the scaler to the data_array and transforms it.
        -> store the mean and std for each column. 
        -> handles inf and NaN values.
        """
        if not isinstance(data_array, np.ndarray):
            raise ValueError("input data_array must be np array.")
        if data_array.ndim != 2:
            raise ValueError("input data_array must be 2D.")
        if data_array.shape[0] == 0:
            logging.warning("fit_transform called with empty data_array (0 rows), return as is.")
            return data_array.copy()

        self.col_names_for_scaler = col_names_for_scaling
        scaled_data = np.zeros_like(data_array, dtype=np.float64)
        self.scaler_params = {} # reset scaler_params for this fit

        data_array_cleaned = data_array.copy().astype(np.float64)

        # 1. check for any inf value and replace with NaN
        inf_mask = np.isinf(data_array_cleaned)
        if np.any(inf_mask):
            data_array_cleaned[inf_mask] = np.nan
            _, inf_col_indices = np.where(inf_mask)
            unique_inf_col_indices = sorted(list(set(inf_col_indices)))
            # code below only to check where inf was located, feel free to uncomment
            # msg_cols = f"(Indices: {unique_inf_col_indices})"
            if self.col_names_for_scaler:
                inf_col_names = [self.col_names_for_scaler[i] for i in unique_inf_col_indices if i < len(self.col_names_for_scaler)]
                if inf_col_names: msg_cols = f"(Names: {inf_col_names})"
            logging.warning(f"fit_transform: 'inf' values found and replaced with NaN in input columns {msg_cols}")
            
        # 2. calculate params (mean, std) and scale column by column
        for i in range(data_array_cleaned.shape[1]):
            col_data = data_array_cleaned[:, i] # might contain NaN
            current_col_name = (self.col_names_for_scaler[i] if self.col_names_for_scaler and i < len(self.col_names_for_scaler) else f"index {i}")

            # we ball
            mean = np.nanmean(col_data)
            std = np.nanstd(col_data)

            # handle case where mean and std might be NaN or std too small
            if np.isnan(mean):
                logging.debug(f"mean for '{current_col_name}' is NaN (all NaNs in column), defaulting mean to 0.")
            if np.isnan(std) or std < 1e-8: # Std is NaN (all NaNs) or zero/very small
                std = 1.0 # default std to 1 to avoid division by zero/small num & produce 0s if mean is also 0
                logging.debug(f"std for '{current_col_name}' is NaN or near zero, default std to 1.")
            
            self.scaler_params[i] = {'mean': mean, 'std': std} # store everything
            scaled_values = (col_data - mean) / std

            # ensure no NaN
            scaled_values[np.isnan(scaled_values)] = 0.
            scaled_data[:, i] = scaled_values
            
        return scaled_data

    def transform(self, data_array: np.ndarray) -> np.ndarray:
        """
        function:
        -> transforms the data_array using pre-computed scaler_params.
        -> handles inf and NaN values.
        """
        if not self.scaler_params:
            raise ValueError("scaler has not been fitted yet, pls call fit_transform first.")
        if not isinstance(data_array, np.ndarray):
            raise ValueError("input data_array must be np array.")
        if data_array.ndim != 2:
            raise ValueError("input data_array must be 2D.")
        if data_array.shape[0] == 0: return data_array.copy()
        if data_array.shape[1] != len(self.scaler_params):
            raise ValueError(f"input data_array has {data_array.shape[1]} columns, but scaler was fitted for {len(self.scaler_params)} columns.")

        scaled_data = np.zeros_like(data_array, dtype=np.float64)
        data_array_cleaned = data_array.copy().astype(np.float64)

        inf_mask = np.isinf(data_array_cleaned)
        if np.any(inf_mask):
            data_array_cleaned[inf_mask] = np.nan
            # Log details similar to fit_transform if desired for verbosity
            logging.warning("transform: 'inf' values found and replaced with NaN in input array.")

        for i in range(data_array_cleaned.shape[1]):
            col_data = data_array_cleaned[:, i]
            params = self.scaler_params.get(i)
            if params is None: # should be caught by shape check above, but just in case :D
                 logging.error(f"transform: no scaler parameters for column index {i}, use default 0/1.")
                 mean, std = 0., 1.
            else:
                mean, std = params['mean'], params['std']
            
            scaled_values = (col_data - mean) / std
            scaled_values[np.isnan(scaled_values)] = 0.
            scaled_data[:, i] = scaled_values
        return scaled_data