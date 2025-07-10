import pandas as pd
import numpy as np
import os
import logging
import yaml
import json
from .feature_engineering import FeatureEngineer
import joblib

class DataProcessor:
    def __init__(self, config_path):
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            if self.config is None:
                logging.warning(f"Configuration file {config_path} is empty. Using default paths.")
                self.config = {}
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {config_path}. Using default paths.")
            self.config = {}
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML configuration file {config_path}: {e}. Using default paths.")
            self.config = {}

        data_paths_config = self.config.get('data_paths', {})
        dp_config = self.config.get('data_processing', {})
        self.indicators_config = self.config['feature_engineering']['indicators']

        self.raw_dir = data_paths_config.get("raw_data_directory", "data/historical_raw")
        self.processed_dir = data_paths_config.get("processed_data_directory", "data/historical_processed")
        
        self.features_path_config = data_paths_config.get("features_list_path", 
                                        os.path.join(self.processed_dir, "final_agent_features.json"))
        self.scaler_params_path_config = data_paths_config.get("scaler_params_path", 
                                        os.path.join(self.processed_dir, "custom_fe_scaler_params.joblib"))
        self.scaler_colnames_path_config = data_paths_config.get("scaler_colnames_path",
                                        os.path.join(self.processed_dir, "custom_fe_scaler_colnames.joblib"))
        
        self.primary_timeframe_config = dp_config.get("primary_timeframe", "15m")
        self.feature_engineer_timeframes = dp_config.get("feature_engineer_timeframes", ["1h", "15m", "5m"])

        self.scaler_params = joblib.load(self.scaler_params_path_config)
        logging.info(f"Loaded pre-trained scaler parameters from {self.scaler_params_path_config}")

        self.scaler_colnames = joblib.load(self.scaler_colnames_path_config)
        logging.info(f"Loaded pre-trained scaler column names from {self.scaler_colnames_path_config}")

        for pth_val in [self.raw_dir, self.processed_dir, 
                        os.path.dirname(self.features_path_config), 
                        os.path.dirname(self.scaler_params_path_config), 
                        os.path.dirname(self.scaler_colnames_path_config)]:
            if pth_val: os.makedirs(pth_val, exist_ok=True)
        
        self.feature_engineer = FeatureEngineer(config_path)
        logging.info("DataProcessor initialized.")

    def _apply_indicators(self, df: pd.DataFrame, timeframe: str):
        df_feat = df.copy()
        if 'Timestamp' in df_feat.columns:
            # This warning you see is also a clue that the initial string->datetime parsing is ambiguous.
            # Specifying the format can make it more robust.
            df_feat['Timestamp'] = pd.to_datetime(df_feat['Timestamp'], utc=True, errors='coerce')
        
            df_feat.dropna(subset=['Timestamp'], inplace=True)
                
            df_feat = df_feat.set_index('Timestamp').sort_index()

        for indicator_cfg in self.indicators_config:
            try:
                name, params, on_col = indicator_cfg['name'].upper(), indicator_cfg.get('params', {}), indicator_cfg.get('on_column', 'Close')
                if on_col not in df_feat.columns: continue

                if hasattr(df_feat.ta, name.lower()):
                    indicator_func = getattr(df_feat.ta, name.lower())
                    result = indicator_func(close=df_feat[on_col], high=df_feat.get('High'), low=df_feat.get('Low'), volume=df_feat.get('Volume'), **params, append=False)
                    if isinstance(result, pd.DataFrame):
                        for col in result.columns:
                            clean_col = col.replace(f'_{params.get("length", "")}_{params.get("std", "")}.0', '').replace('BBM','BB_middle').replace('BBU','BB_upper').replace('BBL','BB_lower').replace('BBB','BBW').replace('BBP','BBP')
                            df_feat[f"{clean_col}_{timeframe}"] = result[col]
                    else:
                        col_name = indicator_cfg.get('output_name_override') or f"{name}_{params.get('length', '')}"
                        df_feat[f"{col_name}_{timeframe}"] = result
                elif name == "PCT_CHANGE":
                    col_name = indicator_cfg.get('output_name_override') or f"PCT_CHANGE_{params.get('periods', 1)}"
                    df_feat[f"{col_name}_{timeframe}"] = df_feat[on_col].pct_change(**params).replace([np.inf, -np.inf], np.nan)
            except Exception as e:
                # Corrected logging call
                logging.error(f"Error calculating {indicator_cfg.get('name')} for {timeframe}: {e}")
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df_feat.columns: df_feat.rename(columns={col: f"{col}_{timeframe}"}, inplace=True)

        df_feat.reset_index(inplace=True)
        return df_feat

    def process_live_data(self, raw_data_dict: dict):
        featured_dfs = {tf: self._apply_indicators(df_raw, tf) for tf, df_raw in raw_data_dict.items() if not df_raw.empty}
        if self.primary_timeframe_config not in featured_dfs or featured_dfs[self.primary_timeframe_config].empty:
            raise ValueError("Primary timeframe data could not be processed for live inference.")

        df_primary = featured_dfs[self.primary_timeframe_config].set_index("Timestamp").sort_index()
        df_aligned = df_primary.copy()

        for tf, df_support in featured_dfs.items():
            if tf == self.primary_timeframe_config: continue
            support_features = [c for c in df_support.columns if f"_{tf}" in c and not any(o in c for o in ['Open_', 'High_', 'Low_', 'Close_', 'Volume_'])]
            if not support_features: continue
            df_aligned = pd.merge_asof(df_aligned, df_support[['Timestamp'] + support_features].set_index('Timestamp').sort_index(), left_index=True, right_index=True, direction='backward')

        unscaled_df = df_aligned.copy().reset_index()
        df_aligned.bfill(inplace=True)
        df_aligned.ffill(inplace=True)
        df_aligned.dropna(inplace=True)

        if df_aligned.empty:
            return None, None # Return None to prevent the ambiguous ValueError

        # Ensure all required columns for scaling are present
        for col in self.scaler_colnames:
            if col not in df_aligned.columns:
                df_aligned[col] = 0

        # --- FIX STARTS HERE: Corrected Scaling Logic ---
        cols_to_transform = [col for col in self.scaler_colnames if col in df_aligned.columns]
        if cols_to_transform:
            # Loop through each column that needs to be scaled
            for col in cols_to_transform:
                try:
                    # 1. Find the numerical index (position) of the column
                    idx = self.scaler_colnames.index(col)

                    # 2. Use that index to get the specific parameter dict for that column
                    params_for_col = self.scaler_params.get(idx)

                    if params_for_col:
                        mean = params_for_col.get('mean')
                        std = params_for_col.get('std')

                        # 3. Apply the scaling formula
                        if mean is not None and std is not None and std != 0:
                            df_aligned[col] = (df_aligned[col] - mean) / std
                        else:
                            logging.warning(f"Invalid parameters for column '{col}'. Scaling skipped.")
                    else:
                        logging.warning(f"No parameters found for column '{col}' at index {idx}. Scaling skipped.")

                except (ValueError, IndexError):
                    logging.warning(f"Column '{col}' not in scaler's column list. Scaling skipped.")
        # --- FIX ENDS HERE ---

        df_aligned_out = df_aligned.reset_index()

        time_format = '%Y-%m-%d %H:%M:%S'
        df_aligned_out['Timestamp'] = df_aligned_out['Timestamp'].dt.strftime(time_format)
        unscaled_df['Timestamp'] = unscaled_df['Timestamp'].dt.strftime(time_format)

        return df_aligned_out, unscaled_df

    def load_raw_data(self, filename):
        filepath = os.path.join(self.raw_dir, filename)
        try:
            df = pd.read_csv(filepath, parse_dates=["Timestamp"])
            logging.info(f"loaded raw data from {filepath} (Shape: {df.shape})")
            return df
        except FileNotFoundError:
            logging.error(f"raw data file not found: {filepath}. Returning empty DataFrame.")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"error loading raw data from {filepath}: {e}")
            return pd.DataFrame()

    def process_all_timeframes(self, config_path: str):
        dp_config = self.config.get('data_processing', {})
        
        train_split_ratio = float(dp_config.get('train_split_ratio', 0.7))
        validation_split_ratio = float(dp_config.get('validation_split_ratio', 0.15))
        
        tf_data = {}
        raw_files_list = [f"klines_{tf}.csv" for tf in self.feature_engineer_timeframes]
        for filename in raw_files_list:
            timeframe_str = filename.split("_")[-1].replace(".csv", "")
            # Ensure this timeframe is expected, otherwise FE might not be configured for it
            if timeframe_str not in self.feature_engineer_timeframes:
                logging.warning(f"File for timeframe '{timeframe_str}' found but not in configured 'feature_engineer_timeframes'. Skipping.")
                continue
            raw_df = self.load_raw_data(filename)
            tf_data[timeframe_str] = self.feature_engineer.calculate_native_features(raw_df, timeframe_str)

        primary_tf_str = self.primary_timeframe_config
        if primary_tf_str not in tf_data or tf_data[primary_tf_str].empty:
            available_tfs = [tf for tf, df in tf_data.items() if not df.empty];
            if not available_tfs: raise ValueError("All TFs are empty after feature calculation.")
            new_primary_tf_str = available_tfs[0]
            logging.warning(f"Primary TF '{primary_tf_str}' data missing. Using '{new_primary_tf_str}' as primary.")
            primary_tf_str = new_primary_tf_str
        
        supporting_tf_dfs = {tf: df for tf, df in tf_data.items() if tf != primary_tf_str and not df.empty}
        
        df_final, final_agent_features_list = self.feature_engineer.align_and_create_final_dataset(
            tf_data[primary_tf_str], supporting_tf_dfs, primary_tf_str
        )

        if df_final.empty:
            logging.error("Final DataFrame is empty after alignment. Cannot proceed.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

        df_final = df_final.sort_values("Timestamp").reset_index(drop=True)

        price_col_for_env = f"Close_{primary_tf_str}"
        if price_col_for_env not in df_final.columns:
            if "Close" in df_final.columns:
                price_col_for_env = "Close" 
            else:
                logging.error(f"Critical: Price column '{price_col_for_env}' (and fallback 'Close') not found in df_final after feature engineering.")
                potential_price_cols = [col for col in df_final.columns if col.startswith("Close_")]
                if potential_price_cols:
                    price_col_for_env = potential_price_cols[0]
                    logging.warning(f"Using '{price_col_for_env}' as price column as primary was not found.")
                else:
                    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), final_agent_features_list


        if "Timestamp" not in df_final.columns:
             logging.error("'Timestamp' column is missing from df_final before splitting. Critical error.")
             return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), final_agent_features_list

        if final_agent_features_list:
            try:
                with open(self.features_path_config, 'w') as f:
                    json.dump(final_agent_features_list, f, indent=4)
                logging.info(f"Agent features list saved to {self.features_path_config} ({len(final_agent_features_list)} features).")
            except Exception as e:
                logging.error(f"Could not save agent features list to {self.features_path_config}: {e}")
        else:
            logging.warning("final_agent_features_list is empty. Not saving features list.")

        n = len(df_final)
        if n == 0:
            logging.error("df_final has zero length before splitting.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), final_agent_features_list

        train_end_idx = int(n * train_split_ratio)
        val_end_idx = train_end_idx + int(n * validation_split_ratio)

        df_train_orig = df_final.iloc[:train_end_idx].copy()
        df_val_orig = df_final.iloc[train_end_idx:val_end_idx].copy()
        df_test_orig = df_final.iloc[val_end_idx:].copy()
        logging.info(f"Data split: Train {len(df_train_orig)}, Validation {len(df_val_orig)}, Test {len(df_test_orig)} rows.")

        if df_train_orig.empty:
            logging.error("Training DataFrame is empty after split. Cannot fit scaler.")
            return df_train_orig, df_val_orig, df_test_orig, final_agent_features_list

        # feature_cols_to_scale should ONLY be from final_agent_features_list
        # These are the features the agent will actually see as input (after observation processing)
        # The raw price column (price_col_for_env) should NOT be in feature_cols_to_scale
        # if it's directly used by the environment for PnL.
        # However, features DERIVED from price (like returns, normalized price for obs) ARE in final_agent_features_list.
        feature_cols_to_scale = [col for col in final_agent_features_list if col in df_train_orig.columns and col != "Timestamp"]

        if not feature_cols_to_scale:
            logging.warning("No agent feature columns found in training data for scaling. Proceeding without scaling these specific features.")
            df_train_scaled = df_train_orig.copy()
            df_val_scaled = df_val_orig.copy()
            df_test_scaled = df_test_orig.copy()
        else:
            logging.info(f"Fitting custom scaler on training data columns: {feature_cols_to_scale}")
            train_scaled_values = self.feature_engineer.fit_transform(
                df_train_orig[feature_cols_to_scale].values, col_names_for_scaling=feature_cols_to_scale
            )
            
            # amke new DataFrames for scaled data to avoid modifying originals directly yet
            df_train_scaled = pd.DataFrame(train_scaled_values, columns=feature_cols_to_scale, index=df_train_orig.index)
            
            # add back non-scaled columns that are essential (Timestamp and the original price column for env)
            # and any other columns from df_final that were not part of final_agent_features_list but might be needed.
            # for now, specifically add Timestamp and price_col_for_env
            df_train_scaled["Timestamp"] = df_train_orig["Timestamp"].values
            if price_col_for_env not in df_train_scaled.columns and price_col_for_env in df_train_orig.columns:
                 df_train_scaled[price_col_for_env] = df_train_orig[price_col_for_env].values


        if hasattr(self.feature_engineer, 'scaler_params') and self.feature_engineer.scaler_params:
            try:
                joblib.dump(self.feature_engineer.scaler_params, self.scaler_params_path_config)
                if hasattr(self.feature_engineer, 'col_names_for_scaler') and self.feature_engineer.col_names_for_scaler is not None:
                    joblib.dump(self.feature_engineer.col_names_for_scaler, self.scaler_colnames_path_config)
                logging.info(f"Scaler params and colnames saved to {self.scaler_params_path_config} and {self.scaler_colnames_path_config}")
            except Exception as e:
                logging.error(f"Error saving scaler params: {e}")
        elif feature_cols_to_scale: # Only warn if we actually tried to scale
            logging.warning("No scaler parameters found in FeatureEngineer to save after fit_transform, though scaling was attempted.")
        
        fitted_scaler_columns = self.feature_engineer.col_names_for_scaler or feature_cols_to_scale

        def scale_and_reconstruct_dataframe(df_orig, scaler_instance, cols_to_scale_list, fitted_cols_order, env_price_col):
            if df_orig.empty:
                return df_orig.copy()

            df_scaled_features_part = df_orig.copy() # Start with a copy to preserve all original columns

            if not cols_to_scale_list: # No features to scale for the agent
                # Ensure Timestamp and price column are present
                if "Timestamp" not in df_scaled_features_part.columns and "Timestamp" in df_final.columns: # Check df_final as ultimate source
                    logging.warning("Timestamp column missing in scale_and_reconstruct_dataframe input df_orig.")
                if env_price_col not in df_scaled_features_part.columns and env_price_col in df_final.columns:
                    logging.warning(f"{env_price_col} column missing in scale_and_reconstruct_dataframe input df_orig.")
                return df_scaled_features_part


            if hasattr(scaler_instance, 'scaler_params') and scaler_instance.scaler_params:
                cols_present_for_scaling = [col for col in fitted_cols_order if col in df_orig.columns]

                if set(cols_present_for_scaling) != set(fitted_cols_order):
                    logging.warning(f"Column mismatch for scaling in scale_and_reconstruct_dataframe. Expected {fitted_cols_order}, found {cols_present_for_scaling}. Skipping scaling for this df.")
                elif cols_present_for_scaling:
                    try:
                        data_to_transform = df_orig[fitted_cols_order].values
                        scaled_values = scaler_instance.transform(data_to_transform)
                        
                        temp_scaled_df = pd.DataFrame(scaled_values, columns=fitted_cols_order, index=df_orig.index)
                        for col in fitted_cols_order:
                            df_scaled_features_part[col] = temp_scaled_df[col] # Update only scaled columns
                    except Exception as e:
                        logging.error(f"Error transforming dataframe in scale_and_reconstruct_dataframe: {e}. Features remain unscaled.")
            
            # Ensure Timestamp and the original price column are present in the final scaled DataFrame
            if "Timestamp" not in df_scaled_features_part.columns and "Timestamp" in df_orig:
                df_scaled_features_part["Timestamp"] = df_orig["Timestamp"]
            if env_price_col not in df_scaled_features_part.columns and env_price_col in df_orig:
                 df_scaled_features_part[env_price_col] = df_orig[env_price_col]
            
            return df_scaled_features_part

        if not feature_cols_to_scale: # If no scaling happened for train, it won't for val/test
            df_val_scaled = df_val_orig.copy()
            df_test_scaled = df_test_orig.copy()
        else:
            df_val_scaled = scale_and_reconstruct_dataframe(df_val_orig, self.feature_engineer, feature_cols_to_scale, fitted_scaler_columns, price_col_for_env)
            df_test_scaled = scale_and_reconstruct_dataframe(df_test_orig, self.feature_engineer, feature_cols_to_scale, fitted_scaler_columns, price_col_for_env)


        # reorder columns: Timestamp first, then agent features, then the specific price column for env.
        # ensures the DataFrames passed to TradingEnv have the raw price column.
        def reorder_final_cols(df, agent_feature_list, env_price_col_name):
            if df.empty:
                return df
            
            ordered_cols = []
            # 1. Timestamp  
            if "Timestamp" in df.columns:
                ordered_cols.append("Timestamp")
            
            # 2. agent features
            # ensure agent_feature_list only contains columns actually in the DataFrame
            actual_agent_features = [col for col in agent_feature_list if col in df.columns and col != "Timestamp"]
            ordered_cols.extend(actual_agent_features)
            
            # 3. the specific price column for the environment
            if env_price_col_name in df.columns and env_price_col_name not in ordered_cols:
                ordered_cols.append(env_price_col_name)
            

            final_ordered_cols = []
            if "Timestamp" in df.columns: final_ordered_cols.append("Timestamp")
            
            # add all agent features
            for col in agent_feature_list:
                if col in df.columns and col not in final_ordered_cols:
                    final_ordered_cols.append(col)
            
            # make sure that price_col_for_env is present
            if env_price_col_name in df.columns and env_price_col_name not in final_ordered_cols:
                final_ordered_cols.append(env_price_col_name)
            
            # add any remaining columns from the original df_final that were not agent features or price col
            # this is to keep all information if needed later, though TradingEnv might not use them all directly.
            for col in df.columns: # df here is df_train_scaled, df_val_scaled, or df_test_scaled
                if col not in final_ordered_cols:
                    final_ordered_cols.append(col)
            
            return df[final_ordered_cols]

        # `final_agent_features_list` are the features the agent model expects.
        # `price_col_for_env` is the specific column name the TradingEnv needs for current price.
        df_train_scaled = reorder_final_cols(df_train_scaled, final_agent_features_list, price_col_for_env)
        df_val_scaled = reorder_final_cols(df_val_scaled, final_agent_features_list, price_col_for_env)
        df_test_scaled = reorder_final_cols(df_test_scaled, final_agent_features_list, price_col_for_env)

        # final check for the price column in the scaled dataframes
        for df_name, df_check in [("train", df_train_scaled), ("validation", df_val_scaled), ("test", df_test_scaled)]:
            if not df_check.empty and price_col_for_env not in df_check.columns:
                logging.error(f"FATAL: Price column '{price_col_for_env}' is MISSING from df_{df_name}_scaled AFTER all processing.")
                logging.error(f"df_{df_name}_scaled columns: {df_check.columns.tolist()}")
            elif not df_check.empty:
                 logging.info(f"Price column '{price_col_for_env}' is PRESENT in df_{df_name}_scaled.")


        return df_train_scaled, df_val_scaled, df_test_scaled, final_agent_features_list

    def save_processed_data(self, df, filename, data_type_info):
        filepath = os.path.join(self.processed_dir, filename)
        try:
            if df.empty:
                logging.warning(f"Attempted to save an empty DataFrame ({data_type_info}) to {filepath}. Skipping.")
                return
            num_data_cols = df.shape[1] - (1 if "Timestamp" in df.columns else 0)
            logging.info(f'saving data to {filepath} ({data_type_info}). \nShape:\n============\n{df.shape}\n============\nData Columns: {num_data_cols}. Timesteps: {df.shape[0]:,}\n')
            df.to_csv(filepath, index=False)
            logging.info(f"({data_type_info}) saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving processed data ({data_type_info}) to {filepath}: {e}")

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     config_path = "config/config.yaml"
#     processor = DataProcessor(config_path = config_path)
    
#     # list raws
#     raw_data_files = ["klines_1h.csv", "klines_15m.csv", "klines_5m.csv"]
    
#     # process data.
#     df_train, df_test = processor.process_all_timeframes(raw_data_files, split_ratio = .8)

#     processor.save_processed_data(df_train, "train_processed_15m.csv", 'train')
#     processor.save_processed_data(df_test, "test_processed_15m.csv", 'test')
