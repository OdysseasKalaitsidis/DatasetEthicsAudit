from typing import Tuple, List, Optional
import pandas as pd
import streamlit as st
import io

class DataManager:
    """
    The Gatekeeper: Handles data loading, caching, and type inference.
    """

    @staticmethod
    @st.cache_data
    def load_data(file, file_name: str) -> Optional[pd.DataFrame]:
        """
        Loads data from CSV or Excel files.
        
        Args:
            file: The uploaded file object.
            file_name: Name of the file to determine extension.
            
        Returns:
            pd.DataFrame or None if loading fails.
        """
        df = None
        try:
            if file_name.endswith('.csv'):
                # Robust Sniffing: Read first line to check for semicolons
                try:
                    string_data = file.getvalue().decode("utf-8")
                except UnicodeDecodeError:
                    string_data = file.getvalue().decode("latin1")
                
                first_line = string_data.split('\n')[0]
                if first_line.count(';') > first_line.count(','):
                    sep = ';'
                else:
                    sep = ','
                
                file.seek(0)
                df = pd.read_csv(file, sep=sep)

            elif file_name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                st.error("Unsupported file format. Please upload CSV or Excel.")
                return None

            if df is not None:
                # Drop completely empty rows
                df.dropna(how='all', inplace=True)
                return df
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
        
        return df

    @staticmethod
    def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Classifies columns into numerical and categorical.
        
        Args:
            df: The dataframe to analyze.
            
        Returns:
            Tuple containing (numerical_columns, categorical_columns).
        """
        # Select numerical columns (float, int)
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Select categorical columns (object, category, bool)
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        return num_cols, cat_cols