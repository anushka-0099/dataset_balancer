import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

class DatasetBalancer:
    def __init__(self, target_column='target', test_size=0.2, random_state=42):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.smote = SMOTE(random_state=self.random_state)
        self.encoder = LabelEncoder()
        self.feature_encoders = {}
        self.df = None
        self.X = None
        self.y = None
        
    def load_data(self, file_path):
        """Load CSV or Excel and separate features and target"""
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format.")
            
            if self.target_column not in self.df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found.")
            
            self.X = self.df.drop(columns=[self.target_column])
            self.y = self.df[self.target_column]
            
            return self.df.shape, Counter(self.y)
            
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def preprocess(self):
        """Handle missing values and encode categorical data"""
        try:
            # Fill numeric missing values with mean
            numeric_cols = self.X.select_dtypes(include=['number']).columns
            self.X[numeric_cols] = self.X[numeric_cols].fillna(self.X[numeric_cols].mean())
            
            # Fill non-numeric missing values with mode
            non_numeric_cols = self.X.select_dtypes(exclude=['number']).columns
            for col in non_numeric_cols:
                self.X[col] = self.X[col].fillna(self.X[col].mode()[0])
            
            # Encode categorical features
            categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns
            self.feature_encoders = {}
            for col in categorical_cols:
                self.feature_encoders[col] = LabelEncoder()
                self.X[col] = self.feature_encoders[col].fit_transform(self.X[col].astype(str))
            
            # Encode target if it's not numeric
            if not pd.api.types.is_numeric_dtype(self.y):
                self.y = self.encoder.fit_transform(self.y.astype(str))
                
        except Exception as e:
            raise ValueError(f"Error during preprocessing: {str(e)}")

    def balance(self):
        """Split and apply SMOTE to the training data"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state
            )
            
            class_counts = Counter(y_train)
            if len(class_counts) < 2:
                return pd.concat([
                    pd.DataFrame(X_train).assign(**{self.target_column: y_train}),
                    pd.DataFrame(X_test).assign(**{self.target_column: y_test})
                ])
                
            X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
            
            # Combine resampled data with test data for download
            balanced_df = pd.concat([
                pd.DataFrame(X_resampled).assign(**{self.target_column: y_resampled}),
                pd.DataFrame(X_test).assign(**{self.target_column: y_test})
            ])
            
            return balanced_df
            
        except Exception as e:
            raise ValueError(f"Error during balancing: {str(e)}")