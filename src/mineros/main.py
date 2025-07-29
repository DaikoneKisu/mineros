import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mineros import load_mineros_db

def eda(mineros_db: dict[str, pd.DataFrame]) -> None:
    for table_name, df in mineros_db.items():
        print(f'\n===== Table: {table_name} =====')
        print(df.info())
        print(df.describe(include='all'))

        # Duplicates analysis
        num_duplicates = df.duplicated().sum()
        print(f'Duplicated rows: {num_duplicates}')

        # Outlier analysis (numeric columns)
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            _fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(8, 4*len(numeric_cols))) # pyright: ignore[reportUnknownMemberType]
            if len(numeric_cols) == 1:
                axes = [axes]
            for i, col in enumerate(numeric_cols):
                sns.boxplot(x=df[col], ax=axes[i])
                axes[i].set_title(f'Outliers in {col}')
            plt.tight_layout()
            plt.show() # pyright: ignore[reportUnknownMemberType]

        # Distribution analysis (numeric and categorical)
        for col in df.columns:
            plt.figure(figsize=(6, 4)) # pyright: ignore[reportUnknownMemberType]
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # Encode categorical for distribution
                value_counts = df[col].value_counts()
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.title(f'Distribution of {col} (categorical)') # pyright: ignore[reportUnknownMemberType]
                plt.xticks(rotation=45) # pyright: ignore[reportUnknownMemberType]
            else:
                sns.histplot(data=df, x=col, kde=True)
                plt.title(f'Distribution of {col} (numeric)') # pyright: ignore[reportUnknownMemberType]
            plt.tight_layout()
            plt.show() # pyright: ignore[reportUnknownMemberType]

        # Correlation analysis (numeric)
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            plt.figure(figsize=(8, 6)) # pyright: ignore[reportUnknownMemberType]
            sns.heatmap(corr, annot=True, cmap='coolwarm') # pyright: ignore[reportUnknownMemberType]
            plt.title('Correlation Matrix') # pyright: ignore[reportUnknownMemberType] # pyright: ignore[reportUnknownMemberType]
            plt.show() # pyright: ignore[reportUnknownMemberType]

def main():
    mineros_db = load_mineros_db()

    eda(mineros_db)