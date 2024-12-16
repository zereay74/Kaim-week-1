import pandas as pd

class DataLoader:
    """
    A class to handle loading of financial data from CSV files.
    """

    def __init__(self, file_path):
        """
        Initializes the DataLoader with the file path to the CSV file.

        :param file_path: str, path to the CSV file
        """
        self.file_path = file_path

    def load_data(self):
        """
        Loads the CSV file into a pandas DataFrame.

        :return: pd.DataFrame containing the data from the CSV file
        """
        try:
            data = pd.read_csv(self.file_path)
            print(f"Data successfully loaded from {self.file_path}")
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: No data in file at {self.file_path}")
            return None
        except Exception as e:
            print(f"An error occurred while loading the file: {e}")
            return None

# Example usage:
# loader = DataLoader("path_to_your_file.csv")
# df = loader.load_data()


def load_csv_data(file_path):

    return pd.read_csv(file_path)



