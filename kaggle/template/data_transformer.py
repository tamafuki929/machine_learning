from sklearn.impute import MissingIndicator
from sklearn.preprocessing import OrdinalEncoder

class DataTransformer:
    """
    __XXX_encoding(): General processing functions called from make_XXX
    __select_XXX(): Function to execute feature selection
    make_XXX(): Function to generate new column XXX
    trans_XXX(): Function to process existing column XXX
    """
    
    def __init__(self, scaler_x = None, scaler_y = None, not_use_cols = None):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y # in classifier, scaler -> encoder
        self.not_use_cols = not_use_cols
    
    def __set_info(self, X, y, train_id):
        self.X = X
        self.y = y
        self.train_id = train_id
    
    def __missing_indicate_encoding(self, data):
        null_cols = data.loc[:, data.isna().any()].columns.to_list()
        new_cols  = [col + "_nan" for col in null_cols]
        data[new_cols] = MissingIndicator().fit_transform(data[null_cols])
        return data
    
    def __frequency_encoding(self, data, columns, only_train = True):
        new_cols = [col + "_freq" for col in columns]
        for i in range(len(columns)):
            if only_train:
                freq = data.loc[self.train_id, columns[i]].value_counts().to_frame(new_cols[i])
            else:
                freq = data[columns[i]].value_counts().to_frame(new_cols[i])
            data = pd.merge(data, freq, left_on = columns[i], how = "left", right_index = True)
        return data
    
    def __label_encoding(self, data, columns):
        enc = OrdinalEncoder()
        data[columns] = enc.fit_transform(data[columns])
        return data
    
    def __bool_to_int_encoding(self, data):
        cols = data.select_dtypes(include = bool).columns.to_list()
        data[cols] = data.loc[:, cols].astype(int)
        return data
    
    def __unordered_category_encoding(self, data, columns):
        data[columns] = data[columns].astype("category")
        return data
    
    def __select_columns(self):
        self.use_cols = self.X.columns
        if self.not_use_cols: 
            self.use_cols = self.use_cols.drop(self.not_use_cols)
        self.X = self.X[self.use_cols]
        
    def make_missing_indicators(self):
        self.X = self.__missing_indicate_encoding(self.X)
    
    def make_lname(self):
        new_col = "lname"
        def extract_lname(data):
            name_df = data["name"].str.split("(", expand = True)[0]
            name_df = name_df.str.split(".", expand = True)[1]
            name_df = name_df.str.strip()
            return name_df.map(lambda x: x.split(" ")[-1])
        
        self.X[new_col] = extract_lname(self.X)
        self.X = self.__frequency_encoding(self.X, [new_col], only_train = False)
        self.X.loc[self.X.lname_freq == 1, new_col] = "alone"
        self.X.loc[self.X.lname_freq.isnull(), new_col] = "alone"
    
    def make_alias(self):
        new_col = "alias"
        name_df = self.X["name"].str.split("(", expand = True)
        self.X[new_col] = name_df[1].notna()
    
    def make_ticket_freq(self):
        self.X = self.__frequency_encoding(self.X, ["ticket"])
    
    def make_cabin_floor(self):
        self.X["cabin_floor"] = self.X.cabin.str[0]
    
    def make_cabin_freq(self):
        self.X = self.__frequency_encoding(self.X, ["cabin"])
    
    def make_cabin_floor_freq(self):
        self.X = self.__frequency_encoding(self.X, ["cabin_floor"])
    
    def trans_bool_to_int(self):
        self.X = self.__bool_to_int_encoding(self.X)
    
    def trans_category_to_int(self):
        cat_cols = self.X.select_dtypes(include = "object").columns.to_list()
        self.X = self.__label_encoding(self.X, cat_cols)
    
    def fit_transform(self, X, y, train_id):
        self.__set_info(X, y, train_id)
        transformers = [self.make_missing_indicators, self.make_lname, 
                        self.make_alias, self.make_ticket_freq, 
                        self.make_cabin_floor, self.make_cabin_freq, 
                        self.make_cabin_floor_freq, self.trans_bool_to_int, 
                        self.__select_columns, self.trans_category_to_int
                       ]
        for transformer in transformers: 
            transformer()
        return self.X, self.y