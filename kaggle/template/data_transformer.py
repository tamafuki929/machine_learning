from sklearn.impute import MissingIndicator
from sklearn.preprocessing import OrdinalEncoder

class DataTransformer:
    """
    __XXX_encoding(): General processing functions called from make_XXX
    __select_XXX(): Function to execute feature selection
    make_XXX(): Function to generate new column XXX
    trans_XXX(): Function to process existing column XXX
    """
    
    def __init__(self, scaler_x = None, scaler_y = None, not_use_cols = None, random_state = 42):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y # in classifier, scaler -> encoder
        self.not_use_cols = not_use_cols
        self.random_state = random_state
    
    def __set_info(self, X, y, train_id, test_id):
        self.X = X
        self.y = y
        self.train_id = train_id
        self.test_id  = test_id
        self.columns = self.X.columns
    
    def __missing_indicator_encoding(self):
        null_cols = self.X.loc[:, self.X.isna().any()].columns.to_list()
        new_cols  = [col + "_nan" for col in null_cols]
        self.X[new_cols] = MissingIndicator().fit_transform(self.X[null_cols])
    
    def __frequency_encoding(self, columns):
        new_cols = [col + "_freq" for col in columns]
        for i in range(len(columns)):
            freq = self.X[columns[i]].value_counts().to_frame(new_cols[i])
            self.X = pd.merge(self.X, freq, left_on = columns[i], how = "left", right_index = True)
    
    def __label_encoding(self, columns):
        enc = OrdinalEncoder()
        self.X[columns] = enc.fit_transform(self.X[columns].values)
    
    def __bool_to_int_encoding(self):
        cols = self.X.select_dtypes(include = bool).columns.to_list()
        self.X[cols] = self.X.loc[:, cols].astype(int)
    
    def __unordered_category_encoding(self, columns):
        self.X[columns] = self.X[columns].astype("category")
    
    def __select_columns(self):
        self.columns = self.X.columns
        if self.not_use_cols: 
            self.columns = self.columns.drop(self.not_use_cols)
        self.X = self.X[self.columns]
    
    def __target_encoding(self, columns):
        new_cols = [col + "_target" for col in columns]
        target_encoder = ce.LeaveOneOutEncoder(cols = columns, random_state = self.random_state)
        X_train = self.X.loc[self.train_id]
        y_train = self.y.loc[self.train_id]
        X_test  = self.X.loc[self.test_id]
        self.X.loc[self.train_id, new_cols] = target_encoder.fit_transform(X_train, y_train)[columns].values
        self.X.loc[self.test_id, new_cols] = target_encoder.transform(X_test)[columns].values
    
    def __binning_encoding(self, col, nbins, qcut = True):
        if qcut: 
            self.X[col] = pd.qcut(self.X[col], nbins, labels = [i + 1 for i in range(nbins)]).astype(float)
        else:
            self.X[col] = pd.qcut(self.X[col], nbins, labels = [i + 1 for i in range(nbins)]).astype(float)
    
    def make_lname(self):
        new_col = "lname"
        def extract_lname(data):
            name_df = data["name"].str.split("(", expand = True)[0]
            name_df = name_df.str.split(".", expand = True)[1]
            name_df = name_df.str.strip()
            return name_df.map(lambda x: x.split(" ")[-1])
        
        self.X[new_col] = extract_lname(self.X)
        self.__frequency_encoding([new_col])
        self.X.loc[self.X.lname_freq == 1, new_col] = "alone"
        self.X.loc[self.X.lname_freq.isnull(), new_col] = "alone"
    
    def make_alias(self):
        new_col = "alias"
        name_df = self.X["name"].str.split("(", expand = True)
        self.X[new_col] = name_df[1].notna()
    
    def make_ticket_freq(self):
        self.__frequency_encoding(["ticket"])
    
    def make_cabin_floor(self):
        self.X["cabin_floor"] = self.X.cabin.str[0]
        self.X.loc[self.X.cabin_floor == "T", "cabin_floor"] = "A"
        self.X.loc[self.X.cabin_floor == "G", "cabin_floor"] = "F"
        self.X.cabin_floor.fillna("M", inplace = True)
    
    def make_cabin_freq(self):
        self.__frequency_encoding(["cabin"])
    
    def make_cabin_floor_freq(self):
        self.__frequency_encoding(["cabin_floor"])
    
    def make_lname_target(self):
        cols = ["lname"]
        self.__target_encoding(cols)
        self.X.loc[self.X.lname_freq < 5, "lname_target"] = self.y.loc[self.train_id].mean()
        
    def make_family_size(self):
        new_col = "family_size"
        self.X[new_col] = self.X.parch + self.X.sibsp + 1
        
    def make_title(self):
        new_col = "title"
        name_df = self.X.name.str.split(",", expand = True)[1]
        name_df = name_df.str.strip()
        self.X[new_col] = name_df.str.split(".", expand = True)[0]
    
    def make_family_fare(self):
        self.X["family_fare"] = self.X["fare"] * self.X["family_size"]
    
    def make_family_age(self):
        self.X["family_age"] = self.X["age"] * self.X["family_size"]
    
    def make_fare_per_age(self):
        self.X["fare_per_age"] = self.X["fare"] / self.X["age"]
    
    def trans_bool_to_int(self):
        self.__bool_to_int_encoding()
    
    def trans_category(self):
        cat_cols = self.X.select_dtypes(include = "object").columns.to_list()
        unordered_cols = ['title', 'embarked', 'cabin_floor']
        self.__label_encoding(cat_cols)
        self.__unordered_category_encoding(unordered_cols)
    
    def trans_fare_binning(self):
        col= "fare"
        self.__binning_encoding(col, nbins = 20, qcut = False)
    
    def trans_age_binning(self):
        col= "age"
        self.__binning_encoding(col, nbins = 10, qcut = False)
    
    def fit_transform(self, X, y, train_id, test_id):
        self.__set_info(X, y, train_id, test_id)
        transformers = [self.__missing_indicator_encoding, self.make_lname, 
                        self.make_alias, self.make_ticket_freq, 
                        self.make_cabin_floor, self.make_cabin_freq, 
                        self.make_cabin_floor_freq, self.make_lname_target, 
                        self.make_family_size, self.make_title, 
                        self.trans_bool_to_int,
                        self.trans_fare_binning, self.trans_age_binning, 
                        self.trans_category, 
                        self.make_family_fare, self.make_family_age, 
                        self.make_fare_per_age, 
                        self.__select_columns
                       ]
        for transformer in transformers: 
            transformer()
        print(self.X.info())
        return self.X, self.y