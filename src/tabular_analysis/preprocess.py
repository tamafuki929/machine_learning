
# %% drop na and use only numerical columns
def make_baseline_dataset(data):
    data = data.select_dtypes(include = np.number)
    data.dropna(axis = 1)
    return data