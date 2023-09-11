import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None
def get_estimated_price(location, size, total_sqft, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = size
    x[1] = total_sqft
    x[2] = bath
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)
def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __locations
    global __data_columns
    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are total sqft ,bath, size

    global __model
    if __model is None:
        with open("./artifacts/banglore_home_prices_model.pickle", 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('location_1st Phase JP Nagar', 2, 1000, 2))
    print(get_estimated_price('location_1st Phase JP Nagar', 3, 1000, 2))
    print(get_estimated_price('location_Whitefield', 3, 1100, 2))  # other location
    print(get_estimated_price('location_7th Phase JP Nagar', 4, 1500, 4))
